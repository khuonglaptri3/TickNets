"""Controlled Mid experiments: select on validation, evaluate test explicitly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision

from models.mid_data import seed_everything
from models.mid_experiment_data import build_experiment_loaders
from models.mid_models import MODEL_REVISIONS, build_mid_model
from models.model_profile import profile_model
from models.ticknet_l_stage4 import ARCHITECTURE_REVISION as STAGE4_REVISION, build_ticknet_l_stage4

MIXING_METHODS = {}
EXPERIMENT_MODELS = {"l_stage4": (build_ticknet_l_stage4, STAGE4_REVISION)}
TRAINER_REVISION = "mid-experiment-v2"


def add_branch_arguments(parser):
    """Branches may add recipe flags without altering historical training."""


def training_criterion(args):
    return torch.nn.CrossEntropyLoss()


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    selected, _ = bootstrap.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--variant", choices=("Mid32", "Mid224"))
    parser.add_argument("--model", choices=("l", *EXPERIMENT_MODELS), default="l")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--mixing", choices=("none", *MIXING_METHODS), default="none")
    parser.add_argument("--mixing-alpha", type=float, default=0.2)
    parser.add_argument("--mixing-probability", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--evaluate", type=Path)
    add_branch_arguments(parser)
    if selected.config:
        try:
            defaults = json.loads(selected.config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(f"Cannot read JSON config: {error}")
        if not isinstance(defaults, dict):
            parser.error("config must be a JSON object")
        actions = {a.dest: a for a in parser._actions if a.dest != "help"}
        unknown = set(defaults) - set(actions)
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        for key, value in defaults.items():
            action = actions[key]
            optional = {"config", "output_dir", "resume", "evaluate", "stop_after_epoch", "variant"}
            if value is None:
                if key not in optional:
                    parser.error(f"config {key} cannot be null")
                continue
            if isinstance(action, argparse._StoreTrueAction) and not isinstance(value, bool):
                parser.error(f"config {key} must be boolean")
            if action.type is int and (isinstance(value, bool) or not isinstance(value, int)):
                parser.error(f"config {key} must be an integer")
            if action.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                parser.error(f"config {key} must be numeric")
            if (action.type is Path or (action.type is None and not isinstance(action, argparse._StoreTrueAction))) and not isinstance(value, str):
                parser.error(f"config {key} must be a string")
            if action.type and value is not None:
                try:
                    defaults[key] = action.type(value)
                except (TypeError, ValueError, OverflowError):
                    parser.error(f"Invalid config value for {key}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for action in parser._actions:
        if action.choices and getattr(args, action.dest) not in action.choices:
            parser.error(f"Invalid {action.dest}: {getattr(args, action.dest)}")
    if args.variant is None or args.output_dir is None:
        parser.error("--variant and --output-dir are required")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size, threads must be positive; workers nonnegative")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between 0 and 1")
    if not all(math.isfinite(v) for v in (args.learning_rate, args.momentum, args.weight_decay,
                                         args.mixing_alpha, args.mixing_probability)):
        parser.error("recipe values must be finite")
    if args.learning_rate <= 0 or args.momentum < 0 or args.weight_decay < 0:
        parser.error("Invalid SGD recipe")
    if args.mixing_alpha <= 0 or not 0 <= args.mixing_probability <= 1:
        parser.error("Invalid mixing alpha/probability")
    if args.resume and args.evaluate:
        parser.error("resume and evaluate are separate modes")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error("stop-after-epoch must be in [1, epochs]")
    return args


def run_epoch(model, loader, criterion, device, optimizer=None, *, mix_fn=None, alpha=0.2, probability=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels_b, lam = labels, 1.0
            if training and mix_fn is not None and torch.rand(()).item() < probability:
                images, labels, labels_b, lam = mix_fn(images, labels, alpha)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; experiment aborted")
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(1)
            correct += lam * (predictions == labels).sum().item() + (1 - lam) * (predictions == labels_b).sum().item()
            loss_sum += loss.item() * labels.numel()
            count += labels.numel()
    if not count:
        raise ValueError("Empty data loader")
    return {"loss": loss_sum / count, "top1": 100 * correct / count, "samples": count}


def rng_state(loaders):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loaders": [loader.generator.get_state() if loader else None for loader in loaders]}


def restore_rng(state, loaders):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for loader, saved in zip(loaders, state["loaders"]):
        if loader is not None and saved is not None:
            loader.generator.set_state(saved)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def build_model(name, variant):
    if name in EXPERIMENT_MODELS:
        builder, revision = EXPERIMENT_MODELS[name]
        return builder(num_classes=5, cifar=variant == "Mid32"), revision
    return build_mid_model(name, variant=variant), MODEL_REVISIONS[name]


def evaluate(model, loader, device, output):
    result = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), device)
    matrix = [[0] * 5 for _ in range(5)]
    rows = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            for target, prediction in zip(labels.tolist(), predictions):
                matrix[target][prediction] += 1
                index = len(rows)
                rows.append({"path": loader.dataset.samples[index][0], "target": target, "prediction": prediction})
    result["macro_f1"] = sum(2 * matrix[i][i] / max(1, sum(matrix[i]) + sum(row[i] for row in matrix)) for i in range(5)) / 5
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "target", "prediction"))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    output = args.output_dir.resolve()
    checkpoint = torch.load(args.evaluate or args.resume, map_location="cpu", weights_only=True) if (args.evaluate or args.resume) else None
    if checkpoint and checkpoint.get("config", {}).get("trainer_revision") != TRAINER_REVISION:
        raise ValueError("Checkpoint is not from this experimental trainer")
    if args.evaluate:
        recorded = checkpoint["config"]
        if args.variant != recorded["variant"]:
            raise ValueError("Checkpoint dataset variant does not match")
        args.model, args.seed = recorded["model"], recorded["seed"]
        args.split_seed, args.val_fraction = recorded["split_seed"], recorded["val_fraction"]
        args.full_train = recorded["full_train"]
    train, val, test, split = build_experiment_loaders(
        args.data_root, args.variant, batch_size=args.batch_size, seed=args.seed,
        split_seed=args.split_seed, val_fraction=args.val_fraction, num_workers=args.num_workers,
        augment=not args.no_augment, pin_memory=device.type == "cuda", full_train=args.full_train)
    model, revision = build_model(args.model, args.variant)
    complexity = profile_model(model, 32 if args.variant == "Mid32" else 224, cross_check=True)
    if not complexity["within_exam_limits"]:
        raise ValueError("Experiment exceeds 6M parameters or strict <1G FLOPs")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("config", "resume", "evaluate", "stop_after_epoch")}
    config.update(trainer_revision=TRAINER_REVISION, architecture_revision=revision,
                  train_samples=split["train_samples"], val_samples=split["val_samples"], test_samples=split["test_samples"],
                  class_to_idx=split["class_to_idx"], membership_sha256=split["membership_sha256"],
                  validation_sha256=split["validation_sha256"],
                  python_version=platform.python_version(), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__), optimizer="SGD", scheduler="CosineAnnealingLR",
                  test_policy="Explicit --evaluate only; never choose settings/checkpoints using test",
                  train_top1_definition="lambda-weighted accuracy against paired labels when mixing; hard-label otherwise",
                  complexity={k: v for k, v in complexity.items() if k != "layers"})
    source_root = Path(__file__).resolve().parent
    config["source_hashes"] = {
        str(path.relative_to(source_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(), *sorted((source_root / "models").glob("*.py"))]
    }
    try:
        config["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip()
        config["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        config["git_revision"] = None
        config["git_dirty"] = None
    manifest = args.data_root / "split_manifest.csv"
    config["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
    if checkpoint:
        for key in ("architecture_revision", "membership_sha256", "validation_sha256", "class_to_idx", "source_manifest_sha256"):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(f"Checkpoint {key} does not match current model/data")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if args.resume:
        ignored = {"output_dir", "data_root", "device", "threads", "git_revision", "git_dirty"}
        old = {k: v for k, v in checkpoint["config"].items() if k not in ignored}
        new = {k: v for k, v in config.items() if k not in ignored}
        if old != new:
            raise ValueError("Resume config/recipe differs from checkpoint")
        if args.resume.resolve().parent != output:
            raise ValueError("Resume in the original run directory")
        if checkpoint["role"] != "last":
            raise ValueError("Resume requires last.pt, not a validation-selected checkpoint")
    elif output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    model.to(device)
    if not args.resume:
        output.mkdir(parents=True)
        if args.evaluate:
            config["evaluation_checkpoint"] = {
                "path": str(args.evaluate.resolve()),
                "sha256": hashlib.sha256(args.evaluate.read_bytes()).hexdigest(),
                "epoch": checkpoint["epoch"], "role": checkpoint["role"],
                "training_config": checkpoint["config"],
            }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "validation_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    if args.evaluate:
        return evaluate(model, test, device, output)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fields = ("epoch", "learning_rate", "train_loss", "train_top1", "train_samples", "val_loss", "val_top1", "val_samples")
    start, best, best_checkpoint, history = 1, None, None, []
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_validation"]
        history, best_checkpoint = checkpoint["epoch_history"], checkpoint["best_checkpoint"]
        if len(history) != checkpoint["epoch"] or [row["epoch"] for row in history] != list(range(1, checkpoint["epoch"] + 1)):
            raise ValueError("Checkpoint epoch history is inconsistent")
        if val is not None and (best_checkpoint is None or best_checkpoint["epoch"] != best["epoch"]):
            raise ValueError("Checkpoint validation-selected model is inconsistent")
        # last.pt is authoritative after any interrupted CSV/best artifact write.
        with (output / "epochs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        if best_checkpoint is not None:
            save_checkpoint(output / "best_val.pt", best_checkpoint)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        restore_rng(checkpoint["rng_state"], (train, val, test))
    stop = args.stop_after_epoch or args.epochs
    if stop < start:
        if args.resume and checkpoint["epoch"] == args.epochs and stop == args.epochs:
            return {"epoch": stop, "best_validation": best, "output_dir": str(output), "recovered_completed_run": True}
        raise ValueError("No remaining epochs under the selected stop/total schedule")
    with (output / "epochs.csv").open("a" if args.resume else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for epoch in range(start, stop + 1):
            lr = optimizer.param_groups[0]["lr"]
            training = run_epoch(model, train, training_criterion(args), device, optimizer,
                                 mix_fn=MIXING_METHODS.get(args.mixing), alpha=args.mixing_alpha, probability=args.mixing_probability)
            validation = run_epoch(model, val, torch.nn.CrossEntropyLoss(), device) if val is not None else None
            improved = validation is not None and (best is None or (validation["top1"], -validation["loss"]) > (best["top1"], -best["loss"]))
            if improved:
                best = {**validation, "epoch": epoch}
                best_checkpoint = {
                    "epoch": epoch, "config": config, "role": "best_val", "best_validation": best,
                    "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                }
            row = {"epoch": epoch, "learning_rate": lr, "train_loss": training["loss"], "train_top1": training["top1"],
                   "train_samples": training["samples"], "val_loss": validation["loss"] if validation else "",
                   "val_top1": validation["top1"] if validation else "", "val_samples": validation["samples"] if validation else 0}
            scheduler.step()
            history.append(row)
            state = {"epoch": epoch, "config": config, "role": "last", "best_validation": best,
                     "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(), "rng_state": rng_state((train, val, test)),
                     "best_checkpoint": best_checkpoint, "epoch_history": history}
            save_checkpoint(output / "last.pt", state)
            if improved:
                save_checkpoint(output / "best_val.pt", best_checkpoint)
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
    return {"epoch": stop, "best_validation": best, "output_dir": str(output)}


if __name__ == "__main__":
    print(json.dumps(main()), flush=True)
