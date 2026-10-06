"""Train and evaluate TickNet-L on CIFAR-10 and CIFAR-100 with SGD and Adam."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cifar_data import NUM_CLASSES, build_cifar_loaders, normalize_dataset_name, seed_everything
from models.model_profile import profile_model
from models.ticknet_l import ARCHITECTURE_REVISION, build_ticknet_l
from models.TickNet import build_TickNet

TRAINER_REVISION = "cifar-trainer-v2"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Execute one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            if logits.ndim != 2 or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid or non-finite logits")
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered during training")

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.numel()
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("Cannot evaluate an empty loader")

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct / total,
        "samples": total,
    }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_classes: int,
) -> Dict[str, Any]:
    """Comprehensive test evaluation: metrics, confusion matrix, macro F1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    loss_sum = 0.0
    correct = 0
    total = 0
    predictions_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if logits.shape != (labels.numel(), num_classes) or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid or non-finite evaluation logits")
            sample_losses = nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = sample_losses.mean()
            preds = logits.argmax(dim=1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")

            loss_sum += sample_losses.double().sum().item()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred, sample_loss in zip(labels.cpu().tolist(), preds.cpu().tolist(), sample_losses.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({"sample_index": len(predictions_rows), "target": target,
                                         "prediction": pred, "negative_log_likelihood": sample_loss})

    if total == 0:
        raise ValueError("Cannot evaluate an empty loader")
    top1 = 100.0 * correct / total
    loss_avg = loss_sum / total

    # Macro F1 computation
    f1_sum = 0.0
    for i in range(num_classes):
        tp = matrix[i][i]
        fp_plus_fn = sum(matrix[i]) + sum(matrix[r][i] for r in range(num_classes))
        f1_sum += (2.0 * tp) / max(1, fp_plus_fn)
    macro_f1 = f1_sum / num_classes

    result = {
        "top1": top1,
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "samples": total,
        "correct": correct,
    }

    # Save artifacts
    from models.cifar_experiment import atomic_json
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "test_metrics.json", result)

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(matrix)

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_index", "target", "prediction", "negative_log_likelihood"))
        writer.writeheader()
        writer.writerows(predictions_rows)

    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file")
    parser.add_argument("--model", choices=("l", "basic"), default=None, help="Model architecture: l or basic")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--nesterov", action="store_true", default=True, help="Enable Nesterov momentum for SGD")
    parser.add_argument("--no-nesterov", dest="nesterov", action="store_false", help="Disable Nesterov momentum for SGD")
    parser.add_argument("--cutout", action="store_true", default=True, help="Enable Cutout data augmentation")
    parser.add_argument("--no-cutout", dest="cutout", action="store_false", help="Disable Cutout data augmentation")
    parser.add_argument("--cutout-length", type=int, default=16, help="Cutout length in pixels")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Beta1 for Adam")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Beta2 for Adam")
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-download", dest="download", action="store_false", default=True)
    parser.add_argument("--skip-test", action="store_true", help="Train and select by validation only")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Path to last.pt checkpoint to resume")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop after specific epoch (smoke testing)")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint on test set")
    parser.add_argument("--allow-eval-source-change", action="store_true",
                        help="Explicitly evaluate compatible weights after a source-only refactor; never permits resume")

    # If --config is passed, read JSON defaults
    raw_args, _ = parser.parse_known_args(argv)
    if raw_args.config:
        config_path = raw_args.config.resolve()
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        try:
            defaults = json.loads(config_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            parser.error(f"Invalid config: {error}")
        allowed = {action.dest for action in parser._actions} - {"help", "config"}
        if not isinstance(defaults, dict) or set(defaults) - allowed:
            parser.error("Config must be an object with known CLI keys")
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args._dataset_explicit = args.dataset is not None
    args._model_explicit = args.model is not None
    args.dataset = args.dataset or "cifar10"
    args.model = args.model or "l"
    for key, choices in (("model", ("l", "basic")), ("dataset", ("cifar10", "cifar100")),
                         ("optimizer", ("sgd", "adam"))):
        if getattr(args, key) not in choices:
            parser.error(f"Invalid {key}")
    for key in ("epochs", "batch_size", "seed", "num_workers", "threads", "cutout_length"):
        value = getattr(args, key)
        minimum = 0 if key in ("seed", "num_workers", "cutout_length") else 1
        if type(value) is not int or value < minimum or (key == "seed" and value >= 2**32 - 2):
            parser.error(f"Invalid integer {key}")
    for key in ("learning_rate", "momentum", "adam_beta1", "adam_beta2", "adam_eps",
                "weight_decay", "val_fraction", "eta_min"):
        value = getattr(args, key)
        if type(value) not in (float, int) or not math.isfinite(value):
            parser.error(f"Invalid finite number {key}")
    for key in ("nesterov", "cutout", "download", "skip_test", "allow_eval_source_change"):
        if type(getattr(args, key)) is not bool:
            parser.error(f"Invalid boolean {key}")
    for key in ("data_root", "output_dir", "resume", "evaluate"):
        value = getattr(args, key)
        if value is not None:
            if not isinstance(value, (str, Path)):
                parser.error(f"Invalid path {key}")
            setattr(args, key, Path(value))
    if not 0 <= args.momentum < 1 or not all(0 <= b < 1 for b in (args.adam_beta1, args.adam_beta2)):
        parser.error("Momentum and Adam betas must be in [0, 1)")
    if args.weight_decay < 0 or args.adam_eps <= 0 or not 0 <= args.eta_min <= args.learning_rate:
        parser.error("Invalid weight decay, epsilon, or minimum learning rate")
    if not 0 < args.val_fraction < 1:
        parser.error("Training requires val-fraction in (0, 1)")
    if args.stop_after_epoch is not None and (type(args.stop_after_epoch) is not int or
                                             not 1 <= args.stop_after_epoch <= args.epochs):
        parser.error("stop-after-epoch must be between 1 and epochs")
    if args.resume and args.evaluate:
        parser.error("resume and evaluate are mutually exclusive")
    if args.allow_eval_source_change and not args.evaluate:
        parser.error("--allow-eval-source-change requires --evaluate; resume remains source-strict")
    if args.allow_eval_source_change and args.output_dir is None:
        parser.error("--allow-eval-source-change requires a fresh --output-dir")

    if args.evaluate is None and args.output_dir is None:
        parser.error("--output-dir is required when training")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive integers")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    return args


def build_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    if args.optimizer == "sgd":
        use_nesterov = bool(getattr(args, "nesterov", True)) and args.momentum > 0
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=use_nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    return optimizer, scheduler


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from models.cifar_training import execute
    return execute(parse_args(argv), build_cifar_loaders, build_ticknet_l, build_TickNet,
                   run_epoch, evaluate, profile_model)


if __name__ == "__main__":
    main()
