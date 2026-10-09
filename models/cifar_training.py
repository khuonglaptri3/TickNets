"""Epoch-boundary recovery and validation-selected evaluation for CIFAR."""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import torch
import torchvision
import numpy as np
import PIL

from .cifar_data import CIFAR_STATS, NUM_CLASSES, seed_everything
from .cifar_experiment import (RECIPE_FIELDS, atomic_checkpoint, atomic_json, capture_rng,
                              cpu_weights, data_evidence, export_history, file_sha256,
                              restore_rng, source_evidence, verify_artifact_hashes, write_completion)
from .ticknet_l import ARCHITECTURE_REVISION


def _check_checkpoint(checkpoint, evidence, *, allow_source_change=False):
    from train_cifar import TRAINER_REVISION
    config = checkpoint.get("config", {})
    if config.get("trainer_revision") != TRAINER_REVISION:
        raise ValueError("Checkpoint trainer revision is incompatible; restart with the reviewed trainer")
    revision = "ticknet-basic-author" if config.get("model") == "basic" else ARCHITECTURE_REVISION
    if config.get("architecture_revision") != revision:
        raise ValueError("Checkpoint architecture revision changed")
    if config.get("source_sha256") != evidence["source_sha256"] and not allow_source_change:
        raise ValueError("Source code changed since the checkpoint")


def _export_committed(output_dir, checkpoint):
    """last.pt is the commit record; reconstruct dependent artifacts after interruption."""
    atomic_json(output_dir / "config.json", checkpoint["config"])
    export_history(output_dir / "epochs.csv", checkpoint["history"])
    best = checkpoint["best_checkpoint"]
    if best is None:
        raise ValueError("Committed checkpoint has no validation-selected weights")
    atomic_checkpoint(output_dir / "best_val.pt", best)


def execute(args, build_loaders, build_l, build_basic, run_epoch, evaluate, profile_model):
    from train_cifar import TRAINER_REVISION, build_optimizer_and_scheduler

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    # Disable reduced precision paths; deterministic algorithms fail explicitly if unsupported.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    evidence = source_evidence()
    checkpoint_path = args.resume or args.evaluate
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(Path(checkpoint_path).resolve(), map_location="cpu", weights_only=False)
        _check_checkpoint(checkpoint, evidence,
                          allow_source_change=bool(args.evaluate and args.allow_eval_source_change))
    if args.evaluate:
        recorded = checkpoint["config"]
        for name in ("dataset", "model"):
            if getattr(args, "_" + name + "_explicit") and getattr(args, name) != recorded[name]:
                raise ValueError(f"Evaluation {name} differs from the checkpoint")
            setattr(args, name, recorded[name])

    num_classes = NUM_CLASSES[args.dataset]
    effective_nesterov = args.optimizer == "sgd" and args.nesterov and args.momentum > 0
    runtime = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "numpy": np.__version__, "pillow": PIL.__version__,
               "cuda": torch.version.cuda if device.type == "cuda" else None,
               "device_type": device.type,
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    recipe = {key: getattr(args, key) for key in RECIPE_FIELDS}
    recipe["nesterov"] = effective_nesterov
    if args.resume:
        recorded = checkpoint["config"]
        for key, value in recipe.items():
            if recorded.get(key) != value:
                raise ValueError(f"Resume recipe changed: {key}")
        if recorded.get("runtime") != runtime:
            raise ValueError("Resume runtime/GPU changed; exact recovery needs the same runtime")
        if not all(key in checkpoint for key in ("rng_state", "best_checkpoint", "history")):
            raise ValueError("Checkpoint is missing recovery state")
        if type(checkpoint.get("epoch")) is not int or not 1 <= checkpoint["epoch"] <= args.epochs:
            raise ValueError("Checkpoint epoch is outside the configured training budget")
        if len(checkpoint["history"]) != checkpoint["epoch"] or [r["epoch"] for r in checkpoint["history"]] != list(range(1, checkpoint["epoch"] + 1)):
            raise ValueError("Checkpoint history is incomplete")
        if any(not math.isfinite(float(row[key])) for row in checkpoint["history"]
               for key in ("train_loss", "train_top1", "val_loss", "val_top1", "learning_rate")):
            raise ValueError("Checkpoint history contains non-finite metrics")
        selected = max(checkpoint["history"], key=lambda row: (row["val_top1"], -row["val_loss"]))
        best_saved = checkpoint["best_checkpoint"]
        if (not isinstance(best_saved, dict) or best_saved.get("epoch") != selected["epoch"] or
                best_saved.get("best_validation", {}).get("top1") != selected["val_top1"] or
                best_saved.get("best_validation", {}).get("loss") != selected["val_loss"]):
            raise ValueError("Checkpoint best weights metadata disagrees with validation history")
        if checkpoint["scheduler_state_dict"].get("last_epoch") != checkpoint["epoch"]:
            raise ValueError("Checkpoint scheduler is not at the committed epoch")
        if args.stop_after_epoch is not None and args.stop_after_epoch < checkpoint["epoch"]:
            raise ValueError("Stop epoch precedes the saved epoch")

    output_dir = (Path(args.output_dir).resolve() if args.output_dir else
                  Path(args.evaluate).resolve().parent)
    if args.allow_eval_source_change and output_dir.exists():
        raise FileExistsError("Source-change evaluation requires a fresh output directory; preserve archived artifacts")
    if not args.evaluate and output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output exists; choose a fresh directory or resume: {output_dir}")
    if args.resume and (output_dir / "last.pt").exists() and file_sha256(output_dir / "last.pt") != file_sha256(args.resume):
        raise ValueError("Destination contains a different checkpoint")

    loaders = build_loaders(args.data_root, args.dataset, batch_size=args.batch_size,
                           val_fraction=0.0 if args.evaluate else args.val_fraction,
                           seed=args.seed, num_workers=args.num_workers,
                           pin_memory=device.type == "cuda", cutout=False if args.evaluate else args.cutout,
                           cutout_length=args.cutout_length, download=args.download)
    train_loader, val_loader, test_loader = loaders
    datasets = data_evidence(loaders)
    if checkpoint:
        recorded_data = checkpoint["config"].get("data_evidence")
        expected_data = {"test": datasets["test"]} if args.evaluate else datasets
        if recorded_data is None or any(recorded_data.get(k) != v for k, v in expected_data.items()):
            raise ValueError("Dataset pixels, labels, or split changed since checkpoint creation")
    if not args.evaluate and (val_loader is None or len(val_loader.dataset) == 0):
        raise ValueError("Validation is required for checkpoint selection")

    model = (build_basic(num_classes=num_classes, typesize="basic", cifar=True)
             if args.model == "basic" else build_l(num_classes=num_classes, cifar=True))
    complexity = profile_model(model, 32, cross_check=True)
    if not complexity["within_exam_limits"]:
        raise ValueError("Model exceeds exam limits (6M params or 1G FLOPs)")
    model = model.to(device)
    if args.evaluate:
        model.load_state_dict(checkpoint["model_state_dict"])
        if (output_dir / "completion.json").exists():
            verify_artifact_hashes(output_dir)
            cached = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
            if cached["checkpoint_sha256"] == file_sha256(args.evaluate):
                return cached
            raise ValueError("Choose a fresh evaluation output directory; completed artifacts are immutable")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate(model, test_loader, device, output_dir, num_classes)
        result.update(checkpoint_epoch=checkpoint["epoch"], checkpoint_sha256=file_sha256(args.evaluate),
                      dataset=args.dataset, model=args.model,
                      test_data_sha256=datasets["test"]["sha256"],
                      training_source_sha256=checkpoint["config"]["source_sha256"],
                      evaluation_source_sha256=evidence["source_sha256"],
                      source_change_accepted=bool(args.allow_eval_source_change))
        atomic_json(output_dir / "test_metrics.json", result)
        return result

    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    criterion = torch.nn.CrossEntropyLoss()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if not key.startswith("_")}
    config.update(recipe, **evidence, trainer_revision=TRAINER_REVISION,
                  architecture_revision="ticknet-basic-author" if args.model == "basic" else ARCHITECTURE_REVISION,
                  runtime=runtime, python_version=platform.python_version(), num_classes=num_classes,
                  learnable_parameters=complexity["learnable_parameters"], flops_forward=complexity["flops"],
                  gflops_forward=complexity["flops"] / 1e9, within_exam_limits=True,
                  counting_convention=complexity["counting_convention"],
                  excluded_operations=complexity["excluded_operations"], data_evidence=datasets,
                  normalization={"mean": list(CIFAR_STATS[args.dataset][0]),
                                 "std": list(CIFAR_STATS[args.dataset][1])})
    history = []
    best = None
    start_epoch = 1
    if args.resume:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng(checkpoint["rng_state"], loaders)
        config = checkpoint["config"]
        history = checkpoint["history"]
        best = checkpoint["best_checkpoint"]
        start_epoch = checkpoint["epoch"] + 1
        # A completed run is immutable; rerunning a notebook verifies and reuses it.
        if checkpoint["epoch"] == args.epochs and (output_dir / "completion.json").exists():
            verify_artifact_hashes(output_dir)
            result = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
            return {"final_test": result, "best_validation": best["best_validation"], "output_dir": str(output_dir)}

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        atomic_checkpoint(output_dir / "last.pt", checkpoint)
        _export_committed(output_dir, checkpoint)
    else:
        atomic_json(output_dir / "config.json", config)
    stop_epoch = args.stop_after_epoch or args.epochs
    for epoch in range(start_epoch, stop_epoch + 1):
        lr = optimizer.param_groups[0]["lr"]
        trained = run_epoch(model, train_loader, criterion, device, optimizer)
        validated = run_epoch(model, val_loader, criterion, device)
        if best is None or (validated["top1"], -validated["loss"]) > (best["best_validation"]["top1"], -best["best_validation"]["loss"]):
            best = {"epoch": epoch, "model_state_dict": cpu_weights(model), "config": config,
                    "best_validation": {**validated, "epoch": epoch}}
        scheduler.step()
        history.append({"epoch": epoch, "learning_rate": lr, "train_loss": trained["loss"],
                        "train_top1": trained["top1"], "train_samples": trained["samples"],
                        "val_loss": validated["loss"], "val_top1": validated["top1"],
                        "val_samples": validated["samples"]})
        checkpoint = {"epoch": epoch, "model_state_dict": cpu_weights(model),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "config": config, "best_checkpoint": best, "history": history,
                      "rng_state": capture_rng(loaders)}
        # Commit first. A failed log/best export can be reconstructed from last.pt.
        atomic_checkpoint(output_dir / "last.pt", checkpoint)
        _export_committed(output_dir, checkpoint)
        print(f"Epoch {epoch:03d}/{args.epochs:03d} LR={lr:.6g} Train={trained['top1']:.2f}% Val={validated['top1']:.2f}%", flush=True)

    if checkpoint["epoch"] < args.epochs or args.skip_test:
        atomic_json(output_dir / "progress.json", {"status": "paused" if checkpoint["epoch"] < args.epochs else "trained",
                    "completed_epochs": checkpoint["epoch"], "target_epochs": args.epochs,
                    "best_validation": best["best_validation"]})
        return {"final_test": None, "best_validation": best["best_validation"], "output_dir": str(output_dir)}
    model.load_state_dict(best["model_state_dict"])
    result = evaluate(model, test_loader, device, output_dir, num_classes)
    result.update(checkpoint_epoch=best["epoch"], checkpoint_sha256=file_sha256(output_dir / "best_val.pt"),
                  dataset=args.dataset, model=args.model, test_data_sha256=datasets["test"]["sha256"])
    atomic_json(output_dir / "test_metrics.json", result)
    write_completion(output_dir, result)
    atomic_json(output_dir / "progress.json", {"status": "complete", "completed_epochs": args.epochs,
                "target_epochs": args.epochs, "best_validation": best["best_validation"]})
    print(f"Completed: {args.dataset} {args.optimizer}, test Top-1={result['top1']:.2f}%", flush=True)
    return {"final_test": result, "best_validation": best["best_validation"], "output_dir": str(output_dir)}
