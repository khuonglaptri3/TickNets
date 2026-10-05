"""Train and evaluate TickNet-L on CIFAR-10 and CIFAR-100 with SGD and Adam."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from models.cifar_data import NUM_CLASSES, build_cifar_loaders, normalize_dataset_name, seed_everything
from models.model_profile import profile_model
from models.ticknet_l import ARCHITECTURE_REVISION, build_ticknet_l

TRAINER_REVISION = "cifar-trainer-v1"


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
    predictions_rows: List[Dict[str, int]] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            loss_sum += loss.item() * labels.numel()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({"sample_index": len(predictions_rows), "target": target, "prediction": pred})

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
    (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(matrix)

    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("sample_index", "target", "prediction"))
        writer.writeheader()
        writer.writerows(predictions_rows)

    return result


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Beta1 for Adam")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Path to last.pt checkpoint to resume")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop after specific epoch (smoke testing)")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a saved checkpoint on test set")

    # If --config is passed, read JSON defaults
    raw_args, _ = parser.parse_known_args(argv)
    if raw_args.config:
        config_path = raw_args.config.resolve()
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)

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
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return optimizer, scheduler


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    seed_everything(args.seed)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_name = normalize_dataset_name(args.dataset)
    num_classes = NUM_CLASSES[dataset_name]

    # Evaluate-only mode
    if args.evaluate:
        ckpt_path = args.evaluate.resolve()
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_ticknet_l(num_classes=num_classes, cifar=True).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        out_dir = args.output_dir.resolve() if args.output_dir else ckpt_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        _, _, test_loader = build_cifar_loaders(
            args.data_root, dataset_name, batch_size=args.batch_size,
            val_fraction=0.0, seed=args.seed, num_workers=args.num_workers,
            pin_memory=device.type == "cuda"
        )
        result = evaluate(model, test_loader, device, out_dir, num_classes)
        print(f"[✓] Evaluated checkpoint {ckpt_path.name}: Top-1={result['top1']:.2f}%, Loss={result['loss']:.4f}, Macro-F1={result['macro_f1']:.4f}")
        return result

    # Training mode
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output directory exists; select a fresh directory or use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    train_loader, val_loader, test_loader = build_cifar_loaders(
        args.data_root, dataset_name, batch_size=args.batch_size,
        val_fraction=args.val_fraction, seed=args.seed,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    # Initialize model
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    complexity = profile_model(model, 32, cross_check=True)
    if not complexity["within_exam_limits"]:
        raise ValueError("Model exceeds exam limits (6M params or 1G FLOPs)")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    # Record config
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        trainer_revision=TRAINER_REVISION,
        architecture_revision=ARCHITECTURE_REVISION,
        num_classes=num_classes,
        learnable_parameters=complexity["learnable_parameters"],
        flops_forward=complexity["flops"],
        gflops_forward=complexity["flops"] / 1e9,
        within_exam_limits=complexity["within_exam_limits"],
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        torchvision_version=str(torchvision.__version__),
        device=str(device),
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    start_epoch = 1
    best_val_record: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []

    # Handle resume
    if args.resume:
        resume_ckpt = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        best_val_record = resume_ckpt.get("best_val_record")
        history = resume_ckpt.get("history", [])
        print(f"[*] Resumed training from epoch {start_epoch} (best val: {best_val_record})")

    stop_epoch = args.stop_after_epoch or args.epochs

    fields = ("epoch", "learning_rate", "train_loss", "train_top1", "train_samples", "val_loss", "val_top1", "val_samples")
    csv_mode = "a" if args.resume and (output_dir / "epochs.csv").is_file() else "w"

    with (output_dir / "epochs.csv").open(csv_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if csv_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, stop_epoch + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)

            val_metrics = None
            if val_loader is not None:
                val_metrics = run_epoch(model, val_loader, criterion, device)

            improved = False
            if val_metrics is not None:
                if best_val_record is None:
                    improved = True
                else:
                    curr_tuple = (val_metrics["top1"], -val_metrics["loss"])
                    best_tuple = (best_val_record["top1"], -best_val_record["loss"])
                    improved = curr_tuple > best_tuple

                if improved:
                    best_val_record = {**val_metrics, "epoch": epoch}
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "config": config,
                        "best_validation": best_val_record,
                    }, output_dir / "best_val.pt")

            scheduler.step()

            row = {
                "epoch": epoch,
                "learning_rate": lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_samples": train_metrics["samples"],
                "val_loss": val_metrics["loss"] if val_metrics else "",
                "val_top1": val_metrics["top1"] if val_metrics else "",
                "val_samples": val_metrics["samples"] if val_metrics else 0,
            }
            writer.writerow(row)
            f.flush()
            history.append(row)

            # Save last.pt for uninterrupted checkpointing
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "best_val_record": best_val_record,
                "history": history,
            }, output_dir / "last.pt")

            val_str = f" | Val Top-1: {val_metrics['top1']:.2f}%, Val Loss: {val_metrics['loss']:.4f}" if val_metrics else ""
            print(f"Epoch {epoch:03d}/{args.epochs:03d} [LR: {lr:.5f}] - Train Top-1: {train_metrics['top1']:.2f}%, Train Loss: {train_metrics['loss']:.4f}{val_str}", flush=True)

    # Final evaluation on test set using the best validation checkpoint (or last checkpoint)
    print("\n[*] Evaluating on official 10,000 test set...")
    best_weights_path = output_dir / "best_val.pt" if (output_dir / "best_val.pt").is_file() else output_dir / "last.pt"
    best_ckpt = torch.load(best_weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_results = evaluate(model, test_loader, device, output_dir, num_classes)
    print(f"\n[✓] Final Test Evaluation Completed:")
    print(f"    - Test Top-1 Accuracy: {test_results['top1']:.2f}%")
    print(f"    - Test Loss:           {test_results['loss']:.4f}")
    print(f"    - Test Macro F1:       {test_results['macro_f1']:.4f}")
    print(f"    - Output Directory:    {output_dir}")

    return {
        "final_test": test_results,
        "best_validation": best_val_record,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    main()
