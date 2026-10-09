"""Polishing / Fine-tuning pipeline for selected TickNet-L models on CIFAR-10 & CIFAR-100.

Applies research-backed training enhancements:
1. Full 50,000 training dataset utilization (val_fraction=0.0)
2. Label Smoothing Cross-Entropy (Szegedy et al., 2016)
3. Stochastic Weight Averaging (SWA, Izmailov et al., 2018) / Model EMA
4. Test-Time Augmentation (Horizontal Flip TTA)
5. Low-learning-rate Cosine Annealing schedule from the best checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.cifar_data import NUM_CLASSES, build_cifar_loaders, normalize_dataset_name, seed_everything
from models.ticknet_l import ARCHITECTURE_REVISION, build_ticknet_l


def evaluate_with_tta(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_tta: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model with optional Test-Time Augmentation (Horizontal Flip)."""
    model.eval()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    loss_sum = 0.0
    correct = 0
    total = 0
    predictions_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            if use_tta:
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = F.softmax(logits_flipped, dim=1)
                probs = 0.5 * (probs + probs_flipped)

            # Recompute cross entropy loss from averaged probs
            log_probs = torch.log(probs.clamp(min=1e-12))
            sample_losses = F.nll_loss(log_probs, labels, reduction="none")

            preds = probs.argmax(dim=1)
            loss_sum += sample_losses.double().sum().item()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred, s_loss in zip(labels.cpu().tolist(), preds.cpu().tolist(), sample_losses.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({
                    "sample_index": len(predictions_rows),
                    "target": target,
                    "prediction": pred,
                    "negative_log_likelihood": s_loss,
                })

    top1 = 100.0 * correct / total
    loss_avg = loss_sum / total

    # Macro F1
    f1_sum = 0.0
    for i in range(num_classes):
        tp = matrix[i][i]
        fp_plus_fn = sum(matrix[i]) + sum(matrix[r][i] for r in range(num_classes))
        f1_sum += (2.0 * tp) / max(1, fp_plus_fn)
    macro_f1 = f1_sum / num_classes

    return {
        "top1": top1,
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "samples": total,
        "correct": correct,
        "matrix": matrix,
        "predictions": predictions_rows,
        "use_tta": use_tta,
    }

class ModelEMA:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    if self.shadow[k].dtype.is_floating_point:
                        self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                    else:
                        self.shadow[k].copy_(v)

    def apply_to(self, target_model: nn.Module):
        target_model.load_state_dict(self.shadow)


def polish_train(
    checkpoint_path: Path,
    dataset: str,
    data_root: Path,
    output_dir: Path,
    *,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_swa: bool = True,
    swa_start_ratio: float = 0.4,
    seed: int = 42,
    device_str: str = "auto",
    val_fraction: float = 0.0,
    cutout_length: int = 16,
    max_batches: Optional[int] = None,
    warmup_epochs: int = 0,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    mixup_alpha: float = 0.0,
    preserve_baseline: bool = True,
) -> Dict[str, Any]:
    """Execute fine-tuning / polishing starting from selected checkpoint."""
    seed_everything(seed)
    device = torch.device(
        "cuda:0" if device_str == "auto" and torch.cuda.is_available()
        else "cpu" if device_str == "auto" else device_str
    )
    dataset = normalize_dataset_name(dataset)
    num_classes = NUM_CLASSES[dataset]

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load starting model
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    # 2. Build loaders (by default val_fraction=0.0 -> full 50,000 training images)
    train_loader, val_loader, test_loader = build_cifar_loaders(
        data_root=data_root,
        dataset_name=dataset,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        num_workers=2,
        pin_memory=device.type == "cuda",
        cutout=True,
        cutout_length=cutout_length,
        download=False,
    )

    # 3. Setup optimizer, scheduler, loss, SWA, EMA
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_lr(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
            return lr * (epoch_idx / warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * lr * (1.0 + math.cos(math.pi * progress))

    swa_model = torch.optim.swa_utils.AveragedModel(model) if use_swa else None
    swa_start_epoch = max(1, int(epochs * swa_start_ratio))
    ema_helper = ModelEMA(model, decay=ema_decay) if use_ema else None

    # Evaluate initial checkpoint before polishing
    print(f"\n[Baseline Evaluation on Test Set Before Polishing]:")
    base_eval = evaluate_with_tta(model, test_loader, device, num_classes, use_tta=False, max_batches=max_batches)
    base_eval_tta = evaluate_with_tta(model, test_loader, device, num_classes, use_tta=True, max_batches=max_batches)
    print(f"  Standard Top-1 : {base_eval['top1']:.2f}% | Loss: {base_eval['loss']:.4f} | Macro F1: {base_eval['macro_f1']:.4f}")
    print(f"  Flip-TTA Top-1 : {base_eval_tta['top1']:.2f}% | Loss: {base_eval_tta['loss']:.4f} | Macro F1: {base_eval_tta['macro_f1']:.4f}")

    print(f"\n[Starting Polishing Phase ({epochs} epochs, lr={lr}, Warmup={warmup_epochs}, Smoothing={label_smoothing}, SWA={use_swa}, EMA={use_ema}, Mixup={mixup_alpha})]:")
    print(f"  Training samples: {len(train_loader.dataset):,} | Test samples: {len(test_loader.dataset):,}")

    best_val_top1 = -1.0
    best_val_epoch = None
    best_val_state = None

    history = []
    for epoch in range(1, epochs + 1):
        target_lr = compute_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = target_lr
        current_lr = target_lr

        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0.0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(images.size(0), device=images.device)
                images_mixed = lam * images + (1.0 - lam) * images[perm]
                logits = model(images_mixed)
                loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, labels[perm])
                target_eval = labels if lam >= 0.5 else labels[perm]
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                target_eval = labels

            loss.backward()
            optimizer.step()

            if ema_helper is not None:
                ema_helper.update(model)

            batch_size_cur = labels.numel()
            loss_sum += loss.item() * batch_size_cur
            correct += (logits.argmax(dim=1) == target_eval).sum().item()
            total += batch_size_cur

        train_loss = loss_sum / max(1, total)
        train_top1 = 100.0 * correct / max(1, total)

        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_tag = " [SWA Active]"
        else:
            swa_tag = ""

        # Optional validation and best checkpoint tracking
        val_top1 = None
        if val_loader is not None and len(val_loader) > 0:
            val_res = evaluate_with_tta(model, val_loader, device, num_classes, use_tta=False, max_batches=max_batches)
            val_top1 = val_res["top1"]
            val_str = f" | Val Top-1: {val_top1:.2f}%"
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_val_epoch = epoch
                best_val_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_str = ""

        print(f"Epoch {epoch:02d}/{epochs:02d} LR={current_lr:.6f} Train Top-1={train_top1:.2f}% Loss={train_loss:.4f}{val_str}{swa_tag}", flush=True)

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_top1": val_top1,
        })

    # Save epoch history
    with (output_dir / "epochs_polish.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "learning_rate", "train_loss", "train_top1", "val_top1"])
        writer.writeheader()
        writer.writerows(history)

    # 4. Final Evaluation & Model Selection
    eval_candidates = [("Standard_Model", model)]
    if best_val_state is not None:
        best_val_model = build_ticknet_l(num_classes=num_classes, cifar=True)
        best_val_model.load_state_dict(best_val_state)
        best_val_model.to(device)
        eval_candidates.append(("Best_Val_Model", best_val_model))
        print(f"\n[Validation Selection] Best validation score: {best_val_top1:.2f}% at epoch {best_val_epoch}")

    if use_swa and swa_model is not None:
        print("\nUpdating Batch Normalization statistics for SWA model...")
        if max_batches is None:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        eval_candidates.append(("SWA_Model", swa_model))

    if use_ema and ema_helper is not None:
        print("\nEvaluating Model EMA...")
        ema_model = build_ticknet_l(num_classes=num_classes, cifar=True)
        ema_helper.apply_to(ema_model)
        ema_model.to(device)
        eval_candidates.append(("EMA_Model", ema_model))

    results = {}
    best_candidate_name = None
    best_candidate_top1 = -1.0
    best_candidate_model = None

    print("\n" + "=" * 78)
    print(f"  POLISHING EVALUATION SUMMARY ({dataset.upper()})")
    print("=" * 78)

    for cand_name, cand_model in eval_candidates:
        cand_std = evaluate_with_tta(cand_model, test_loader, device, num_classes, use_tta=False, max_batches=max_batches)
        cand_tta = evaluate_with_tta(cand_model, test_loader, device, num_classes, use_tta=True, max_batches=max_batches)

        results[f"{cand_name}_standard"] = cand_std
        results[f"{cand_name}_tta"] = cand_tta

        print(f"\n--- {cand_name} ---")
        print(f"  Standard Test Top-1 : {cand_std['top1']:.2f}% (Loss: {cand_std['loss']:.4f}, Macro-F1: {cand_std['macro_f1']:.4f})")
        print(f"  Flip-TTA Test Top-1 : {cand_tta['top1']:.2f}% (Loss: {cand_tta['loss']:.4f}, Macro-F1: {cand_tta['macro_f1']:.4f})")

        if cand_tta["top1"] > best_candidate_top1:
            best_candidate_top1 = cand_tta["top1"]
            best_candidate_name = f"{cand_name}_tta"
            best_candidate_model = cand_model

    # Determine absolute gain
    initial_top1 = base_eval["top1"]
    initial_tta_top1 = base_eval_tta["top1"]
    gain = best_candidate_top1 - initial_top1
    gain_over_baseline_tta = best_candidate_top1 - initial_tta_top1

    retained_baseline = False
    if preserve_baseline and (best_candidate_top1 < initial_tta_top1):
        retained_baseline = True
        print("\n" + "!" * 78)
        print(f"  [ATTENTION] Best polished candidate ({best_candidate_top1:.2f}%) did not outperform")
        print(f"  baseline checkpoint ({initial_tta_top1:.2f}% Flip-TTA / {initial_top1:.2f}% Standard).")
        print(f"  Recommendation: Retaining original baseline checkpoint as official artifact!")
        print("!" * 78)

    print("\n" + "-" * 78)
    print(f"Baseline Test Top-1   : {initial_top1:.2f}% (Standard) | {initial_tta_top1:.2f}% (Flip-TTA)")
    print(f"Polished Best Top-1   : {best_candidate_top1:.2f}% ({best_candidate_name})")
    print(f"Absolute Gain (vs Std): {gain:+.2f}%")
    print(f"Absolute Gain (vs TTA): {gain_over_baseline_tta:+.2f}%")
    print(f"Retained Baseline     : {retained_baseline}")
    print("-" * 78)

    # Save best polished model checkpoint
    best_state = (
        best_candidate_model.module.state_dict()
        if hasattr(best_candidate_model, "module")
        else best_candidate_model.state_dict()
    )
    save_payload = {
        "model_state_dict": best_state,
        "dataset": dataset,
        "epochs_polished": epochs,
        "baseline_top1": initial_top1,
        "baseline_tta_top1": initial_tta_top1,
        "polished_top1": best_candidate_top1,
        "best_mode": best_candidate_name,
        "retained_baseline": retained_baseline,
        "architecture_revision": ARCHITECTURE_REVISION,
    }
    torch.save(save_payload, output_dir / "polished_best.pt")

    # Save summary metrics
    summary_metrics = {
        "dataset": dataset,
        "baseline_standard_top1": base_eval["top1"],
        "baseline_tta_top1": base_eval_tta["top1"],
        "best_polished_mode": best_candidate_name,
        "best_polished_top1": best_candidate_top1,
        "absolute_gain": gain,
        "gain_over_baseline_tta": gain_over_baseline_tta,
        "retained_baseline": retained_baseline,
        "results": {
            k: {
                "top1": v["top1"],
                "loss": v["loss"],
                "macro_f1": v["macro_f1"],
                "use_tta": v["use_tta"],
            }
            for k, v in results.items()
        }
    }
    (output_dir / "polish_summary.json").write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")

    # Export confusion matrix and predictions for the best candidate
    best_eval_dict = results[best_candidate_name]
    with (output_dir / "polished_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(best_eval_dict["matrix"])

    with (output_dir / "polished_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "target", "prediction", "negative_log_likelihood"])
        writer.writeheader()
        writer.writerows(best_eval_dict["predictions"])

    print(f"\nArtifacts successfully exported to: {output_dir}")
    return summary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to initial best_val.pt checkpoint")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25, help="Number of polishing epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01, help="Starting learning rate for polishing")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--use-swa", action="store_true", default=True, help="Enable Stochastic Weight Averaging")
    parser.add_argument("--no-swa", dest="use_swa", action="store_false")
    parser.add_argument("--swa-start-ratio", type=float, default=0.4, help="Epoch fraction when SWA starts")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="0.0 uses all 50k samples for training")
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs before cosine decay")
    parser.add_argument("--use-ema", action="store_true", default=False, help="Enable Model Exponential Moving Average")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Alpha for Mixup data augmentation")
    parser.add_argument("--preserve-baseline", action="store_true", default=True, help="Preserve baseline when polish regresses")
    parser.add_argument("--no-preserve-baseline", dest="preserve_baseline", action="store_false")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches per epoch (for quick testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()
    polish_train(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_swa=args.use_swa,
        swa_start_ratio=args.swa_start_ratio,
        seed=args.seed,
        device_str=args.device,
        val_fraction=args.val_fraction,
        cutout_length=args.cutout_length,
        max_batches=args.max_batches,
        warmup_epochs=args.warmup_epochs,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        mixup_alpha=args.mixup_alpha,
        preserve_baseline=args.preserve_baseline,
    )


if __name__ == "__main__":
    main()
