"""Train CIFAR-100 TickNet-L with RandAugment and batch-level CutMix.

This is a separate entry point. The Phase 8 trainer and the original CIFAR
trainer remain unchanged. Validation and test images always use plain transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from models.cifar_training import execute
from models.model_profile import profile_model
from models.ticknet_l import build_ticknet_l
from models.TickNet import build_TickNet
from train_cifar import evaluate, parse_args, run_epoch as plain_run_epoch
from train_cifar_randaugment import (RANDAUGMENT_MAGNITUDE, RANDAUGMENT_NUM_OPS,
                                     build_randaugment_loaders)


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float,
                 num_classes: int = 100):
    """Swap one image patch and weight targets by its actual clipped area."""
    if images.ndim != 4 or labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("CutMix expects a BCHW batch and one label per image")
    if len(images) < 2:
        raise ValueError("CutMix needs at least two images")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("CutMix alpha must be positive and finite")
    batch, _, height, width = images.shape
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    patch_width = int(width * math.sqrt(1.0 - lam))
    patch_height = int(height * math.sqrt(1.0 - lam))
    center_x = int(torch.randint(width, ()).item())
    center_y = int(torch.randint(height, ()).item())
    x1 = max(0, center_x - patch_width // 2)
    y1 = max(0, center_y - patch_height // 2)
    x2 = min(width, center_x + (patch_width + 1) // 2)
    y2 = min(height, center_y + (patch_height + 1) // 2)
    perm = torch.randperm(batch, device=images.device)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    retained = 1.0 - (x2 - x1) * (y2 - y1) / (width * height)
    original = F.one_hot(labels, num_classes).to(dtype=images.dtype)
    partner = F.one_hot(labels[perm], num_classes).to(dtype=images.dtype)
    return mixed, retained * original + (1.0 - retained) * partner


def cutmix_run_epoch(model, loader, criterion, device, optimizer=None, *,
                     alpha: float, probability: float):
    if optimizer is None:
        return plain_run_epoch(model, loader, criterion, device)
    model.train(True)
    loss_sum = 0.0
    correct_weight = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = F.one_hot(labels, 100).to(dtype=images.dtype)
        if len(images) > 1 and torch.rand(()).item() < probability:
            images, targets = cutmix_batch(images, labels, alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != targets.shape or not torch.isfinite(logits).all():
            raise FloatingPointError("Invalid or non-finite CutMix logits")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CutMix loss")
        loss.backward()
        optimizer.step()
        batch_size = len(labels)
        loss_sum += float(loss.item()) * batch_size
        predicted = logits.argmax(dim=1)
        correct_weight += float(targets.gather(1, predicted[:, None]).sum().item())
        total += batch_size
    if total == 0:
        raise ValueError("Cannot train on an empty loader")
    # Training accuracy is target-weighted because CutMix has soft labels.
    return {"loss": loss_sum / total, "top1": 100.0 * correct_weight / total,
            "samples": total}


def main(argv=None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    extra_parser.add_argument("--cutmix-probability", type=float, default=0.5)
    extra, remaining = extra_parser.parse_known_args(argv)
    args = parse_args(remaining)
    if args.dataset != "cifar100" or args.model != "l":
        raise ValueError("This experiment requires CIFAR-100 and TickNet-L")
    if args.cutout:
        raise ValueError("Cutout must be disabled when CutMix is enabled")
    if not math.isfinite(extra.cutmix_alpha) or extra.cutmix_alpha <= 0:
        raise ValueError("CutMix alpha must be positive and finite")
    if not math.isfinite(extra.cutmix_probability) or not 0 < extra.cutmix_probability <= 1:
        raise ValueError("CutMix probability must be in (0, 1]")
    args.augmentation = "randaugment_cutmix"
    args.randaugment_num_ops = RANDAUGMENT_NUM_OPS
    args.randaugment_magnitude = RANDAUGMENT_MAGNITUDE
    args.cutmix_alpha = extra.cutmix_alpha
    args.cutmix_probability = extra.cutmix_probability
    args.cutmix_train_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.randaugment_train_sha256 = hashlib.sha256(
        (Path(__file__).parent / "train_cifar_randaugment.py").read_bytes()).hexdigest()
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("config", {})
        for name in ("augmentation", "randaugment_num_ops", "randaugment_magnitude",
                     "cutmix_alpha", "cutmix_probability", "cutmix_train_sha256",
                     "randaugment_train_sha256"):
            if recorded.get(name) != getattr(args, name):
                raise ValueError(f"CutMix source or recipe differs from checkpoint: {name}")

    def run_epoch(model, loader, criterion, device, optimizer=None):
        return cutmix_run_epoch(model, loader, criterion, device, optimizer,
                                alpha=args.cutmix_alpha,
                                probability=args.cutmix_probability)

    return execute(args, build_randaugment_loaders, build_ticknet_l, build_TickNet,
                   run_epoch, evaluate, profile_model)


if __name__ == "__main__":
    main()
