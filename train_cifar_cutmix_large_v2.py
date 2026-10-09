"""Train TickNet-L Large v2 with the Phase 9 alpha=1.0 CutMix recipe."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch

from models.cifar_training_large_v2 import execute
from models.model_profile import profile_model
from models.ticknet_l_large_v2 import build_ticknet_l_large_v2
from models.TickNet import build_TickNet
from train_cifar import evaluate, parse_args
from train_cifar_cutmix import cutmix_run_epoch
from train_cifar_randaugment import (RANDAUGMENT_MAGNITUDE, RANDAUGMENT_NUM_OPS,
                                     build_randaugment_loaders)


def main(argv=None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    extra_parser.add_argument("--cutmix-probability", type=float, default=0.5)
    extra, remaining = extra_parser.parse_known_args(argv)
    args = parse_args(remaining)
    if args.dataset != "cifar100" or args.model != "l" or args.cutout:
        raise ValueError("Large v2 CutMix requires CIFAR-100, model=l and no Cutout")
    if not math.isfinite(extra.cutmix_alpha) or extra.cutmix_alpha <= 0:
        raise ValueError("CutMix alpha must be positive and finite")
    if not math.isfinite(extra.cutmix_probability) or not 0 < extra.cutmix_probability <= 1:
        raise ValueError("CutMix probability must be in (0, 1]")
    # The shared parser accepts l/basic; record the actual builder separately.
    args.model = "l_large_v2"
    args.augmentation = "randaugment_cutmix_large_v2"
    args.randaugment_num_ops = RANDAUGMENT_NUM_OPS
    args.randaugment_magnitude = RANDAUGMENT_MAGNITUDE
    args.cutmix_alpha = extra.cutmix_alpha
    args.cutmix_probability = extra.cutmix_probability
    own_path = Path(__file__).resolve()
    args.cutmix_train_sha256 = hashlib.sha256(own_path.read_bytes()).hexdigest()
    args.base_cutmix_train_sha256 = hashlib.sha256(
        (own_path.parent / "train_cifar_cutmix.py").read_bytes()).hexdigest()
    args.randaugment_train_sha256 = hashlib.sha256(
        (own_path.parent / "train_cifar_randaugment.py").read_bytes()).hexdigest()
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("config", {})
        for name in ("augmentation", "randaugment_num_ops", "randaugment_magnitude",
                     "cutmix_alpha", "cutmix_probability", "cutmix_train_sha256",
                     "base_cutmix_train_sha256", "randaugment_train_sha256"):
            if recorded.get(name) != getattr(args, name):
                raise ValueError(f"Large v2 CutMix source or recipe differs: {name}")

    def run_epoch(model, loader, criterion, device, optimizer=None):
        return cutmix_run_epoch(model, loader, criterion, device, optimizer,
                                alpha=args.cutmix_alpha,
                                probability=args.cutmix_probability)

    return execute(args, build_randaugment_loaders, build_ticknet_l_large_v2,
                   build_TickNet, run_epoch, evaluate, profile_model)


if __name__ == "__main__":
    main()
