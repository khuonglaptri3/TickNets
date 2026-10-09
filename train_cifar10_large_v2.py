"""Train the 14-block TickNet-L Large v2 on CIFAR-10 with the Phase 5 recipe."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from models.cifar_data import build_cifar_loaders
from models.cifar_training_large_v2 import execute
from models.model_profile import profile_model
from models.ticknet_l_large_v2 import build_ticknet_l_large_v2
from models.TickNet import build_TickNet
from train_cifar import evaluate, parse_args, run_epoch


def main(argv=None):
    args = parse_args(argv)
    if args.dataset != "cifar10" or args.model != "l" or not args.cutout:
        raise ValueError("Phase 14 requires CIFAR-10, model=l and Cutout")

    args.model = "l_large_v2"
    args.augmentation = "crop_flip_cutout"
    own_path = Path(__file__).resolve()
    args.phase14_train_sha256 = hashlib.sha256(own_path.read_bytes()).hexdigest()
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("config", {})
        for name in ("augmentation", "phase14_train_sha256"):
            if recorded.get(name) != getattr(args, name):
                raise ValueError(f"Phase 14 source or recipe differs: {name}")

    return execute(args, build_cifar_loaders, build_ticknet_l_large_v2,
                   build_TickNet, run_epoch, evaluate, profile_model)


if __name__ == "__main__":
    main()
