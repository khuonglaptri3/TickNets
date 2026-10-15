"""Train TickNet-L with RandAugment while keeping train_cifar.py unchanged.

The existing CIFAR trainer still owns checkpoints, validation selection and test
artifacts. This entry point changes only the training image transform.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torchvision import transforms

from models.cifar_data import CIFAR_STATS, Cutout, build_cifar_loaders, normalize_dataset_name
from models.cifar_training import execute
from models.model_profile import profile_model
from models.ticknet_l import build_ticknet_l
from models.TickNet import build_TickNet
from train_cifar import evaluate, parse_args, run_epoch


RANDAUGMENT_NUM_OPS = 2
RANDAUGMENT_MAGNITUDE = 7


def build_randaugment_loaders(data_root, dataset_name, **kwargs):
    """Preserve the existing split and evaluation transforms; augment train only."""
    if "augment" in kwargs:
        raise ValueError("RandAugment loader controls the training transform")
    train_loader, val_loader, test_loader = build_cifar_loaders(
        data_root, dataset_name, augment=False, **kwargs)
    mean, std = CIFAR_STATS[normalize_dataset_name(dataset_name)]
    training_transform = [
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=RANDAUGMENT_NUM_OPS,
                               magnitude=RANDAUGMENT_MAGNITUDE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if kwargs.get("cutout", True) and kwargs.get("cutout_length", 16) > 0:
        training_transform.append(Cutout(n_holes=1, length=kwargs.get("cutout_length", 16)))
    if not hasattr(train_loader.dataset, "transform"):
        raise TypeError("Expected a CIFAR training dataset with a transform")
    train_loader.dataset.transform = transforms.Compose(training_transform)
    return train_loader, val_loader, test_loader


def main(argv=None):
    args = parse_args(argv)
    if args.model != "l":
        raise ValueError("This RandAugment experiment is defined for TickNet-L")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.augmentation = "randaugment"
    args.randaugment_num_ops = RANDAUGMENT_NUM_OPS
    args.randaugment_magnitude = RANDAUGMENT_MAGNITUDE
    args.randaugment_train_sha256 = source_hash
    if args.resume or args.evaluate:
        checkpoint_path = args.resume or args.evaluate
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        recorded = checkpoint.get("config", {})
        if (recorded.get("randaugment_train_sha256") != source_hash or
                recorded.get("augmentation") != "randaugment" or
                recorded.get("randaugment_num_ops") != RANDAUGMENT_NUM_OPS or
                recorded.get("randaugment_magnitude") != RANDAUGMENT_MAGNITUDE):
            raise ValueError("RandAugment source or recipe differs from the checkpoint")
    return execute(args, build_randaugment_loaders, build_ticknet_l, build_TickNet,
                   run_epoch, evaluate, profile_model)


if __name__ == "__main__":
    main()
