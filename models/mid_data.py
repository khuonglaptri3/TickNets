"""Seeded ImageFolder loaders for the paired Mid32/Mid224 exam datasets."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ("bird", "cat", "dog", "frog", "horse")
IMAGE_SIZES = {"Mid32": 32, "Mid224": 224}


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class RequireImageSize:
    """Catch wrong dataset roots instead of silently resizing their images."""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        if image.size != (self.size, self.size):
            raise ValueError(f"Expected native {self.size}x{self.size} image, got {image.size}")
        return image


def build_mid_loaders(data_root, variant: str, *, batch_size=64, seed=42,
                      num_workers=0, augment=True, pin_memory=False):
    """Seed sampler/workers; call seed_everything(seed) first for reproducible
    augmentation/model initialization, including when num_workers is zero.
    """
    if variant not in IMAGE_SIZES:
        raise ValueError(f"Unknown variant {variant!r}; choose Mid32 or Mid224")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    size = IMAGE_SIZES[variant]
    train_ops = [RequireImageSize(size)]
    if augment:
        train_ops.extend([transforms.RandomCrop(size, padding=size // 8, padding_mode="reflect"),
                          transforms.RandomHorizontalFlip()])
    train_ops.append(transforms.ToTensor())
    root = Path(data_root) / variant
    train = datasets.ImageFolder(root / "train", transform=transforms.Compose(train_ops))
    test = datasets.ImageFolder(root / "test", transform=transforms.Compose([RequireImageSize(size), transforms.ToTensor()]))
    expected_mapping = {name: index for index, name in enumerate(CLASSES)}
    if train.class_to_idx != expected_mapping or test.class_to_idx != expected_mapping:
        raise ValueError(f"Unexpected class mapping: train={train.class_to_idx}, test={test.class_to_idx}")
    train_ids = {(label, Path(path).name) for path, label in train.samples}
    test_ids = {(label, Path(path).name) for path, label in test.samples}
    if train_ids & test_ids:
        raise ValueError("The same class/filename occurs in both train and test")
    common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker,
                  pin_memory=pin_memory, drop_last=False)
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
    # A separate generator prevents test iteration from changing the train RNG stream.
    test_loader = DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **common)
    return train_loader, test_loader
