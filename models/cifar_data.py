"""DataLoader and transforms for CIFAR-10 and CIFAR-100 datasets."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .mid_data import seed_everything, seed_worker

# Standard normalization constants for CIFAR-10 and CIFAR-100
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)

CIFAR_STATS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
}

NUM_CLASSES: Dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
}


def normalize_dataset_name(name: str) -> str:
    cleaned = name.strip().lower().replace("-", "").replace("_", "")
    if cleaned in ("cifar10", "cifar100"):
        return cleaned
    raise ValueError(f"Unknown CIFAR dataset {name!r}; choose 'cifar10' or 'cifar100'")


def get_cifar_transforms(dataset_name: str, *, augment: bool = True) -> transforms.Compose:
    """Standard CIFAR preprocessing: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize."""
    canon_name = normalize_dataset_name(dataset_name)
    mean, std = CIFAR_STATS[canon_name]

    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class TransformedSubset(Dataset):
    """Subsets a dataset and applies an explicit transform to PIL images."""

    def __init__(self, data: np.ndarray, targets: List[int], indices: List[int], transform=None):
        self.data = data
        self.targets = targets
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img_arr = self.data[real_idx]
        target = int(self.targets[real_idx])
        img = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.indices)


def stratified_split_indices(
    targets: List[int],
    num_classes: int,
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministically split targets into stratified train and validation indices."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_to_indices[int(label)].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c in sorted(class_to_indices):
        indices = class_to_indices[c]
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    # Sort for deterministic iteration order
    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def build_cifar_datasets(
    data_root: str | Path,
    dataset_name: str,
    *,
    val_fraction: float = 0.0,
    seed: int = 42,
    download: bool = True,
    augment: bool = True,
) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """Download and return train, validation (optional), and test datasets."""
    canon_name = normalize_dataset_name(dataset_name)
    data_path = Path(data_root).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    dataset_cls = datasets.CIFAR10 if canon_name == "cifar10" else datasets.CIFAR100
    num_classes = NUM_CLASSES[canon_name]

    train_transform = get_cifar_transforms(canon_name, augment=augment)
    eval_transform = get_cifar_transforms(canon_name, augment=False)

    test_set = dataset_cls(root=str(data_path), train=False, download=download, transform=eval_transform)

    if val_fraction <= 0.0:
        train_set = dataset_cls(root=str(data_path), train=True, download=download, transform=train_transform)
        val_set = None
    else:
        # Load raw data without transform to split into TransformedSubsets
        raw_train = dataset_cls(root=str(data_path), train=True, download=download, transform=None)
        train_idx, val_idx = stratified_split_indices(raw_train.targets, num_classes, val_fraction, seed=seed)
        train_set = TransformedSubset(raw_train.data, raw_train.targets, train_idx, transform=train_transform)
        val_set = TransformedSubset(raw_train.data, raw_train.targets, val_idx, transform=eval_transform)

    return train_set, val_set, test_set


def build_cifar_loaders(
    data_root: str | Path,
    dataset_name: str,
    *,
    batch_size: int = 128,
    val_fraction: float = 0.0,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
    augment: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Build reproducible DataLoaders for CIFAR-10 or CIFAR-100."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    train_set, val_set, test_set = build_cifar_datasets(
        data_root=data_root,
        dataset_name=dataset_name,
        val_fraction=val_fraction,
        seed=seed,
        download=download,
        augment=augment,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
        )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader, test_loader
