"""Shared, paired validation splits for the four midterm experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.mid_data import CLASSES, IMAGE_SIZES, build_mid_loaders, seed_worker


def _sample_keys(dataset):
    keys = [f"{dataset.classes[label]}/{Path(path).name}" for path, label in dataset.samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate class/filename keys: sample identities must be unique")
    return keys


def _selected_dataset(dataset, keys, *, transform):
    """Keep ImageFolder metadata while giving each split its own transform."""
    selected = set(keys)
    result = copy.copy(dataset)
    result.samples = [sample for key, sample in sorted(zip(_sample_keys(dataset), dataset.samples))
                      if key in selected]
    result.imgs = result.samples
    result.targets = [label for _, label in result.samples]
    result.transform = transform
    return result


def _sha256(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_experiment_loaders(data_root, variant, *, batch_size=64, seed=42,
                             split_seed=123, val_fraction=0.1, num_workers=0,
                             augment=True, pin_memory=False, full_train=False):
    """Return train, validation (or None), test loaders and JSON provenance.

    Validation is sampled from sorted class/filename keys with a local Python
    RNG seeded only by split_seed. Each class contributes round(n * fraction)
    samples, clamped to [1, n - 1]; classes with fewer than two samples require
    full_train. The same named samples at both resolutions have identical
    membership regardless of their absolute root or the training seed.

    The membership hash covers the class mapping and *original* official
    train/test keys. The validation hash covers sorted validation keys. Both
    use compact UTF-8 JSON with sorted object keys; neither hashes image bytes,
    resolution, paths, or training options. Variant and image_size are recorded
    separately for checkpoint evaluation. full_train retains all official
    training samples and records an empty validation selection.

    As with build_mid_loaders, call seed_everything(seed) before training to
    seed augmentation/model initialization, especially with num_workers=0.
    Every loader has its own generator, so evaluation cannot advance the
    training sampler or worker RNG stream.
    """
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")

    original_train, test_loader = build_mid_loaders(
        data_root, variant, batch_size=batch_size, seed=seed,
        num_workers=num_workers, augment=augment, pin_memory=pin_memory,
    )
    dataset = original_train.dataset
    official_train_keys = sorted(_sample_keys(dataset))
    test_keys = _sample_keys(test_loader.dataset)
    validation_keys = []
    if not full_train:
        rng = random.Random(split_seed)
        # Group once so preparing the official 25,000 images stays linear.
        by_class = {name: [] for name in CLASSES}
        for key in official_train_keys:
            by_class[key.split("/", 1)[0]].append(key)
        for name, keys in by_class.items():
            if len(keys) < 2:
                raise ValueError(f"Class {name!r} needs at least two train samples for validation")
            count = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
            validation_keys.extend(rng.sample(keys, count))
        validation_keys.sort()
    validation_set = set(validation_keys)
    train_keys = [key for key in official_train_keys if key not in validation_set]

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=False)
    train_dataset = _selected_dataset(dataset, train_keys, transform=dataset.transform)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              generator=original_train.generator, **common)
    validation_loader = None
    if not full_train:
        validation_dataset = _selected_dataset(dataset, validation_keys,
                                               transform=test_loader.dataset.transform)
        validation_loader = DataLoader(validation_dataset, shuffle=False,
                                       generator=torch.Generator().manual_seed(seed + 2), **common)

    mapping = dict(dataset.class_to_idx)
    split_info = {
        "class_to_idx": mapping,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "train_samples": len(train_keys),
        "val_samples": len(validation_keys),
        "test_samples": len(test_keys),
        "train_keys": train_keys,
        "validation_keys": validation_keys,
        "test_keys": test_keys,
        "membership_sha256": _sha256({"class_to_idx": mapping,
                                       "train_keys": official_train_keys,
                                       "test_keys": sorted(test_keys)}),
        "validation_sha256": _sha256(validation_keys),
        "variant": variant,
        "image_size": IMAGE_SIZES[variant],
        "full_train": full_train,
        "split_method": "sorted-class-filename/python-random-sample/round-clamped/v1",
        "fingerprint_schema": "compact-sorted-utf8-json/sha256/v1",
    }
    return train_loader, validation_loader, test_loader, split_info
