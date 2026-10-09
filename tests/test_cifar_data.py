"""Unit tests for models.cifar_data module."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from models.cifar_data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    NUM_CLASSES,
    TransformedSubset,
    get_cifar_transforms,
    normalize_dataset_name,
    stratified_split_indices,
)
from models.ticknet_l import build_ticknet_l


def test_normalize_dataset_name():
    assert normalize_dataset_name("cifar10") == "cifar10"
    assert normalize_dataset_name("CIFAR-10") == "cifar10"
    assert normalize_dataset_name("cifar_10") == "cifar10"
    assert normalize_dataset_name("cifar100") == "cifar100"
    assert normalize_dataset_name("CIFAR-100") == "cifar100"
    assert normalize_dataset_name("cifar_100") == "cifar100"
    with pytest.raises(ValueError, match="Unknown CIFAR dataset"):
        normalize_dataset_name("imagenet")


@pytest.mark.parametrize("dataset_name", ("cifar10", "cifar100"))
def test_get_cifar_transforms_shape_and_type(dataset_name):
    train_tf = get_cifar_transforms(dataset_name, augment=True)
    eval_tf = get_cifar_transforms(dataset_name, augment=False)

    sample_img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))

    tensor_train = train_tf(sample_img)
    tensor_eval = eval_tf(sample_img)

    assert tensor_train.shape == (3, 32, 32)
    assert tensor_train.dtype == torch.float32
    assert tensor_eval.shape == (3, 32, 32)
    assert tensor_eval.dtype == torch.float32

    # Check normalization applied (not bounded in [0, 1])
    assert tensor_eval.min() < 0.0 or tensor_eval.max() > 1.0


def test_cutout_transform():
    from models.cifar_data import Cutout
    cutout = Cutout(n_holes=1, length=16)
    tensor = torch.ones((3, 32, 32), dtype=torch.float32)
    masked = cutout(tensor)
    assert masked.shape == (3, 32, 32)
    assert (masked == 0.0).any()
    assert (masked == 1.0).any()


def test_stratified_split_indices_proportions_and_disjoint():
    num_classes = 10
    samples_per_class = 50
    targets = []
    for c in range(num_classes):
        targets.extend([c] * samples_per_class)

    train_idx, val_idx = stratified_split_indices(targets, num_classes, val_fraction=0.1, seed=42)

    # 10% of 50 is 5 per class -> 50 total val, 450 total train
    assert len(val_idx) == 50
    assert len(train_idx) == 450

    # Ensure no overlap
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(len(targets)))

    # Verify per-class stratification
    val_labels = [targets[i] for i in val_idx]
    for c in range(num_classes):
        assert val_labels.count(c) == 5


def test_transformed_subset():
    data = np.random.randint(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    targets = list(range(20))
    indices = [2, 5, 9]

    tf = get_cifar_transforms("cifar10", augment=False)
    subset = TransformedSubset(data, targets, indices, transform=tf)

    assert len(subset) == 3
    img_tensor, target = subset[0]
    assert img_tensor.shape == (3, 32, 32)
    assert target == 2


@pytest.mark.parametrize("dataset_name", ("cifar10", "cifar100"))
def test_cifar_batches_forward_pass_ticknet_l(dataset_name):
    num_classes = NUM_CLASSES[dataset_name]
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    model.eval()

    # Create synthetic batch conforming to DataLoader output
    batch_size = 4
    tf = get_cifar_transforms(dataset_name, augment=True)
    images = torch.stack([
        tf(Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)))
        for _ in range(batch_size)
    ])
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (batch_size, num_classes)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
