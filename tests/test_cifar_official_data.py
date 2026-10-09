"""Optional integration with original CIFAR files, never network downloads in tests."""
import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from models.cifar_data import build_cifar_loaders, seed_everything
from models.cifar_experiment import data_evidence
from models.ticknet_l import build_ticknet_l
from scripts.download_cifar import extracted_dataset_valid


DATA_ROOT = Path(os.environ.get("CIFAR_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))


@pytest.mark.parametrize("dataset,nc", [("cifar10", 10), ("cifar100", 100)])
def test_official_files_split_counts_and_real_images(dataset, nc):
    if not extracted_dataset_valid(dataset, DATA_ROOT):
        pytest.skip(f"Verified original {dataset} files are not available")
    seed_everything(42)
    loaders = build_cifar_loaders(DATA_ROOT, dataset, batch_size=4, val_fraction=0.1,
                                  num_workers=0, download=False)
    train, validation, test = loaders
    assert [len(loader.dataset) for loader in loaders] == [45_000, 5_000, 10_000]
    assert not set(train.dataset.indices) & set(validation.dataset.indices)
    assert set(train.dataset.indices) | set(validation.dataset.indices) == set(range(50_000))
    for loader, per_class in ((train, 45_000 // nc), (validation, 5_000 // nc)):
        counts = Counter(loader.dataset.targets[i] for i in loader.dataset.indices)
        assert counts == {c: per_class for c in range(nc)}
    assert Counter(test.dataset.targets) == {c: 10_000 // nc for c in range(nc)}
    evidence = data_evidence(loaders)
    assert evidence["train"]["sha256"] != evidence["validation"]["sha256"]
    images, labels = next(iter(train))
    model = build_ticknet_l(nc, cifar=True)
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
