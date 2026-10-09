"""Unit tests for the polishing / fine-tuning pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import polish_cifar
from models.ticknet_l import build_ticknet_l


class DummyDataset(Dataset):
    def __init__(self, num_samples=16, num_classes=10):
        self.data = np.random.randn(num_samples, 3, 32, 32).astype(np.float32)
        self.targets = [i % num_classes for i in range(num_samples)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.targets[idx]


def test_evaluate_with_tta():
    num_classes = 10
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    dataset = DummyDataset(num_samples=16, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=8)
    device = torch.device("cpu")

    std_res = polish_cifar.evaluate_with_tta(model, loader, device, num_classes, use_tta=False)
    assert "top1" in std_res
    assert "loss" in std_res
    assert "macro_f1" in std_res
    assert std_res["samples"] == 16
    assert std_res["use_tta"] is False

    tta_res = polish_cifar.evaluate_with_tta(model, loader, device, num_classes, use_tta=True)
    assert tta_res["samples"] == 16
    assert tta_res["use_tta"] is True
    assert 0.0 <= tta_res["top1"] <= 100.0


def test_polish_train_end_to_end(tmp_path, monkeypatch):
    num_classes = 10
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    init_ckpt = tmp_path / "init_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, init_ckpt)

    # Mock build_cifar_loaders to return small dummy datasets
    def mock_loaders(*args, **kwargs):
        train = DataLoader(DummyDataset(num_samples=16, num_classes=num_classes), batch_size=8, shuffle=True)
        val = DataLoader(DummyDataset(num_samples=8, num_classes=num_classes), batch_size=8)
        test = DataLoader(DummyDataset(num_samples=8, num_classes=num_classes), batch_size=8)
        return train, val, test

    monkeypatch.setattr(polish_cifar, "build_cifar_loaders", mock_loaders)

    out_dir = tmp_path / "polish_output"
    summary = polish_cifar.polish_train(
        checkpoint_path=init_ckpt,
        dataset="cifar10",
        data_root=tmp_path / "data",
        output_dir=out_dir,
        epochs=2,
        batch_size=8,
        lr=0.01,
        use_swa=True,
        swa_start_ratio=0.5,
        device_str="cpu",
    )

    assert (out_dir / "polished_best.pt").exists()
    assert (out_dir / "polish_summary.json").exists()
    assert (out_dir / "epochs_polish.csv").exists()
    assert (out_dir / "polished_confusion_matrix.csv").exists()
    assert (out_dir / "polished_test_predictions.csv").exists()

    saved_meta = json.loads((out_dir / "polish_summary.json").read_text(encoding="utf-8"))
    assert saved_meta["dataset"] == "cifar10"
    assert "best_polished_top1" in saved_meta
    assert "absolute_gain" in saved_meta


def test_polish_train_v2_features(tmp_path, monkeypatch):
    num_classes = 10
    model = build_ticknet_l(num_classes=num_classes, cifar=True)
    init_ckpt = tmp_path / "init_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, init_ckpt)

    def mock_loaders(*args, **kwargs):
        train = DataLoader(DummyDataset(num_samples=16, num_classes=num_classes), batch_size=8, shuffle=True)
        val = DataLoader(DummyDataset(num_samples=8, num_classes=num_classes), batch_size=8)
        test = DataLoader(DummyDataset(num_samples=8, num_classes=num_classes), batch_size=8)
        return train, val, test

    monkeypatch.setattr(polish_cifar, "build_cifar_loaders", mock_loaders)

    out_dir = tmp_path / "polish_output_v2"
    summary = polish_cifar.polish_train(
        checkpoint_path=init_ckpt,
        dataset="cifar10",
        data_root=tmp_path / "data",
        output_dir=out_dir,
        epochs=3,
        batch_size=8,
        lr=0.001,
        warmup_epochs=1,
        use_ema=True,
        ema_decay=0.99,
        mixup_alpha=0.2,
        val_fraction=0.1,
        use_swa=False,
        preserve_baseline=True,
        device_str="cpu",
    )

    assert (out_dir / "polished_best.pt").exists()
    assert (out_dir / "polish_summary.json").exists()
    saved_meta = json.loads((out_dir / "polish_summary.json").read_text(encoding="utf-8"))
    assert "EMA_Model_standard" in saved_meta["results"]
    assert "Best_Val_Model_standard" in saved_meta["results"]
    assert "retained_baseline" in saved_meta
