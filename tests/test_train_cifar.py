"""Unit tests for train_cifar module."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train_cifar
from models.ticknet_l import build_ticknet_l


def test_parse_args_and_config(tmp_path: Path):
    cfg_file = tmp_path / "test_cfg.json"
    cfg_file.write_text(json.dumps({
        "dataset": "cifar100",
        "optimizer": "adam",
        "learning_rate": 0.0005,
        "epochs": 150,
        "batch_size": 64,
    }), encoding="utf-8")

    args = train_cifar.parse_args(["--config", str(cfg_file), "--output-dir", str(tmp_path / "out")])
    assert args.dataset == "cifar100"
    assert args.optimizer == "adam"
    assert args.learning_rate == 0.0005
    assert args.epochs == 150
    assert args.batch_size == 64

    # Test CLI override
    override_args = train_cifar.parse_args([
        "--config", str(cfg_file),
        "--learning-rate", "0.001",
        "--output-dir", str(tmp_path / "out2"),
    ])
    assert override_args.learning_rate == 0.001


def test_build_optimizer_and_scheduler():
    model = nn.Linear(10, 2)

    args_sgd = train_cifar.parse_args([
        "--optimizer", "sgd", "--learning-rate", "0.1",
        "--momentum", "0.9", "--weight-decay", "1e-4",
        "--output-dir", "dummy_out"
    ])
    opt_sgd, sched_sgd = train_cifar.build_optimizer_and_scheduler(model, args_sgd)
    assert isinstance(opt_sgd, torch.optim.SGD)
    assert opt_sgd.param_groups[0]["lr"] == 0.1
    assert opt_sgd.param_groups[0]["momentum"] == 0.9
    assert opt_sgd.param_groups[0]["weight_decay"] == 1e-4
    assert isinstance(sched_sgd, torch.optim.lr_scheduler.CosineAnnealingLR)

    args_adam = train_cifar.parse_args([
        "--optimizer", "adam", "--learning-rate", "0.001",
        "--adam-beta1", "0.9", "--adam-beta2", "0.999",
        "--weight-decay", "1e-4", "--output-dir", "dummy_out"
    ])
    opt_adam, sched_adam = train_cifar.build_optimizer_and_scheduler(model, args_adam)
    assert isinstance(opt_adam, torch.optim.Adam)
    assert opt_adam.param_groups[0]["lr"] == 0.001
    assert opt_adam.param_groups[0]["betas"] == (0.9, 0.999)
    assert opt_adam.param_groups[0]["weight_decay"] == 1e-4


def test_run_epoch_train_and_eval():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_res = train_cifar.run_epoch(model, loader, criterion, device, optimizer)
    assert "loss" in train_res and "top1" in train_res and train_res["samples"] == 8
    assert 0.0 <= train_res["top1"] <= 100.0

    eval_res = train_cifar.run_epoch(model, loader, criterion, device, optimizer=None)
    assert eval_res["samples"] == 8


def test_evaluate_artifacts_and_metrics(tmp_path: Path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    device = torch.device("cpu")

    x = torch.randn(12, 3, 32, 32)
    y = torch.randint(0, 10, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    metrics = train_cifar.evaluate(model, loader, device, tmp_path, num_classes=10)

    assert "top1" in metrics and "loss" in metrics and "macro_f1" in metrics
    assert metrics["samples"] == 12
    assert (tmp_path / "test_metrics.json").is_file()
    assert (tmp_path / "confusion_matrix.csv").is_file()
    assert (tmp_path / "test_predictions.csv").is_file()

    with (tmp_path / "confusion_matrix.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 10
    assert len(rows[0]) == 10


def test_train_cifar_smoke_run(tmp_path: Path, monkeypatch):
    # Create small synthetic dataset loaders to test end-to-end training pipeline
    x_train = torch.randn(16, 3, 32, 32)
    y_train = torch.randint(0, 10, (16,))
    x_test = torch.randn(8, 3, 32, 32)
    y_test = torch.randint(0, 10, (8,))

    train_l = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
    val_l = DataLoader(TensorDataset(x_train[:4], y_train[:4]), batch_size=4)
    test_l = DataLoader(TensorDataset(x_test, y_test), batch_size=4)

    monkeypatch.setattr(train_cifar, "build_cifar_loaders", lambda *args, **kwargs: (train_l, val_l, test_l))

    out_dir = tmp_path / "smoke_run"
    result = train_cifar.main([
        "--dataset", "cifar10",
        "--output-dir", str(out_dir),
        "--epochs", "2",
        "--batch-size", "4",
        "--optimizer", "sgd",
        "--learning-rate", "0.1",
        "--device", "cpu",
        "--num-workers", "0",
    ])

    assert (out_dir / "epochs.csv").is_file()
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "best_val.pt").is_file()
    assert (out_dir / "last.pt").is_file()
    assert (out_dir / "test_metrics.json").is_file()

    with (out_dir / "epochs.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[1]["epoch"] == "2"

    # Test evaluate CLI mode with saved best_val.pt
    eval_dir = tmp_path / "eval_run"
    eval_res = train_cifar.main([
        "--dataset", "cifar10",
        "--evaluate", str(out_dir / "best_val.pt"),
        "--output-dir", str(eval_dir),
        "--device", "cpu",
        "--num-workers", "0",
    ])
    assert eval_res["samples"] == 8
    assert (eval_dir / "test_metrics.json").is_file()


def test_train_cifar_author_basic_model_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    x_train = torch.randn(8, 3, 32, 32)
    y_train = torch.randint(0, 10, (8,))
    x_val = torch.randn(4, 3, 32, 32)
    y_val = torch.randint(0, 10, (4,))
    x_test = torch.randn(4, 3, 32, 32)
    y_test = torch.randint(0, 10, (4,))

    train_l = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
    val_l = DataLoader(TensorDataset(x_val, y_val), batch_size=4)
    test_l = DataLoader(TensorDataset(x_test, y_test), batch_size=4)

    monkeypatch.setattr(train_cifar, "build_cifar_loaders", lambda *args, **kwargs: (train_l, val_l, test_l))

    out_dir = tmp_path / "basic_smoke_run"
    result = train_cifar.main([
        "--model", "basic",
        "--dataset", "cifar10",
        "--output-dir", str(out_dir),
        "--epochs", "1",
        "--batch-size", "4",
        "--optimizer", "sgd",
        "--learning-rate", "0.1",
        "--device", "cpu",
        "--num-workers", "0",
    ])

    assert (out_dir / "best_val.pt").is_file()
    assert (out_dir / "test_metrics.json").is_file()
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "basic"
    assert cfg["architecture_revision"] == "ticknet-basic-author"

    # Test evaluate with author basic model checkpoint
    eval_dir = tmp_path / "basic_eval_run"
    eval_res = train_cifar.main([
        "--dataset", "cifar10",
        "--evaluate", str(out_dir / "best_val.pt"),
        "--output-dir", str(eval_dir),
        "--device", "cpu",
        "--num-workers", "0",
    ])
    assert eval_res["samples"] == 4

