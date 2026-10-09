"""Recovery must match uninterrupted stochastic training, and fail before writes."""
import csv
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import train_cifar
from models.cifar_data import seed_worker
from models.cifar_experiment import file_sha256


class RandomImages(Dataset):
    def __init__(self, nc=10, augmented=False, offset=0):
        self.data = np.random.default_rng(10 + offset).normal(size=(8, 3, 32, 32)).astype(np.float32)
        self.targets = [i % nc for i in range(8)]
        self.augmented = augmented

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        image = torch.from_numpy(self.data[i].copy())
        if self.augmented:
            image = image + (random.random() + np.random.random() + torch.rand(())) * 0.01
        return image, self.targets[i]


@pytest.fixture
def tiny_pipeline(monkeypatch):
    def loaders(*args, **kwargs):
        nc = 100 if args[1] == "cifar100" else 10
        seed = kwargs["seed"]
        train = DataLoader(RandomImages(nc, augmented=True), batch_size=4, shuffle=True,
                           num_workers=kwargs["num_workers"], worker_init_fn=seed_worker,
                           generator=torch.Generator().manual_seed(seed))
        val = DataLoader(RandomImages(nc, offset=1), batch_size=4,
                         generator=torch.Generator().manual_seed(seed + 1))
        test = DataLoader(RandomImages(nc, offset=2), batch_size=4,
                          generator=torch.Generator().manual_seed(seed + 2))
        return train, val, test

    def build(num_classes, **kwargs):
        return nn.Sequential(nn.Flatten(), nn.Linear(3072, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, num_classes))

    monkeypatch.setattr(train_cifar, "build_cifar_loaders", loaders)
    monkeypatch.setattr(train_cifar, "build_ticknet_l", build)
    monkeypatch.setattr(train_cifar, "build_TickNet", build)
    return loaders


def command(output, **overrides):
    values = dict(dataset="cifar10", model="l", optimizer="sgd", learning_rate=0.01,
                  epochs=3, batch_size=4, num_workers=0, device="cpu", output_dir=output)
    values.update(overrides)
    return [part for key, value in values.items() for part in ("--" + key.replace("_", "-"), str(value))]


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_equal(left, right)
    else:
        assert a == b


def test_source_refactor_evaluation_is_explicit_and_records_provenance(tiny_pipeline, tmp_path, monkeypatch):
    from models import cifar_training
    output = tmp_path / "original"
    train_cifar.main(command(output))
    checkpoint = output / "best_val.pt"
    before_hash = file_sha256(checkpoint)
    original_source = json.loads((output / "config.json").read_text())["source_sha256"]
    evidence = cifar_training.source_evidence()
    monkeypatch.setattr(cifar_training, "source_evidence", lambda: {**evidence, "source_sha256": "refactored-source"})
    evaluation = ["--evaluate", str(checkpoint), "--output-dir", str(tmp_path / "eval"), "--device", "cpu", "--num-workers", "0"]
    with pytest.raises(ValueError, match="Source code changed"):
        train_cifar.main(evaluation)
    result = train_cifar.main(evaluation + ["--allow-eval-source-change"])
    assert result["evaluation_source_sha256"] == "refactored-source"
    assert result["training_source_sha256"] == original_source
    assert result["source_change_accepted"] is True
    assert file_sha256(checkpoint) == before_hash
    with pytest.raises(SystemExit):
        train_cifar.parse_args(command(tmp_path / "resume") + ["--resume", str(output / "last.pt"), "--allow-eval-source-change"])
    with pytest.raises(ValueError, match="Source code changed"):
        train_cifar.main(command(output) + ["--resume", str(output / "last.pt")])


@pytest.mark.parametrize("dataset", ("cifar10", "cifar100"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_resume_matches_weights_optimizer_scheduler_history_and_best(tiny_pipeline, tmp_path, dataset, optimizer):
    full, split = tmp_path / "full", tmp_path / "split"
    train_cifar.main(command(full, dataset=dataset, optimizer=optimizer))
    paused = train_cifar.main(command(split, dataset=dataset, optimizer=optimizer) + ["--stop-after-epoch", "1"])
    assert paused["final_test"] is None
    assert not (split / "test_metrics.json").exists()
    # Only last.pt needs to survive the transfer to a fresh Kaggle session.
    (split / "best_val.pt").unlink()
    (split / "epochs.csv").unlink()
    resumed = tmp_path / "resumed"
    train_cifar.main(command(resumed, dataset=dataset, optimizer=optimizer) + ["--resume", str(split / "last.pt")])
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "history"):
        assert_state_equal(a[key], b[key])
    assert_state_equal(a["best_checkpoint"]["model_state_dict"], b["best_checkpoint"]["model_state_dict"])
    assert a["best_checkpoint"]["epoch"] == b["best_checkpoint"]["epoch"]
    assert (full / "epochs.csv").read_bytes() == (resumed / "epochs.csv").read_bytes()


def test_resume_restores_worker_augmentation(tiny_pipeline, tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    train_cifar.main(command(full, epochs=2, num_workers=2))
    train_cifar.main(command(split, epochs=2, num_workers=2) + ["--stop-after-epoch", "1"])
    train_cifar.main(command(split, epochs=2, num_workers=2) + ["--resume", str(split / "last.pt")])
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(split / "last.pt", weights_only=False)
    assert_state_equal(a["model_state_dict"], b["model_state_dict"])
    assert a["history"] == b["history"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is not available")
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_cuda_resume_restores_dropout_and_optimizer(tiny_pipeline, tmp_path, optimizer):
    full, split = tmp_path / "full", tmp_path / "split"
    settings = dict(epochs=2, optimizer=optimizer, device="cuda:0")
    train_cifar.main(command(full, **settings))
    train_cifar.main(command(split, **settings) + ["--stop-after-epoch", "1"])
    train_cifar.main(command(split, **settings) + ["--resume", str(split / "last.pt")])
    a = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(split / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "history"):
        assert_state_equal(a[key], b[key])


@pytest.mark.parametrize("field,value", [("learning_rate", 0.02), ("seed", 43), ("optimizer", "adam"),
                                         ("batch_size", 2), ("epochs", 4), ("num_workers", 1),
                                         ("dataset", "cifar100"), ("model", "basic")])
def test_changed_recipe_rejected_without_touching_run(tiny_pipeline, tmp_path, field, value):
    output = tmp_path / "run"
    train_cifar.main(command(output) + ["--stop-after-epoch", "1"])
    before = {p.name: file_sha256(p) for p in output.iterdir()}
    with pytest.raises(ValueError, match="recipe"):
        train_cifar.main(command(output, **{field: value}) + ["--resume", str(output / "last.pt")])
    assert before == {p.name: file_sha256(p) for p in output.iterdir()}


@pytest.mark.parametrize("change", ("source", "revision", "runtime", "history", "rng", "epoch", "best", "scheduler"))
def test_incompatible_checkpoint_rejected_before_destination_creation(tiny_pipeline, tmp_path, change):
    output = tmp_path / "run"
    train_cifar.main(command(output) + ["--stop-after-epoch", "1"])
    ckpt = torch.load(output / "last.pt", weights_only=False)
    if change == "source":
        ckpt["config"]["source_sha256"] = {}
    elif change == "revision":
        ckpt["config"]["architecture_revision"] = "obsolete"
    elif change == "runtime":
        ckpt["config"]["runtime"] = {}
    elif change == "history":
        ckpt["history"] = []
    elif change == "rng":
        del ckpt["rng_state"]
    elif change == "epoch":
        ckpt["epoch"] = 0
    elif change == "best":
        ckpt["best_checkpoint"] = None
    else:
        ckpt["scheduler_state_dict"]["last_epoch"] = 0
    corrupt = tmp_path / "incompatible.pt"
    torch.save(ckpt, corrupt)
    destination = tmp_path / "new"
    with pytest.raises(ValueError):
        train_cifar.main(command(destination) + ["--resume", str(corrupt)])
    assert not destination.exists()


def test_changed_dataset_rejected_before_writes(tiny_pipeline, monkeypatch, tmp_path):
    output = tmp_path / "run"
    train_cifar.main(command(output) + ["--stop-after-epoch", "1"])
    def changed(*args, **kwargs):
        loaders = tiny_pipeline(*args, **kwargs)
        loaders[0].dataset.data[0, 0, 0, 0] += 1
        return loaders
    monkeypatch.setattr(train_cifar, "build_cifar_loaders", changed)
    with pytest.raises(ValueError, match="Dataset"):
        train_cifar.main(command(tmp_path / "new") + ["--resume", str(output / "last.pt")])
    assert not (tmp_path / "new").exists()


def test_completed_resume_does_not_evaluate_test_again(tiny_pipeline, monkeypatch, tmp_path):
    output = tmp_path / "run"
    initial = train_cifar.main(command(output))
    monkeypatch.setattr(train_cifar, "evaluate", lambda *a: pytest.fail("Test repeated"))
    restored = train_cifar.main(command(output) + ["--resume", str(output / "last.pt")])
    assert restored["final_test"] == initial["final_test"]


def test_skip_test_then_finish_without_more_training(tiny_pipeline, tmp_path):
    output = tmp_path / "run"
    train_cifar.main(command(output) + ["--skip-test"])
    assert not (output / "test_metrics.json").exists()
    result = train_cifar.main(command(output) + ["--resume", str(output / "last.pt")])
    assert result["final_test"]["samples"] == 8


def test_best_checkpoint_uses_loss_tie_break_and_survives_worse_epoch(tiny_pipeline, monkeypatch, tmp_path):
    validation = iter([(90, 0.5), (90, 0.4), (80, 0.1)])
    original = train_cifar.run_epoch
    def controlled(model, loader, criterion, device, optimizer=None):
        metrics = original(model, loader, criterion, device, optimizer)
        if optimizer is None:
            top1, loss = next(validation)
            metrics.update(top1=top1, loss=loss)
        return metrics
    monkeypatch.setattr(train_cifar, "run_epoch", controlled)
    result = train_cifar.main(command(tmp_path / "run"))
    assert result["best_validation"]["epoch"] == 2
    assert result["final_test"]["checkpoint_epoch"] == 2


def test_committed_epoch_recovers_after_best_export_failure(tiny_pipeline, monkeypatch, tmp_path):
    import models.cifar_training as training
    original = training._export_committed
    monkeypatch.setattr(training, "_export_committed", lambda *a: (_ for _ in ()).throw(OSError("disk interrupted")))
    output = tmp_path / "run"
    with pytest.raises(OSError, match="disk"):
        train_cifar.main(command(output) + ["--stop-after-epoch", "1"])
    assert (output / "last.pt").is_file()
    monkeypatch.setattr(training, "_export_committed", original)
    train_cifar.main(command(output) + ["--resume", str(output / "last.pt")])
    with (output / "epochs.csv").open() as handle:
        assert [int(r["epoch"]) for r in csv.DictReader(handle)] == [1, 2, 3]


def test_evaluate_infers_cifar100_and_rejects_conflicting_dataset(tiny_pipeline, tmp_path):
    output = tmp_path / "run"
    train_cifar.main(command(output, dataset="cifar100"))
    args = ["--evaluate", str(output / "best_val.pt"), "--device", "cpu", "--num-workers", "0"]
    result = train_cifar.main(args + ["--output-dir", str(tmp_path / "eval")])
    assert result["dataset"] == "cifar100"
    with pytest.raises(ValueError, match="dataset"):
        train_cifar.main(args + ["--dataset", "cifar10", "--output-dir", str(tmp_path / "bad")])
    assert not (tmp_path / "bad").exists()


def test_evaluate_reuses_verified_metrics_in_completed_folder(tiny_pipeline, monkeypatch, tmp_path):
    output = tmp_path / "run"
    result = train_cifar.main(command(output))
    monkeypatch.setattr(train_cifar, "evaluate", lambda *a: pytest.fail("Repeated evaluation"))
    evaluated = train_cifar.main(["--evaluate", str(output / "best_val.pt"), "--device", "cpu", "--num-workers", "0"])
    assert evaluated == result["final_test"]
