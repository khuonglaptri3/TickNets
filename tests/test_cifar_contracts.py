"""Numerical metrics, invalid input, preprocessing and actual CIFAR model budgets."""
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

import train_cifar
from models import cifar_data
from models.cifar_data import CIFAR_STATS, Cutout, get_cifar_transforms, stratified_split_indices
from models.model_profile import profile_model
from models.TickNet import build_TickNet
from models.ticknet_l import build_ticknet_l
from test_cifar_recovery import command, tiny_pipeline


@pytest.mark.parametrize("key,value", [
    ("epochs", 0), ("epochs", 1.5), ("epochs", True), ("epochs", None),
    ("batch_size", 0), ("num_workers", -1), ("seed", -1), ("seed", 2**32),
    ("learning_rate", 0), ("learning_rate", float("nan")), ("learning_rate", float("inf")),
    ("learning_rate", True), ("val_fraction", 0), ("val_fraction", 1),
    ("momentum", 1), ("momentum", -0.1), ("weight_decay", -1),
    ("adam_beta1", 1), ("adam_beta2", -0.1), ("adam_eps", 0),
    ("eta_min", -1), ("eta_min", 1), ("cutout", "false"), ("nesterov", 1),
    ("model", "unknown"), ("optimizer", "adamw"), ("dataset", "Mid32"),
    ("stop_after_epoch", 201), ("stop_after_epoch", 0), ("unknown", 1),
])
def test_invalid_config_is_argument_error(tmp_path, key, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({key: value}))
    with pytest.raises(SystemExit) as error:
        train_cifar.parse_args(["--config", str(path), "--output-dir", str(tmp_path / "run")])
    assert error.value.code == 2


@pytest.mark.parametrize("content", ("[]", "null", "{broken"))
def test_malformed_or_nonobject_config(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(SystemExit):
        train_cifar.parse_args(["--config", str(path), "--output-dir", str(tmp_path / "run")])


@pytest.mark.parametrize("momentum,nesterov,expected", [(0, True, False), (0.9, True, True), (0.9, False, False)])
def test_effective_nesterov(momentum, nesterov, expected):
    args = train_cifar.parse_args(["--output-dir", "dummy", "--momentum", str(momentum)] +
                                 ([] if nesterov else ["--no-nesterov"]))
    optimizer, _ = train_cifar.build_optimizer_and_scheduler(nn.Linear(2, 2), args)
    assert optimizer.defaults["nesterov"] is expected


@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_cosine_rate_values_and_eta_min(optimizer):
    args = train_cifar.parse_args(["--output-dir", "dummy", "--optimizer", optimizer,
                                  "--epochs", "2", "--learning-rate", "0.1", "--eta-min", "0.01"])
    opt, scheduler = train_cifar.build_optimizer_and_scheduler(nn.Linear(2, 2), args)
    opt.step()
    scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.055)
    opt.step()
    scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_metrics_match_known_predictions_and_partial_batch(tmp_path):
    logits = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 5., 0.], [0., 5., 0.]])
    labels = torch.tensor([0, 0, 1, 2])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=3)
    metrics = train_cifar.evaluate(nn.Identity(), loader, torch.device("cpu"), tmp_path, 3)
    assert metrics["correct"] == 2 and metrics["samples"] == 4
    assert metrics["top1"] == 50
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    epoch = train_cifar.run_epoch(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert epoch["loss"] == pytest.approx(metrics["loss"])
    assert epoch["top1"] == metrics["top1"]


@pytest.mark.parametrize("evaluate", (False, True))
def test_empty_loader_has_clear_error_and_no_artifacts(tmp_path, evaluate):
    loader = DataLoader(TensorDataset(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        if evaluate:
            train_cifar.evaluate(nn.Identity(), loader, torch.device("cpu"), tmp_path, 3)
        else:
            train_cifar.run_epoch(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert not (tmp_path / "test_metrics.json").exists()


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_nonfinite_logits_do_not_write_metrics(tmp_path, value):
    loader = DataLoader(TensorDataset(torch.full((1, 3), value), torch.zeros(1, dtype=torch.long)))
    with pytest.raises(FloatingPointError):
        train_cifar.evaluate(nn.Identity(), loader, torch.device("cpu"), tmp_path, 3)
    assert not (tmp_path / "test_metrics.json").exists()


@pytest.mark.parametrize("center,length,area", [(16, 16, 256), (0, 16, 64), (31, 16, 81), (16, 5, 25), (0, 5, 9)])
def test_cutout_exact_region_and_edges(monkeypatch, center, length, area):
    monkeypatch.setattr(np.random, "randint", lambda *a: center)
    original = torch.ones(3, 32, 32)
    result = Cutout(length=length)(original)
    assert (result[0] == 0).sum().item() == area
    assert torch.equal(result[0], result[1]) and torch.equal(result[1], result[2])
    assert torch.all(original == 1)


@pytest.mark.parametrize("dataset", ("cifar10", "cifar100"))
def test_eval_transform_is_deterministic_and_uses_recorded_stats(dataset):
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    transform = get_cifar_transforms(dataset, augment=False)
    assert [type(t) for t in transform.transforms] == [transforms.ToTensor, transforms.Normalize]
    a, b = transform(image), transform(image)
    assert torch.equal(a, b)
    mean, std = CIFAR_STATS[dataset]
    assert torch.allclose(a[:, 0, 0], (torch.full((3,), 128 / 255) - torch.tensor(mean)) / torch.tensor(std))
    disabled = get_cifar_transforms(dataset, cutout=False)
    assert not any(isinstance(t, Cutout) for t in disabled.transforms)


@pytest.mark.parametrize("nc", (10, 100))
def test_stratified_official_counts_repeatability_and_all_members(nc):
    labels = [c for c in range(nc) for _ in range(50_000 // nc)]
    train, val = stratified_split_indices(labels, nc, 0.1, 42)
    assert len(train) == 45_000 and len(val) == 5_000
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(50_000))
    assert (train, val) == stratified_split_indices(labels, nc, 0.1, 42)
    assert (train, val) != stratified_split_indices(labels, nc, 0.1, 43)
    assert all(sum(labels[i] == c for i in val) == 5_000 // nc for c in range(nc))


class MockOfficialCifar:
    def __init__(self, root, train, download, transform=None):
        self.targets = [c for c in range(10) for _ in range(10 if train else 2)]
        self.data = np.random.default_rng(1 if train else 2).integers(0, 256, (len(self.targets), 32, 32, 3), dtype=np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.data[i])), self.targets[i]


def test_real_loader_construction_keeps_split_and_eval_transforms_separate(monkeypatch, tmp_path):
    monkeypatch.setattr(cifar_data.datasets, "CIFAR10", MockOfficialCifar)
    train, val, test = cifar_data.build_cifar_loaders(tmp_path, "cifar10", val_fraction=0.1, num_workers=0)
    assert len(train.dataset) == 90 and len(val.dataset) == 10 and len(test.dataset) == 20
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert train.dataset.data is val.dataset.data
    assert any(isinstance(t, Cutout) for t in train.dataset.transform.transforms)
    assert not any(isinstance(t, Cutout) for t in val.dataset.transform.transforms + test.dataset.transform.transforms)
    assert torch.equal(val.dataset[0][0], val.dataset[0][0])


@pytest.mark.parametrize("model", ("l", "basic"))
@pytest.mark.parametrize("nc", (10, 100))
def test_actual_cifar_models_have_connected_parameters_and_independent_budget(model, nc):
    net = build_ticknet_l(nc, cifar=True) if model == "l" else build_TickNet(nc, typesize="basic", cifar=True)
    profile = profile_model(net, 32, cross_check=True)
    assert profile["learnable_parameters"] <= 6_000_000 and profile["flops"] < 1_000_000_000
    assert profile["pytorch_cross_check_flops"] == profile["flops"]
    nn.functional.cross_entropy(net(torch.randn(2, 3, 32, 32)), torch.tensor([0, nc - 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("dataset", ("cifar10", "cifar100"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
@pytest.mark.parametrize("model", ("l", "basic"))
def test_real_models_complete_train_checkpoint_and_evaluation(tiny_pipeline, monkeypatch, tmp_path, dataset, optimizer, model):
    monkeypatch.setattr(train_cifar, "build_ticknet_l", build_ticknet_l)
    monkeypatch.setattr(train_cifar, "build_TickNet", build_TickNet)
    output = tmp_path / "run"
    result = train_cifar.main(command(output, dataset=dataset, optimizer=optimizer, model=model, epochs=1))
    assert result["final_test"]["samples"] == 8
    from scripts.verify_cifar_run import verify_run
    cfg, metrics, best = verify_run(output)
    assert cfg["dataset"] == dataset and cfg["optimizer"] == optimizer
    assert metrics["checkpoint_epoch"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is not available")
@pytest.mark.parametrize("dataset", ("cifar10", "cifar100"))
@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_actual_ticknet_l_cuda_training_is_finite(dataset, optimizer):
    cifar_data.seed_everything(42)
    nc = 10 if dataset == "cifar10" else 100
    model = build_ticknet_l(nc, cifar=True).cuda()
    args = train_cifar.parse_args(["--output-dir", "dummy", "--optimizer", optimizer,
                                  "--learning-rate", "0.1" if optimizer == "sgd" else "0.001"])
    opt, _ = train_cifar.build_optimizer_and_scheduler(model, args)
    images = torch.randn(2, 3, 32, 32, device="cuda")
    labels = torch.tensor([0, nc - 1], device="cuda")
    opt.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
