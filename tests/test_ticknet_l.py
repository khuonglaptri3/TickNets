"""Validate final CIFAR training and the declared inference budget."""

import pytest
import torch
from torch import nn

from models.ticknet_l import build_ticknet_l
from models.TickNet import build_TickNet
from models.model_profile import profile_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("num_classes", (10, 100))
def test_l_trains_on_both_cifar_tasks_with_no_disconnected_parameters(num_classes):
    torch.manual_seed(42)
    model = build_ticknet_l(num_classes=num_classes, cifar=True).train()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, num_classes)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, num_classes - 1]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("num_classes", (10, 100))
def test_l_satisfies_budget_with_independent_operator_count(num_classes):
    baseline = profile_model(build_TickNet(num_classes, typesize="basic", cifar=True), 32, cross_check=True)
    candidate = profile_model(build_ticknet_l(num_classes, cifar=True), 32, cross_check=True)
    assert candidate["learnable_parameters"] <= 6_000_000
    assert candidate["flops"] < 1_000_000_000
    assert candidate["flops"] < baseline["flops"]
    assert candidate["pytorch_cross_check_flops"] == candidate["flops"]
    assert candidate["within_exam_limits"]


def test_profiler_counts_grouped_convolution_and_linear_without_mutating_model():
    model = nn.Sequential(
        nn.Conv2d(3, 6, 3, padding=1, groups=3, bias=True),
        nn.BatchNorm2d(6), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(6, 5),
    ).train()
    # Exercise restoration of mixed train/eval flags as well as BN statistics.
    model[1].eval()
    original_flags = [module.training for module in model.modules()]
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    profile = profile_model(model, 32, cross_check=True)
    expected_macs = 32 * 32 * 6 * 9 + 6 * 5
    assert profile["macs"] == expected_macs
    assert profile["flops"] == 2 * expected_macs
    assert [module.training for module in model.modules()] == original_flags
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name]), name
