"""Validate native-resolution training and the declared inference budget."""
import hashlib

import pytest
import torch
from torch import nn

from models.mid_models import build_mid_model
from models.model_profile import profile_model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", (32, 224))
def test_l_trains_at_both_native_resolutions_with_no_disconnected_parameters(size):
    torch.manual_seed(42)
    model = build_mid_model("l", variant=f"Mid{size}").train()
    logits = model(torch.randn(2, 3, size, size))
    assert logits.shape == (2, 5)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("size", (32, 224))
def test_l_satisfies_budget_with_independent_operator_count(size):
    baseline = profile_model(build_mid_model("basic", variant=f"Mid{size}"), size, cross_check=True)
    candidate = profile_model(build_mid_model("l", variant=f"Mid{size}"), size, cross_check=True)
    assert candidate["learnable_parameters"] <= 6_000_000
    assert candidate["flops"] <= 800_000_000
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


@pytest.mark.parametrize("revision", (None, "ticknet-l-obsolete"))
def test_l_checkpoint_requires_matching_architecture_revision(tmp_path, monkeypatch, revision):
    import train_mid
    from torch.utils.data import DataLoader, TensorDataset

    # Small in-memory batches suffice: the failure must precede evaluation.
    dataset = TensorDataset(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    dataset.class_to_idx = {label: i for i, label in enumerate(("bird", "cat", "dog", "frog", "horse"))}
    loader = DataLoader(dataset, batch_size=2)
    monkeypatch.setattr(train_mid, "build_mid_loaders", lambda *args, **kwargs: (loader, loader))
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("fixture manifest\n", encoding="utf-8")
    checkpoint = tmp_path / "l.pt"
    torch.save({"config": {"model": "l", "variant": "Mid32", "architecture_revision": revision,
                           "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "class_to_idx": dataset.class_to_idx}, checkpoint)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="architecture revision"):
        train_mid.main(["--data-root", str(tmp_path), "--variant", "Mid32", "--evaluate", str(checkpoint),
                        "--output-dir", str(output), "--device", "cpu"])
    assert not output.exists()
