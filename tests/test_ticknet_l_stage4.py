"""Stage 4 depth ablation: architecture, budget, gradients and real CLI runs."""
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from models.mid_models import MODEL_REVISIONS, build_mid_model
from models.model_profile import profile_model
from models.ticknet_l import CompressedPDPBlock
from test_mid_dataset import source_dataset
from test_mid_pipeline import prepared


ROOT = Path(__file__).resolve().parents[1]
REVISION = "ticknet-l-v1-stage4-depth3"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def candidate_module():
    assert importlib.util.find_spec("models.ticknet_l_stage4") is not None, "Stage4 builder is missing"
    return importlib.import_module("models.ticknet_l_stage4")


@pytest.mark.parametrize("size", (32, 224))
def test_original_l_keeps_seven_blocks_parameters_and_revision(size):
    model = build_mid_model("l", variant=f"Mid{size}")
    assert sum(isinstance(m, CompressedPDPBlock) for m in model.modules()) == 7
    assert sum(p.numel() for p in model.parameters()) == 1_096_260
    assert model.architecture_revision == MODEL_REVISIONS["l"] == "ticknet-l-v1"
    assert len(model.backbone.stage4) == 2


@pytest.mark.parametrize("size", (32, 224))
def test_candidate_adds_only_initialized_stage4_identity_block(size):
    torch.manual_seed(42)
    original = build_mid_model("l", variant=f"Mid{size}")
    torch.manual_seed(42)
    model = candidate_module().build_ticknet_l_stage4(cifar=size == 32)
    assert model.in_size == (size, size)
    assert model.architecture_revision == REVISION
    assert sum(isinstance(m, CompressedPDPBlock) for m in model.modules()) == 8
    assert sum(p.numel() for p in model.parameters()) == 1_236_030
    assert [len(getattr(model.backbone, f"stage{i}")) for i in range(1, 6)] == [1, 1, 2, 3, 1]
    for name, value in original.state_dict().items():
        assert torch.equal(model.state_dict()[name], value), name
    block = model.backbone.stage4.unit3
    assert isinstance(block, CompressedPDPBlock)
    assert isinstance(block.shortcut, nn.Identity)
    assert (block.pw1.conv.in_channels, block.pw1.conv.out_channels) == (288, 216)
    assert (block.pw2.conv.in_channels, block.pw2.conv.out_channels) == (216, 288)
    assert [conv.kernel_size for conv in block.dw.branches] == [(3, 3), (5, 5)]
    assert all(conv.stride == (1, 1) and conv.groups == 108 for conv in block.dw.branches)
    for module in block.modules():
        if isinstance(module, nn.Conv2d) and module.bias is not None:
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.BatchNorm2d):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.equal(module.bias, torch.zeros_like(module.bias))


@pytest.mark.parametrize(("size", "flops"), ((32, 174_236_864), (224, 846_991_472)))
def test_candidate_exact_budget_matches_independent_operator_count(size, flops):
    model = candidate_module().build_ticknet_l_stage4(cifar=size == 32)
    profile = profile_model(model, size, cross_check=True)
    assert profile["learnable_parameters"] == 1_236_030
    assert profile["flops"] == profile["pytorch_cross_check_flops"] == flops
    assert profile["macs"] == flops // 2
    assert profile["within_exam_limits"]
    assert profile["output_shape"] == [1, 5]


@pytest.mark.parametrize("size", (32, 224))
def test_candidate_all_parameters_receive_finite_gradients(size):
    torch.manual_seed(42)
    model = candidate_module().build_ticknet_l_stage4(cifar=size == 32).train()
    logits = model(torch.randn(2, 3, size, size))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_experiment_config_changes_only_model_and_registers_builder():
    path = ROOT / "configs/midterm/experiment_stage4.json"
    if not path.is_file():
        path = ROOT / "configs/midterm/experiment.json"
    assert path.is_file(), "Stage4 experiment config is missing"
    baseline = json.loads((ROOT / "configs/midterm/baseline.json").read_text(encoding="utf-8"))
    assert json.loads(path.read_text(encoding="utf-8")) == {**baseline, "model": "l_stage4"}
    trainer = importlib.import_module("train_mid_experiment")
    assert trainer.EXPERIMENT_MODELS.get("l_stage4") == (candidate_module().build_ticknet_l_stage4, REVISION)
    args = trainer.parse_args(["--config", str(path), "--variant", "Mid32", "--output-dir", "runs/fixture"])
    assert args.model == "l_stage4" and args.epochs == 200 and args.learning_rate == 0.1


@pytest.mark.parametrize("mode", ("--evaluate", "--resume"))
@pytest.mark.parametrize("revision", (None, "ticknet-l-v1", "obsolete-revision"))
def test_wrong_checkpoint_revision_fails_before_outputs(prepared, tmp_path, mode, revision):
    trainer = importlib.import_module("train_mid_experiment")
    assert "l_stage4" in trainer.EXPERIMENT_MODELS, "Stage4 trainer registration is missing"
    checkpoint = tmp_path / "wrong.pt"
    # No state dict: revision rejection must precede weight loading or output writes.
    torch.save({"config": {"trainer_revision": trainer.TRAINER_REVISION, "model": "l_stage4",
                           "variant": "Mid32", "architecture_revision": revision, "seed": 42,
                           "split_seed": 123, "val_fraction": 0.1, "full_train": False}}, checkpoint)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="architecture_revision"):
        trainer.main(["--data-root", str(prepared), "--variant", "Mid32", "--model", "l_stage4",
                      mode, str(checkpoint), "--output-dir", str(output), "--device", "cpu"])
    assert not output.exists()


@pytest.mark.parametrize(("variant", "flops"), (("Mid32", 174_236_864), ("Mid224", 846_991_472)))
def test_config_one_epoch_cli_training_and_explicit_evaluation(prepared, tmp_path, variant, flops):
    config = ROOT / "configs/midterm/experiment_stage4.json"
    if not config.is_file():
        config = ROOT / "configs/midterm/experiment.json"
    assert config.is_file(), "Stage4 experiment config is missing"
    output, evaluation = tmp_path / "run", tmp_path / "evaluation"
    command = [sys.executable, str(ROOT / "train_mid_experiment.py"), "--config", str(config),
               "--data-root", str(prepared), "--variant", variant, "--output-dir", str(output),
               "--epochs", "1", "--batch-size", "5", "--device", "cpu", "--threads", "2"]
    trained = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert saved["model"] == "l_stage4" and saved["architecture_revision"] == REVISION
    assert saved["complexity"]["flops"] == saved["complexity"]["pytorch_cross_check_flops"] == flops
    assert saved["complexity"]["learnable_parameters"] == 1_236_030
    with (output / "epochs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["train_samples"]) == 15 and int(rows[0]["val_samples"]) == 5
    assert not (output / "test_metrics.json").exists()
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(output / "best_val.pt", map_location="cpu", weights_only=True)
    assert last["epoch"] == best["epoch"] == 1
    assert last["role"] == "last" and best["role"] == "best_val"
    assert best["config"]["architecture_revision"] == REVISION
    torch.manual_seed(42)
    initial = candidate_module().build_ticknet_l_stage4(cifar=variant == "Mid32")
    key = "backbone.stage4.unit3.pw1.conv.weight"
    assert not torch.equal(last["model_state_dict"][key], initial.state_dict()[key])
    evaluated = subprocess.run([sys.executable, str(ROOT / "train_mid_experiment.py"),
                               "--data-root", str(prepared), "--variant", variant,
                               "--evaluate", str(output / "best_val.pt"), "--output-dir", str(evaluation),
                               "--device", "cpu", "--threads", "2"], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    metrics = json.loads((evaluation / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["samples"] == 10
    assert (evaluation / "test_predictions.csv").is_file()
    assert (evaluation / "confusion_matrix.csv").is_file()
