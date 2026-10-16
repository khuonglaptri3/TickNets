"""Tests for models/model_l_from_final.py adapted from Final_report for Mid32 and Mid224 (5 classes)."""
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from models.mid_models import MODEL_REVISIONS, build_mid_model
from models.model_profile import profile_model
from models.ticknet_l import CompressedPDPBlock as OriginalCompressedPDPBlock


ROOT = Path(__file__).resolve().parents[1]
REVISION = "ticknet-l-from-final-v1"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def candidate_module():
    assert importlib.util.find_spec("models.model_l_from_final") is not None, (
        "models.model_l_from_final module is missing"
    )
    return importlib.import_module("models.model_l_from_final")


@pytest.mark.parametrize("size", (32, 224))
def test_original_l_remains_untouched_for_pinned_checkpoints(size):
    model = build_mid_model("l", variant=f"Mid{size}")
    assert sum(isinstance(m, OriginalCompressedPDPBlock) for m in model.modules()) == 7
    assert sum(p.numel() for p in model.parameters()) == 1_096_260
    assert model.architecture_revision == MODEL_REVISIONS["l"] == "ticknet-l-v1"


@pytest.mark.parametrize("size", (32, 224))
def test_model_l_from_final_structure_and_stage_depths(size):
    mod = candidate_module()
    model = mod.build_model_l_from_final(num_classes=5, cifar=(size == 32))
    assert model.in_size == (size, size)
    assert model.architecture_revision == REVISION
    assert sum(isinstance(m, mod.CompressedPDPBlock) for m in model.modules()) == 10
    assert [len(getattr(model.backbone, f"stage{i}")) for i in range(1, 6)] == [1, 2, 3, 3, 1]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1_280_827


@pytest.mark.parametrize(("size", "flops"), ((32, 205_200_640), (224, 941_805_232)))
def test_model_l_from_final_exact_budget_and_cross_check(size, flops):
    mod = candidate_module()
    model = mod.build_model_l_from_final(num_classes=5, cifar=(size == 32))
    profile = profile_model(model, size, cross_check=True)
    assert profile["learnable_parameters"] == 1_280_827
    assert profile["flops"] == profile["pytorch_cross_check_flops"] == flops
    assert profile["macs"] == flops // 2
    assert profile["within_exam_limits"]
    assert profile["output_shape"] == [1, 5]


@pytest.mark.parametrize("size", (32, 224))
def test_model_l_from_final_all_parameters_receive_finite_gradients(size):
    torch.manual_seed(42)
    mod = candidate_module()
    model = mod.build_model_l_from_final(num_classes=5, cifar=(size == 32)).train()
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


def test_trainer_and_configs_register_l_from_final():
    mod = candidate_module()
    assert MODEL_REVISIONS.get("l_from_final") == REVISION
    m32 = build_mid_model("l_from_final", num_classes=5, variant="Mid32")
    m224 = build_mid_model("l_from_final", num_classes=5, variant="Mid224")
    assert m32.in_size == (32, 32) and m224.in_size == (224, 224)

    trainer = importlib.import_module("train_mid_experiment")
    assert trainer.EXPERIMENT_MODELS.get("l_from_final") == (mod.build_model_l_from_final, REVISION)

    cfg_base = ROOT / "configs/midterm/experiment_l_from_final.json"
    cfg_lr015 = ROOT / "configs/midterm/l_from_final_lr015.json"
    cfg_sm010 = ROOT / "configs/midterm/l_from_final_smoothing010.json"
    for cfg in (cfg_base, cfg_lr015, cfg_sm010):
        assert cfg.is_file(), f"Missing config: {cfg}"
        args = trainer.parse_args(["--config", str(cfg), "--variant", "Mid32", "--output-dir", "runs/fixture"])
        assert args.model == "l_from_final"


@pytest.mark.parametrize(
    ("notebook_rel", "expected_variant", "expected_config"),
    (
        (
            "docs/kaggle/experiments/Kaggle_Mid32_L_From_Final_LR015.ipynb",
            "Mid32",
            "configs/midterm/l_from_final_lr015.json",
        ),
        (
            "docs/kaggle/experiments/Kaggle_Mid224_L_From_Final_Smoothing010.ipynb",
            "Mid224",
            "configs/midterm/l_from_final_smoothing010.json",
        ),
    ),
)
def test_kaggle_l_from_final_notebooks_are_valid_and_compile(
    notebook_rel, expected_variant, expected_config
):
    nb_path = ROOT / notebook_rel
    assert nb_path.is_file(), f"Missing notebook: {nb_path}"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 6
    for idx, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"{notebook_rel}#cell-{idx}", "exec")

    namespace = {}
    exec("".join(code_cells[0]["source"]), namespace)
    assert namespace["REPO_URL"] == "https://github.com/khuonglaptri3/TickNets.git"
    assert namespace["BRANCH"] == "release/midterm-model-l-selected"
    assert len(namespace["RUN_SPECS"]) == 1
    spec = namespace["RUN_SPECS"][0]
    assert spec["variant"] == expected_variant
    assert spec["config"] == expected_config
    assert (ROOT / spec["config"]).is_file()

