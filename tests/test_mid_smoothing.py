"""Train-only smoothing, controlled LR recipes, and exact epoch-boundary resume."""
import csv
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import train_mid_experiment as trainer
from models.mid_experiment_data import build_experiment_loaders
from models.mid_models import build_mid_model
from test_mid_dataset import source_dataset
from test_mid_pipeline import prepared


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "midterm"
RECIPES = (
    ("experiment.json", 0.05, 0.1),
    ("smoothing_005.json", 0.05, 0.1),
    ("smoothing_010.json", 0.1, 0.1),
    ("lr_005.json", 0.0, 0.05),
    ("lr_015.json", 0.0, 0.15),
)


def parse(*extra):
    return trainer.parse_args(["--variant", "Mid32", "--output-dir", "runs/unit", *extra])


def run_arguments(data, output, *extra):
    return ["--config", str(CONFIGS / "experiment.json"), "--data-root", str(data),
            "--variant", "Mid32", "--output-dir", str(output), "--device", "cpu",
            "--threads", "2", "--num-workers", "0", *extra]


@pytest.mark.parametrize("smoothing", [0.0, 0.05, 0.1, 0.999])
def test_training_criterion_matches_hand_distribution_and_torch_ce(smoothing):
    args = parse("--label-smoothing", str(smoothing))
    logits = torch.tensor([[4., -2., 1., 0., 3.], [0., 3., -1., 2., 1.],
                           [-3., 0., 2., 4., 1.]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 4, 2])
    actual = trainer.training_criterion(args)(logits, labels)
    distribution = F.one_hot(labels, num_classes=5).to(logits.dtype) * (1 - smoothing) + smoothing / 5
    hand = -(distribution * logits.log_softmax(1)).sum(1).mean()
    reference = F.cross_entropy(logits, labels, label_smoothing=smoothing)
    torch.testing.assert_close(actual, hand)
    torch.testing.assert_close(actual, reference)
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    hand_grad = torch.autograd.grad(hand, logits)[0]
    torch.testing.assert_close(actual_grad, hand_grad)


def test_cli_and_baseline_config_default_to_zero_smoothing():
    logits, labels = torch.tensor([[3., 1., 0., -2., 4.]]), torch.tensor([0])
    for args in (parse(), parse("--config", str(CONFIGS / "baseline.json"))):
        assert args.label_smoothing == 0.0
        torch.testing.assert_close(trainer.training_criterion(args)(logits, labels),
                                   F.cross_entropy(logits, labels))


@pytest.mark.parametrize("value", ["-0.01", "1", "1.01", "nan", "inf", "-inf", "not-a-float"])
def test_invalid_cli_smoothing_is_rejected(value):
    with pytest.raises(SystemExit) as error:
        parse(f"--label-smoothing={value}")
    assert error.value.code == 2


@pytest.mark.parametrize("value", [-0.01, 1, 1.01, float("nan"), float("inf"),
                                   float("-inf"), "not-a-float", None, [], {}])
def test_invalid_config_smoothing_is_rejected(tmp_path, value):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"label_smoothing": value}), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        parse("--config", str(config))
    assert error.value.code == 2


def test_cli_overrides_config_smoothing_and_learning_rate(tmp_path):
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps({"label_smoothing": 0.05, "learning_rate": 0.1}), encoding="utf-8")
    configured = parse("--config", str(config))
    overridden = parse("--config", str(config), "--label-smoothing", "0", "--learning-rate", "0.15")
    assert (configured.label_smoothing, configured.learning_rate) == (0.05, 0.1)
    assert (overridden.label_smoothing, overridden.learning_rate) == (0.0, 0.15)


@pytest.mark.parametrize("name,smoothing,learning_rate", RECIPES)
def test_recipe_runs_one_real_epoch_with_five_classes(prepared, tmp_path, name, smoothing, learning_rate):
    config_path = CONFIGS / name
    assert config_path.is_file(), f"Missing controlled recipe: {name}"
    args = parse("--config", str(config_path))
    assert (args.label_smoothing, args.learning_rate) == (smoothing, learning_rate)
    assert (args.model, args.batch_size, args.epochs, args.momentum, args.weight_decay,
            args.split_seed, args.seed, args.mixing, args.val_fraction) == (
                "l", 64, 200, 0.9, 1e-4, 123, 42, "none", 0.1)
    output = tmp_path / "run"
    result = trainer.main(run_arguments(prepared, output, "--config", str(config_path),
                                        "--epochs", "1", "--no-augment"))
    assert result["epoch"] == 1
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config == checkpoint["config"]
    assert config["label_smoothing"] == smoothing and config["learning_rate"] == learning_rate
    assert config["architecture_revision"] == "ticknet-l-v1"
    assert config["class_to_idx"] == {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}
    assert (config["train_samples"], config["val_samples"], config["test_samples"]) == (15, 5, 10)
    original_l = build_mid_model("l", variant="Mid32")
    original_l.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with (output / "epochs.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and float(rows[0]["learning_rate"]) == learning_rate
    assert (output / "best_val.pt").is_file()
    assert not (output / "test_metrics.json").exists()


def losses(model, loader, smoothing):
    model.eval()
    with torch.inference_mode():
        batches = [(model(images), labels) for images, labels in loader]
        logits = torch.cat([batch[0] for batch in batches])
        labels = torch.cat([batch[1] for batch in batches])
        return (F.cross_entropy(logits, labels).item(),
                F.cross_entropy(logits, labels, label_smoothing=smoothing).item())


def test_validation_and_explicit_test_use_hard_label_loss(prepared, tmp_path):
    assert (CONFIGS / "experiment.json").is_file(), "Missing default smoothing recipe"
    output = tmp_path / "train"
    trainer.main(run_arguments(prepared, output, "--epochs", "1", "--label-smoothing", "0.1", "--no-augment"))
    checkpoint = torch.load(output / "best_val.pt", map_location="cpu", weights_only=True)
    model = build_mid_model("l", variant="Mid32")
    model.load_state_dict(checkpoint["model_state_dict"])
    _, validation, test, _ = build_experiment_loaders(prepared, "Mid32", batch_size=64, augment=False)
    hard_val, smooth_val = losses(model, validation, 0.1)
    assert abs(hard_val - smooth_val) > 1e-5, "Fixture must distinguish hard and smoothed loss"
    assert checkpoint["best_validation"]["loss"] == pytest.approx(hard_val, rel=1e-6)
    with (output / "epochs.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["val_loss"]) == pytest.approx(hard_val, rel=1e-6)
    evaluated = trainer.main(run_arguments(prepared, tmp_path / "eval", "--label-smoothing", "0.1",
                                           "--evaluate", str(output / "best_val.pt")))
    hard_test, smooth_test = losses(model, test, 0.1)
    assert abs(hard_test - smooth_test) > 1e-5
    assert evaluated["loss"] == pytest.approx(hard_test, rel=1e-6)
    assert evaluated["samples"] == 10


def assert_same_state(left, right):
    """Recursively compare model, optimizer, scheduler, and saved RNG tensors."""
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_state(a, b)
    else:
        assert left == right


def test_same_smoothing_resume_exactly_matches_uninterrupted_run(prepared, tmp_path):
    assert (CONFIGS / "experiment.json").is_file(), "Missing default smoothing recipe"
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    trainer.main(run_arguments(prepared, full, "--epochs", "2", "--batch-size", "5"))
    trainer.main(run_arguments(prepared, resumed, "--epochs", "2", "--batch-size", "5", "--stop-after-epoch", "1"))
    trainer.main(run_arguments(prepared, resumed, "--epochs", "2", "--batch-size", "5",
                               "--resume", str(resumed / "last.pt")))
    a, b = (torch.load(p / "last.pt", map_location="cpu", weights_only=True) for p in (full, resumed))
    assert a["epoch"] == b["epoch"] == 2
    assert a["config"]["label_smoothing"] == b["config"]["label_smoothing"] == 0.05
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state", "best_validation"):
        assert_same_state(a[key], b[key])
    assert (full / "epochs.csv").read_bytes() == (resumed / "epochs.csv").read_bytes()


@pytest.mark.parametrize("changed", ["0", "0.1"])
def test_changed_smoothing_resume_is_rejected_without_writes(prepared, tmp_path, changed):
    assert (CONFIGS / "experiment.json").is_file(), "Missing default smoothing recipe"
    output = tmp_path / "resume"
    trainer.main(run_arguments(prepared, output, "--epochs", "2", "--stop-after-epoch", "1"))
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="config/recipe differs"):
        trainer.main(run_arguments(prepared, output, "--epochs", "2", "--resume", str(output / "last.pt"),
                                   "--label-smoothing", changed))
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == before
