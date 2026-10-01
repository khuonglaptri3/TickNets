"""Validation selection, explicit test access, and resumable experiments."""
import importlib
import json

import pytest
import torch

from test_mid_pipeline import prepared
from test_mid_dataset import source_dataset
from test_mid_experiment_data import paired_data


def trainer():
    assert importlib.util.find_spec("train_mid_experiment") is not None, "experimental trainer is missing"
    return importlib.import_module("train_mid_experiment")


def arguments(prepared, output, *extra):
    return ["--data-root", str(prepared), "--variant", "Mid32", "--output-dir", str(output),
            "--epochs", "2", "--batch-size", "5", "--device", "cpu", "--threads", "2",
            "--num-workers", "0", "--no-augment", *extra]


def test_training_selects_validation_checkpoint_without_evaluating_test(prepared, tmp_path):
    module = trainer()
    output = tmp_path / "run"
    result = module.main(arguments(prepared, output))
    assert result["epoch"] == 2
    assert not (output / "test_metrics.json").exists()
    config = json.loads((output / "config.json").read_text())
    assert config["train_samples"] == 15 and config["val_samples"] == 5
    assert (output / "best_val.pt").is_file() and (output / "last.pt").is_file()
    checkpoint = torch.load(output / "best_val.pt", weights_only=True)
    assert checkpoint["config"]["architecture_revision"] == "ticknet-l-v1"
    evaluation = module.main(["--data-root", str(prepared), "--variant", "Mid32", "--evaluate",
                              str(output / "best_val.pt"), "--output-dir", str(tmp_path / "eval"),
                              "--device", "cpu", "--threads", "2"])
    assert evaluation["samples"] == 10
    assert (tmp_path / "eval" / "test_predictions.csv").is_file()
    assert (tmp_path / "eval" / "confusion_matrix.csv").is_file()
    provenance = json.loads((tmp_path / "eval" / "config.json").read_text())["evaluation_checkpoint"]
    assert provenance["training_config"] == checkpoint["config"]
    assert provenance["epoch"] == checkpoint["epoch"] and provenance["role"] == "best_val"
    import hashlib
    assert provenance["sha256"] == hashlib.sha256((output / "best_val.pt").read_bytes()).hexdigest()


def test_epoch_boundary_resume_matches_uninterrupted_training(prepared, tmp_path):
    module = trainer()
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    module.main(arguments(prepared, full))
    module.main(arguments(prepared, resumed, "--stop-after-epoch", "1"))
    module.main(arguments(prepared, resumed, "--resume", str(resumed / "last.pt")))
    a, b = (torch.load(p / "last.pt", weights_only=True) for p in (full, resumed))
    assert a["epoch"] == b["epoch"] == 2
    assert a["scheduler_state_dict"] == b["scheduler_state_dict"]
    for name in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]), name
    assert (full / "epochs.csv").read_bytes() == (resumed / "epochs.csv").read_bytes()


def test_full_train_has_no_validation_selection(prepared, tmp_path):
    module = trainer()
    output = tmp_path / "full_train"
    module.main(arguments(prepared, output, "--epochs", "1", "--full-train"))
    config = json.loads((output / "config.json").read_text())
    assert config["train_samples"] == 20 and config["val_samples"] == 0
    assert not (output / "best_val.pt").exists()
    assert not (output / "test_metrics.json").exists()


def test_resume_rejects_changed_recipe_before_writing(prepared, tmp_path):
    module = trainer()
    output = tmp_path / "resume"
    module.main(arguments(prepared, output, "--stop-after-epoch", "1"))
    original = (output / "epochs.csv").read_bytes()
    with pytest.raises(ValueError, match="config|recipe"):
        module.main(arguments(prepared, output, "--resume", str(output / "last.pt"), "--learning-rate", "0.15"))
    assert (output / "epochs.csv").read_bytes() == original


def test_config_unknown_keys_and_cli_overrides(tmp_path):
    module = trainer()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"learning_rate": 0.15, "epochs": 3}))
    args = module.parse_args(["--config", str(config), "--variant", "Mid32", "--output-dir", "runs/x", "--epochs", "5"])
    assert args.learning_rate == 0.15 and args.epochs == 5
    config.write_text(json.dumps({"learning_rait": 0.15}))
    with pytest.raises(SystemExit):
        module.parse_args(["--config", str(config), "--variant", "Mid32", "--output-dir", "runs/x"])


def test_resume_restores_worker_augmented_training(paired_data, tmp_path):
    module = trainer()
    full, resumed = tmp_path / "workers_full", tmp_path / "workers_resumed"
    def flags(output, *extra):
        return ["--data-root", str(paired_data), "--variant", "Mid32", "--output-dir", str(output),
                "--epochs", "2", "--batch-size", "8", "--device", "cpu", "--threads", "2",
                "--num-workers", "2", *extra]
    module.main(flags(full))
    module.main(flags(resumed, "--stop-after-epoch", "1"))
    module.main(flags(resumed, "--resume", str(resumed / "last.pt")))
    a, b = (torch.load(path / "last.pt", weights_only=True) for path in (full, resumed))
    for name in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][name], b["model_state_dict"][name]), name
    assert (full / "epochs.csv").read_bytes() == (resumed / "epochs.csv").read_bytes()
