"""Reject recipe/provenance drift and recover interrupted artifact writes."""
import csv
import json
from pathlib import Path
import subprocess

import pytest
import torch

import train_mid_experiment as trainer
from test_mid_dataset import source_dataset
from test_mid_pipeline import prepared
from test_mid_experiment_training import arguments


@pytest.mark.parametrize("key,value", [("epochs", 2.9), ("num_workers", 0.9), ("seed", True),
    ("learning_rate", True), ("epochs", None), ("epochs", float("inf")), ("learning_rate", None)])
def test_invalid_json_values_exit_as_argument_errors(tmp_path, key, value):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({key: value}))
    with pytest.raises(SystemExit) as error:
        trainer.parse_args(["--config", str(config), "--variant", "Mid32", "--output-dir", "runs/x"])
    assert error.value.code == 2


def test_malformed_json_is_an_argument_error(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text('{"epochs":')
    with pytest.raises(SystemExit) as error:
        trainer.parse_args(["--config", str(config), "--variant", "Mid32", "--output-dir", "runs/x"])
    assert error.value.code == 2


@pytest.mark.parametrize("change", ("modify", "remove", "add"))
@pytest.mark.parametrize("mode", ("resume", "evaluate"))
def test_changed_manifest_rejected_before_writes(prepared, tmp_path, mode, change):
    manifest = prepared / "split_manifest.csv"
    original = manifest.read_bytes()
    if change == "add":
        manifest.unlink()
    output = tmp_path / "train"
    trainer.main(arguments(prepared, output, "--stop-after-epoch", "1"))
    snapshot = (output / "last.pt").read_bytes()
    if change == "remove":
        manifest.unlink()
    else:
        manifest.write_bytes(original + b"\n# changed provenance\n")
    destination = output if mode == "resume" else tmp_path / "evaluation"
    flags = arguments(prepared, destination, "--" + mode, str(output / "last.pt"))
    with pytest.raises(ValueError, match="manifest"):
        trainer.main(flags)
    assert (output / "last.pt").read_bytes() == snapshot
    if mode == "evaluate":
        assert not destination.exists()


def test_source_revision_comes_from_trainer_checkout(prepared, tmp_path, monkeypatch):
    source = Path(trainer.__file__).resolve().parent
    expected = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "run"
    trainer.main(arguments(prepared, output, "--epochs", "1"))
    config = json.loads((output / "config.json").read_text())
    assert config["git_revision"] == expected
    assert isinstance(config["git_dirty"], bool)
    assert len(config["source_hashes"]["train_mid_experiment.py"]) == 64


def test_failed_last_write_resumes_from_committed_epoch(prepared, tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    save = trainer.save_checkpoint
    def failing_save(path, state):
        if path.name == "last.pt" and state["epoch"] == 2:
            raise OSError("injected last write failure")
        save(path, state)
    with monkeypatch.context() as context:
        context.setattr(trainer, "save_checkpoint", failing_save)
        with pytest.raises(OSError, match="injected"):
            trainer.main(arguments(prepared, output))
    trainer.main(arguments(prepared, output, "--resume", str(output / "last.pt")))
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    with (output / "epochs.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [1, 2]


def test_missing_best_artifact_is_restored_before_worse_epoch(prepared, tmp_path, monkeypatch):
    output = tmp_path / "interrupted_best"
    save, run_epoch = trainer.save_checkpoint, trainer.run_epoch
    validations = []
    def controlled_epoch(*args, **kwargs):
        result = run_epoch(*args, **kwargs)
        optimizer = args[4] if len(args) > 4 else kwargs.get("optimizer")
        if optimizer is None:
            validations.append(True)
            result.update(top1=80.0 if len(validations) == 1 else 20.0,
                          loss=0.1 if len(validations) == 1 else 2.0)
        return result
    monkeypatch.setattr(trainer, "run_epoch", controlled_epoch)
    def failing_best(path, state):
        if path.name == "best_val.pt":
            raise OSError("injected best write failure")
        save(path, state)
    with monkeypatch.context() as context:
        context.setattr(trainer, "save_checkpoint", failing_best)
        with pytest.raises(OSError, match="injected"):
            trainer.main(arguments(prepared, output))
    assert not (output / "best_val.pt").exists()
    committed = torch.load(output / "last.pt", weights_only=True)
    trainer.main(arguments(prepared, output, "--resume", str(output / "last.pt")))
    best = torch.load(output / "best_val.pt", weights_only=True)
    assert best["epoch"] == 1 and best["best_validation"]["top1"] == 80
    for name in best["model_state_dict"]:
        assert torch.equal(best["model_state_dict"][name], committed["model_state_dict"][name]), name


def test_completed_epoch_can_recover_best_without_more_training(prepared, tmp_path, monkeypatch):
    output = tmp_path / "completed"
    save = trainer.save_checkpoint
    def fail_best(path, state):
        if path.name == "best_val.pt":
            raise OSError("injected final best write failure")
        save(path, state)
    with monkeypatch.context() as context:
        context.setattr(trainer, "save_checkpoint", fail_best)
        with pytest.raises(OSError, match="injected"):
            trainer.main(arguments(prepared, output, "--epochs", "1"))
    snapshot = (output / "last.pt").read_bytes()
    result = trainer.main(arguments(prepared, output, "--epochs", "1", "--resume", str(output / "last.pt")))
    assert result["epoch"] == 1
    assert (output / "last.pt").read_bytes() == snapshot
    assert torch.load(output / "best_val.pt", weights_only=True)["epoch"] == 1
    with (output / "epochs.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
