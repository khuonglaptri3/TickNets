"""A results archive must contain complete, internally consistent experiment evidence."""
import json
import zipfile

import pytest

import train_cifar
from models.cifar_experiment import atomic_json, write_completion
from scripts.aggregate_grid_search import aggregate_results
from scripts.verify_cifar_run import verify_run
from scripts import run_kaggle_phase as phase_runner
from test_cifar_recovery import command, tiny_pipeline


def completed(tiny_pipeline, tmp_path, name="cifar10_sgd_lr010", **settings):
    output = tmp_path / name
    train_cifar.main(command(output, epochs=1, **settings))
    return output


def test_aggregation_reads_val_top1_and_keeps_numeric_precision(tiny_pipeline, tmp_path):
    name = "cifar10_sgd_lr010"
    output = completed(tiny_pipeline, tmp_path, name)
    cfg, metrics, best = verify_run(output)
    frame = aggregate_results(tmp_path, tmp_path / "summary.csv", tmp_path / "summary.md", expected_experiments=[name])
    row = frame.iloc[0]
    assert row["Best Val Acc (%)"] == float(best["val_top1"])
    assert row["Best Val Ep"] == 1
    assert row["Test Loss"] == metrics["loss"]
    assert row["Macro F1"] == metrics["macro_f1"]
    assert frame["Test Loss"].dtype.kind == "f"


def test_missing_runs_cannot_look_like_a_full_grid(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results(tmp_path)
    assert aggregate_results(tmp_path, allow_partial=True).empty


@pytest.mark.parametrize("artifact", ("epochs.csv", "best_val.pt", "test_metrics.json", "confusion_matrix.csv", "test_predictions.csv", "config.json", "last.pt"))
def test_altered_artifact_fails_hash_verification(tiny_pipeline, tmp_path, artifact):
    output = completed(tiny_pipeline, tmp_path)
    with (output / artifact).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_run(output)


@pytest.mark.parametrize("field,value", [("top1", 101), ("macro_f1", 1), ("samples", 7),
                                       ("correct", 99), ("checkpoint_epoch", 2),
                                       ("dataset", "cifar100"), ("checkpoint_sha256", "wrong"), ("loss", 123)])
def test_even_resealed_incorrect_metrics_fail_semantic_verification(tiny_pipeline, tmp_path, field, value):
    output = completed(tiny_pipeline, tmp_path)
    metrics = json.loads((output / "test_metrics.json").read_text())
    metrics[field] = value
    atomic_json(output / "test_metrics.json", metrics)
    write_completion(output, metrics)
    with pytest.raises(ValueError):
        verify_run(output)


def test_synthetic_results_cannot_be_passed_off_as_official(tiny_pipeline, tmp_path):
    output = completed(tiny_pipeline, tmp_path)
    with pytest.raises(ValueError, match="10,000"):
        verify_run(output, official=True)


@pytest.mark.parametrize("phase", (1, 2, 3, 4))
def test_each_phase_exports_verified_manifest_or_recovery(tiny_pipeline, monkeypatch, tmp_path, phase):
    runs, output = tmp_path / "runs", tmp_path / "export"
    original_recipe = phase_runner.phase_recipe
    def recipe(name):
        result = original_recipe(name)
        result["epochs"] = 1
        result["batch_size"] = 4
        result["num_workers"] = 0
        return result
    monkeypatch.setattr(phase_runner, "phase_recipe", recipe)
    names = phase_runner.PHASES[phase]
    for name in names:
        cfg = recipe(name)
        train_cifar.main(command(runs / name, dataset=cfg["dataset"], optimizer=cfg["optimizer"],
                                 learning_rate=cfg["learning_rate"], epochs=1))
    target, complete = phase_runner.package_phase(phase, runs, output, official=False)
    assert complete and target.name.endswith("results.zip")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("phase_manifest.json"))
        assert manifest["status"] == "complete" and manifest["verified_experiments"] == list(names)
        assert "summary.csv" in archive.namelist()
        assert all(name + "/best_val.pt" in archive.namelist() for name in names)
    (runs / names[1] / "completion.json").unlink()
    target, complete = phase_runner.package_phase(phase, runs, output, official=False)
    assert not complete and target.name.endswith("recovery.zip")
    with zipfile.ZipFile(target) as archive:
        assert json.loads(archive.read("phase_manifest.json"))["status"] == "paused"


def test_wrong_phase_config_cannot_be_exported(tiny_pipeline, tmp_path):
    output = completed(tiny_pipeline, tmp_path)
    with pytest.raises(ValueError, match="configuration"):
        verify_run(output, expected_config={"learning_rate": 0.15})


def test_atomic_checkpoint_failure_preserves_previously_committed_file(tmp_path, monkeypatch):
    import torch
    from models.cifar_experiment import atomic_checkpoint, file_sha256
    path = tmp_path / "last.pt"
    atomic_checkpoint(path, {"epoch": 1})
    before = file_sha256(path)
    def interrupted(value, temporary):
        temporary.write_bytes(b"partial write")
        raise OSError("disk failure")
    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="disk"):
        atomic_checkpoint(path, {"epoch": 2})
    assert file_sha256(path) == before
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]
