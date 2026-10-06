"""Numerical reporting and complete, immutable run copies."""
import importlib

import numpy as np
import pandas as pd
import pytest


def reporter():
    assert importlib.util.find_spec("scripts.build_final_report") is not None
    return importlib.import_module("scripts.build_final_report")


def test_classification_report_computes_precision_recall_f1_and_zero_division():
    report = reporter().classification_report(np.array([[8, 2, 0], [1, 9, 0], [0, 0, 0]]), ["a", "b", "c"])
    assert report.loc["a", "precision"] == pytest.approx(8 / 9)
    assert report.loc["a", "recall"] == 0.8
    assert report.loc["a", "f1-score"] == pytest.approx(16 / 19)
    assert report.loc["b", "f1-score"] == pytest.approx(18 / 21)
    assert (report.loc["c"] == 0).all()
    assert report.loc["accuracy", "f1-score"] == 0.85
    assert report.loc["macro avg", "f1-score"] == pytest.approx((16 / 19 + 18 / 21) / 3)
    assert report.loc["weighted avg", "support"] == 20


def test_selection_uses_validation_accuracy_then_loss_not_test_accuracy():
    frame = pd.DataFrame([
        {"Experiment": "a", "Dataset": "CIFAR10", "Best Val Acc (%)": 91, "Best Val Loss": 0.3, "Test Top-1 (%)": 98},
        {"Experiment": "b", "Dataset": "CIFAR10", "Best Val Acc (%)": 92, "Best Val Loss": 0.4, "Test Top-1 (%)": 94},
        {"Experiment": "c", "Dataset": "CIFAR10", "Best Val Acc (%)": 92, "Best Val Loss": 0.2, "Test Top-1 (%)": 90},
        {"Experiment": "d", "Dataset": "CIFAR100", "Best Val Acc (%)": 70, "Best Val Loss": 1.0, "Test Top-1 (%)": 71},
    ])
    assert reporter().select_by_validation(frame) == {"CIFAR10": "c", "CIFAR100": "d"}


def test_archive_copies_all_outputs_and_checks_bytes_before_overwrite(tmp_path):
    report = reporter()
    source, destination = tmp_path / "runs", tmp_path / "archive"
    (source / "run" / "nested").mkdir(parents=True)
    for name in ("best_val.pt", "last.pt", "config.json", "progress.json"):
        (source / "run" / name).write_bytes(name.encode())
    (source / "run" / "nested" / "extra.bin").write_bytes(b"anything")
    (source / "phase.csv").write_bytes(b"summary")
    manifest = report.copy_run_archive(source, destination)
    assert len(manifest) == 6
    for name in manifest:
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    assert report.copy_run_archive(source, destination) == manifest
    (destination / "run" / "last.pt").write_bytes(b"do not overwrite me")
    with pytest.raises(ValueError, match="differs"):
        report.copy_run_archive(source, destination)
    assert (destination / "run" / "last.pt").read_bytes() == b"do not overwrite me"


def test_report_exports_plots_and_reports_from_verified_fixture(tiny_pipeline, tmp_path):
    # Fixture artifacts are deliberately non-official and cannot enter the CLI final report.
    from test_cifar_results import completed
    run = completed(tiny_pipeline, tmp_path)
    output = tmp_path / "report_assets"
    report = reporter()
    from scripts.verify_cifar_run import verify_run
    config, metrics, best = verify_run(run)
    report.export_run_assets(run, output, config, best)
    for suffix in ("learning_curves.png", "confusion_matrix.png", "confusion_matrix.csv", "classification_report.csv", "epochs.csv"):
        assert (output / f"{run.name}_{suffix}").stat().st_size > 0
    table = pd.read_csv(output / f"{run.name}_classification_report.csv", index_col=0)
    assert table.loc["macro avg", "f1-score"] == pytest.approx(metrics["macro_f1"])


from test_cifar_recovery import tiny_pipeline
