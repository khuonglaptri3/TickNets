"""Validate a completed run against its logs, checkpoint and sample predictions."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from models.cifar_experiment import file_sha256, verify_artifact_hashes


def verify_run(path, *, official=False, expected_config=None):
    path = Path(path)
    completion = verify_artifact_hashes(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    with (path / "epochs.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    if len(history) != config["epochs"] or [int(r["epoch"]) for r in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Incomplete or duplicated epoch history")
    best = max(history, key=lambda row: (float(row["val_top1"]), -float(row["val_loss"])))
    best_epoch = int(best["epoch"])
    if metrics.get("checkpoint_epoch") != best_epoch or completion["checkpoint_epoch"] != best_epoch:
        raise ValueError("Test checkpoint was not selected by validation")
    if metrics.get("checkpoint_sha256") != file_sha256(path / "best_val.pt"):
        raise ValueError("Test checkpoint hash differs from best_val.pt")
    if config["learnable_parameters"] > 6_000_000 or config["flops_forward"] >= 1_000_000_000:
        raise ValueError("Exam budget exceeded")
    if metrics.get("dataset") != config["dataset"] or metrics.get("model") != config["model"]:
        raise ValueError("Metric identity differs from config")
    if metrics.get("test_data_sha256") != config["data_evidence"]["test"]["sha256"]:
        raise ValueError("Test dataset provenance differs from config")
    if expected_config:
        for key, value in expected_config.items():
            if config.get(key) != value:
                raise ValueError(f"Phase configuration differs: {key}")
    nc = config["num_classes"]
    with (path / "confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
        matrix = [[int(v) for v in row] for row in csv.reader(handle)]
    if len(matrix) != nc or any(len(row) != nc or any(v < 0 for v in row) for row in matrix):
        raise ValueError("Invalid confusion matrix")
    recomputed = [[0] * nc for _ in range(nc)]
    with (path / "test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    for i, row in enumerate(predictions):
        target, prediction = int(row["target"]), int(row["prediction"])
        if int(row["sample_index"]) != i or not (0 <= target < nc and 0 <= prediction < nc):
            raise ValueError("Invalid prediction indices or labels")
        recomputed[target][prediction] += 1
    losses = [float(row["negative_log_likelihood"]) for row in predictions]
    if any(not math.isfinite(loss) or loss < 0 for loss in losses):
        raise ValueError("Invalid per-sample test loss")
    samples = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(nc))
    if samples <= 0 or recomputed != matrix or metrics["samples"] != samples or metrics["correct"] != correct:
        raise ValueError("Prediction and confusion matrix totals disagree")
    if config["data_evidence"]["test"]["samples"] != samples:
        raise ValueError("Evaluation omitted or repeated test samples")
    f1 = sum(2 * matrix[i][i] / max(1, sum(matrix[i]) + sum(r[i] for r in matrix)) for i in range(nc)) / nc
    for key, expected in (("top1", 100 * correct / samples), ("macro_f1", f1)):
        if not math.isclose(metrics[key], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Incorrect reported {key}")
    if not math.isfinite(metrics["loss"]) or metrics["loss"] < 0:
        raise ValueError("Invalid test loss")
    if not math.isclose(metrics["loss"], math.fsum(losses) / samples, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Reported test loss disagrees with per-sample loss")
    if official:
        if samples != 10_000 or config["num_classes"] != (10 if config["dataset"] == "cifar10" else 100):
            raise ValueError("Official CIFAR test requires 10,000 samples and matching class count")
        if config["epochs"] != 200 or config["val_fraction"] != 0.1:
            raise ValueError("Phase requires the frozen 200-epoch, 90/10 protocol")
        if any(int(r["train_samples"]) != 45_000 or int(r["val_samples"]) != 5_000 for r in history):
            raise ValueError("Official train/validation sample counts differ")
    return config, metrics, best
