"""Verify all eight final runs, preserve complete outputs, and render report assets.

No training or inference is performed. All figures derive from recorded artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cifar_experiment import file_sha256
from scripts.aggregate_grid_search import EXPECTED_EXPERIMENTS, aggregate_results
from scripts.verify_cifar_run import verify_run
from scripts.run_kaggle_phase import phase_recipe

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def copy_run_archive(source: Path, destination: Path) -> dict:
    """Copy every file, including last.pt and phase manifests, without overwrites."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and archive must be separate, non-nested directories")
    entries = sorted(source.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Run archives must not contain symlinks")
    files = [path for path in entries if path.is_file()]
    if not files:
        raise ValueError("No run output to archive")
    hashes = {path.relative_to(source).as_posix(): file_sha256(path) for path in files}
    # Preflight every existing destination before writing any new files.
    for name, digest in hashes.items():
        target = destination / name
        if target.exists() and (not target.is_file() or file_sha256(target) != digest):
            raise ValueError(f"Archive file differs; refusing to overwrite: {name}")
    for name, digest in hashes.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source / name, target)
        if file_sha256(target) != digest:
            raise ValueError(f"Archive copy verification failed: {name}")
    return hashes


def classification_report(matrix, labels):
    """Equivalent per-class and aggregate metrics; undefined ratios become zero."""
    matrix = np.asarray(matrix, dtype=np.int64)
    tp, support, predicted = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp)), where=(support + predicted) != 0)
    total = int(support.sum())
    if total <= 0:
        raise ValueError("Classification report requires test samples")
    frame = pd.DataFrame({"precision": precision, "recall": recall, "f1-score": f1, "support": support}, index=labels)
    accuracy = float(tp.sum() / total)
    frame.loc["accuracy"] = [accuracy, accuracy, accuracy, total]
    frame.loc["macro avg"] = [precision.mean(), recall.mean(), f1.mean(), total]
    frame.loc["weighted avg"] = [np.average(values, weights=support) for values in (precision, recall, f1)] + [total]
    frame["support"] = frame["support"].astype(int)
    frame.index.name = "class"
    return frame


def select_by_validation(frame):
    ordered = frame.sort_values(["Best Val Acc (%)", "Best Val Loss", "Experiment"], ascending=[False, True, True])
    return ordered.drop_duplicates("Dataset").set_index("Dataset")["Experiment"].to_dict()


def export_run_assets(run, output, config, best):
    output.mkdir(parents=True, exist_ok=True)
    name = run.name
    history = pd.read_csv(run / "epochs.csv")
    shutil.copy2(run / "epochs.csv", output / f"{name}_epochs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric, title, ylabel in zip(axes, ("loss", "top1"), ("Loss", "Accuracy"), ("Cross-entropy", "Top-1 (%)")):
        ax.plot(history.epoch, history[f"train_{metric}"], label="Train", linewidth=1.4)
        ax.plot(history.epoch, history[f"val_{metric}"], label="Validation", linewidth=1.4)
        ax.axvline(int(best["epoch"]), color="#666666", linestyle="--", linewidth=1, label=f"Selected epoch {best['epoch']}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(name.replace("_", " "))
    fig.savefig(output / f"{name}_learning_curves.png", dpi=180)
    plt.close(fig)

    matrix = np.loadtxt(run / "confusion_matrix.csv", delimiter=",", dtype=np.int64)
    nc = config["num_classes"]
    labels = CIFAR10_CLASSES if nc == 10 else [f"class_{i:03d}" for i in range(nc)]
    # Keep the raw matrix CSV byte-for-byte; class-map CSV explains its row/column order.
    shutil.copy2(run / "confusion_matrix.csv", output / f"{name}_confusion_matrix.csv")
    classification_report(matrix, labels).to_csv(output / f"{name}_classification_report.csv")
    fig, ax = plt.subplots(figsize=(7, 6) if nc == 10 else (12, 10), constrained_layout=True)
    if nc == 10:
        displayed = matrix
        plot = ax.imshow(displayed, cmap="Blues")
        ax.set_xticks(range(nc), labels, rotation=45, ha="right")
        ax.set_yticks(range(nc), labels)
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > displayed.max() / 2 else "black")
        color_label = "Test samples"
    else:
        displayed = 100 * matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        plot = ax.imshow(displayed, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(0, nc, 5))
        ax.set_yticks(range(0, nc, 5))
        color_label = "Percentage of true class (%)"
    ax.set(xlabel="Predicted class" + (" ID" if nc == 100 else ""),
           ylabel="True class" + (" ID" if nc == 100 else ""), title=name.replace("_", " "))
    fig.colorbar(plot, ax=ax, label=color_label)
    fig.savefig(output / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)


def build_report(runs_dir, output_dir, archive_dir=None):
    # Verify originals before archiving or producing a report that looks complete.
    verified = {name: verify_run(runs_dir / name, official=True, expected_config=phase_recipe(name))
                for name in EXPECTED_EXPERIMENTS}
    input_files = [path for name in EXPECTED_EXPERIMENTS
                   for path in sorted((runs_dir / name).rglob("*")) if path.is_file()]
    input_files += [path for path in sorted(runs_dir.glob("phase*")) if path.is_file()]
    archive_hashes = copy_run_archive(runs_dir, archive_dir) if archive_dir else {
        path.relative_to(runs_dir).as_posix(): file_sha256(path) for path in input_files
    }
    if archive_dir:
        for name in EXPECTED_EXPERIMENTS:
            verify_run(archive_dir / name, official=True)
    assets = output_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    summary = aggregate_results(runs_dir, output_dir / "grid_search_summary.csv", output_dir / "grid_search_summary.md", official=True)
    summary["Phase"] = [i // 2 + 1 for i in range(len(EXPECTED_EXPERIMENTS))]
    summary["Best Val Loss"] = [float(verified[name][2]["val_loss"]) for name in summary.Experiment]
    selected = select_by_validation(summary)
    summary["Selected"] = [name in selected.values() for name in summary.Experiment]
    summary.to_csv(assets / "final_required_summary.csv", index=False)
    selection = {}
    profiles = {}
    for dataset, name in selected.items():
        config, metrics, best = verified[name]
        selection[dataset] = {"experiment": name, "checkpoint": f"{name}/best_val.pt",
                              "checkpoint_sha256": metrics["checkpoint_sha256"],
                              "best_epoch": int(best["epoch"]), "validation_top1": float(best["val_top1"]),
                              "validation_loss": float(best["val_loss"]), "test_top1": metrics["top1"],
                              "selection_rule": "Highest validation top1; ties use lowest validation loss, then run name"}
        profiles[dataset] = {key: config[key] for key in (
            "num_classes", "learnable_parameters", "flops_forward", "gflops_forward",
            "counting_convention", "excluded_operations", "architecture_revision")}
    for name, (config, metrics, best) in verified.items():
        export_run_assets(runs_dir / name, assets, config, best)
    for dataset, group in summary.groupby("Dataset", sort=False):
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        x = np.arange(len(group))
        for offset, key, color, label in [(-0.2, "Best Val Acc (%)", "#4878b5", "Validation"),
                                           (0.2, "Test Top-1 (%)", "#59a14f", "Test (reported only)")]:
            bars = ax.bar(x + offset, group[key], width=0.38, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        labels = [f"{row.Optimizer} lr={row.LR:g}" + ("\n(selected by validation)" if row.Selected else "") for row in group.itertuples()]
        ax.set_xticks(x, labels, fontsize=9)
        ax.set(title=f"{dataset} — TickNet-L, seed 42, 200 epochs", ylabel="Top-1 accuracy (%)", ylim=(0, 110))
        ax.legend(loc="upper center", ncols=2, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(assets / f"{dataset.lower()}_optimizer_comparison.png", dpi=180)
        plt.close(fig)
        group.to_csv(assets / f"{dataset.lower()}_comparison.csv", index=False)
    class_rows = [{"dataset": "cifar10", "class_id": i, "label": label} for i, label in enumerate(CIFAR10_CLASSES)]
    class_rows += [{"dataset": "cifar100", "class_id": i, "label": f"class_{i:03d}"} for i in range(100)]
    pd.DataFrame(class_rows).to_csv(assets / "class_mapping.csv", index=False)
    for path, value in ((output_dir / "selected_checkpoints.json", selection),
                        (assets / "model_profiles_final.json", profiles),
                        (output_dir / "archive_manifest.json", {"algorithm": "sha256", "files": archive_hashes})):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Verified {len(verified)} runs; archived/hashed {len(archive_hashes)} files; report: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--archive-dir", type=Path, help="Copy all outputs here; refuse conflicting files")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/experiments"))
    args = parser.parse_args()
    build_report(args.runs_dir, args.output_dir, args.archive_dir)


if __name__ == "__main__":
    main()
