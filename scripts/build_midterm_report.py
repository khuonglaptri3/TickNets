"""Validate saved Midterm evidence and rebuild the report's figures/tables.

With --evaluate, test inference is run on the two pinned checkpoints first.
Without that flag, existing inference evidence is verified and reused.
The script never trains or selects another checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/MIDTERM_REPORT.md"
RESULTS = ROOT / "docs/05_results"
DOSSIER = RESULTS / "experiment_midterm_smoothing_lr"
OUTPUT = RESULTS / "selected_checkpoints"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
INFERENCE = OUTPUT / "inference"
CLASSES = ["bird", "cat", "dog", "frog", "horse"]
RUNS = [
    ("Mid224", "l_mid224_smoothing010", 176, "checkpoint224.py"),
    ("Mid32", "l_mid32_lr015", 186, "checkpoint32.py"),
]
SOURCES: set[Path] = set()


def csv_rows(path):
    SOURCES.add(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(map(str, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def replace_block(text, name, body):
    pattern = rf"<!-- BEGIN {name} -->.*?<!-- END {name} -->"
    replacement = f"<!-- BEGIN {name} -->\n{body}\n<!-- END {name} -->"
    text, count = re.subn(pattern, lambda _: replacement, text, flags=re.S)
    if count != 1:
        raise ValueError(f"Expected one report block {name}; found {count}")
    return text


def save_figure(fig, filename):
    fig.savefig(FIGURES / filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_pinned_checkpoints():
    """Use the same model, test transforms and batch size as the entrypoints."""
    sys.path.insert(0, str(ROOT))
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from models.mid_data import RequireImageSize
    from models.mid_models import build_mid_model

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INFERENCE.mkdir(parents=True, exist_ok=True)
    for variant, name, epoch, entrypoint in RUNS:
        checkpoint = ROOT / "checkpoints" / name / "best_val.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = payload["config"]
        assert cfg["variant"] == variant and cfg["model"] == "l"
        assert payload["epoch"] == epoch
        model = build_mid_model("l", num_classes=5, variant=variant)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        archived = torch.load(DOSSIER / "checkpoints" / name / "best_val.pt", map_location="cpu", weights_only=True)
        a, b = payload["model_state_dict"], archived["model_state_dict"]
        equal_weights = a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
        model = model.to(device).eval()
        size = 224 if variant == "Mid224" else 32
        dataset = datasets.ImageFolder(ROOT / "data" / variant / "test",
                                       transform=transforms.Compose([RequireImageSize(size), transforms.ToTensor()]))
        assert dataset.class_to_idx == dict(zip(CLASSES, range(5))) and len(dataset) == 250
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        predictions = []
        cm = np.zeros((5, 5), dtype=int)
        offset = 0
        with torch.inference_mode():
            for images, targets in loader:
                logits = model(images.to(device))
                losses = torch.nn.functional.cross_entropy(logits, targets.to(device), reduction="none").cpu().tolist()
                labels = logits.argmax(dim=1).cpu().tolist()
                for target, prediction, nll in zip(targets.tolist(), labels, losses):
                    path = Path(dataset.samples[offset][0])
                    predictions.append(dict(path=path.relative_to(ROOT).as_posix(), target=target,
                                            target_name=CLASSES[target], prediction=prediction,
                                            prediction_name=CLASSES[prediction], correct=target == prediction, nll=nll))
                    cm[target, prediction] += 1
                    offset += 1
        f1 = 2 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1))
        metrics = {"top1": 100 * int(np.trace(cm)) / 250, "samples": 250,
                   "loss": sum(r["nll"] for r in predictions) / 250, "macro_f1": float(f1.mean())}
        stem = variant.lower()
        with (INFERENCE / f"{stem}_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        with (INFERENCE / f"{stem}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual_class", *(f"predicted_{c}" for c in CLASSES)])
            writer.writerows([[c, *cm[i].tolist()] for i, c in enumerate(CLASSES)])
        with (INFERENCE / f"{stem}_classification_report.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_name", "precision", "recall", "f1-score", "support"])
            writer.writerows([[c, cm[i,i] / cm[:,i].sum(), cm[i,i] / cm[i,:].sum(), f1[i], 50] for i, c in enumerate(CLASSES)])
        (INFERENCE / f"{stem}_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        data_hash = hashlib.sha256()
        for filename, _ in dataset.samples:
            path = Path(filename)
            data_hash.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
            data_hash.update(path.read_bytes())
        metadata = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "variant": variant,
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                    "selected_epoch": epoch, "entrypoint": entrypoint, "batch_size": 16,
                    "device": str(device), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                    "pinned_and_archived_state_dict_equal": equal_weights,
                    "ordered_test_paths_and_image_bytes_sha256": data_hash.hexdigest()}
        (INFERENCE / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Evaluated {variant}: {metrics['top1']:.2f}% ({int(np.trace(cm))}/250), loss={metrics['loss']:.8f}", flush=True)


def load_run(variant, name, expected_epoch, entrypoint):
    pinned = ROOT / "checkpoints" / name
    config = read_json(pinned / "config.json")
    epochs = csv_rows(pinned / "epochs.csv")
    assert [int(row["epoch"]) for row in epochs] == list(range(1, 201)), name
    assert config["variant"] == variant and config["architecture_revision"] == "ticknet-l-v1"
    assert (config["train_samples"], config["val_samples"], config["test_samples"]) == (22500, 2500, 250)
    best = max(epochs, key=lambda row: (float(row["val_top1"]), -float(row["val_loss"])))
    assert int(best["epoch"]) == expected_epoch
    stem = variant.lower()
    metadata = read_json(INFERENCE / f"{stem}_metadata.json")
    assert metadata["checkpoint_sha256"] == sha256(pinned / "best_val.pt")
    metrics = read_json(INFERENCE / f"{stem}_test_metrics.json")
    cm_rows = csv_rows(INFERENCE / f"{stem}_confusion_matrix.csv")
    assert [row["actual_class"] for row in cm_rows] == CLASSES
    cm = np.array([[int(row[f"predicted_{c}"]) for c in CLASSES] for row in cm_rows])
    assert cm.shape == (5, 5) and np.all(cm.sum(axis=1) == 50)
    predictions = csv_rows(INFERENCE / f"{stem}_predictions.csv")
    reconstructed = np.zeros((5, 5), dtype=int)
    for row in predictions:
        reconstructed[int(row["target"]), int(row["prediction"])] += 1
    assert len(predictions) == metrics["samples"] == 250
    assert np.array_equal(reconstructed, cm)
    accuracy = 100 * int(np.trace(cm)) / int(cm.sum())
    precision = np.diag(cm) / cm.sum(axis=0)
    recall = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1))
    assert math.isclose(accuracy, metrics["top1"], abs_tol=1e-10)
    assert math.isclose(float(f1.mean()), metrics["macro_f1"], abs_tol=1e-10)
    classification = csv_rows(INFERENCE / f"{stem}_classification_report.csv")
    for index, row in enumerate(classification[:5]):
        assert row["class_name"] == CLASSES[index]
        assert math.isclose(float(row["precision"]), precision[index], abs_tol=1e-10)
        assert math.isclose(float(row["recall"]), recall[index], abs_tol=1e-10)
        assert math.isclose(float(row["f1-score"]), f1[index], abs_tol=1e-10)
    log_path = ROOT / "docs/08_appendix/validation" / entrypoint.replace(".py", ".txt")
    SOURCES.add(log_path)
    log = log_path.read_text(encoding="utf-8")
    recorded_top1 = float(re.search(r"Test Top-1 Accuracy\s*:\s*([\d.]+)%", log)[1])
    log_cm = []
    for line in log.splitlines():
        match = re.search(r"\|\s*\[([\d, ]+)\]\s*$", line)
        if match:
            log_cm.append([int(x) for x in match[1].split(",")])
    assert recorded_top1 == accuracy and np.array_equal(np.array(log_cm), cm)
    shutil.copyfile(pinned / "epochs.csv", TABLES / f"{variant.lower()}_epochs.csv")
    for filename in ["confusion_matrix.csv", "classification_report.csv", "predictions.csv"]:
        shutil.copyfile(INFERENCE / f"{stem}_{filename}", TABLES / f"{stem}_{filename}")
    checkpoint = pinned / "best_val.pt"
    SOURCES.add(checkpoint)
    return dict(variant=variant, name=name, config=config, epochs=epochs, best=best,
                metrics=metrics, cm=cm, precision=precision, recall=recall, f1=f1,
                predictions=predictions, checkpoint=checkpoint, entrypoint=entrypoint, metadata=metadata)


def learning_curves(run):
    epochs = run["epochs"]
    x = [int(r["epoch"]) for r in epochs]
    selected = int(run["best"]["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), layout="constrained")
    smooth = run["config"]["label_smoothing"]
    for ax, metric, label in zip(axes, ["loss", "top1"], ["Cross-entropy", "Top-1 accuracy (%)"]):
        train_label = f"Train (smoothing={smooth:g})" if metric == "loss" and smooth else "Train"
        ax.plot(x, [float(r[f"train_{metric}"]) for r in epochs], color="#2563eb", label=train_label, linewidth=1.6)
        ax.plot(x, [float(r[f"val_{metric}"]) for r in epochs], color="#d97706", label="Validation (hard labels)", linewidth=1.6)
        ax.axvline(selected, color="#334155", linestyle="--", linewidth=1.2, label=f"Selected epoch {selected}")
        ax.scatter([selected], [float(run["best"][f"val_{metric}"])], color="#d97706", zorder=5)
        ax.set(xlabel="Epoch", ylabel=label, xlim=(1, 200))
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{run['variant']} | TickNet-L v1 | full 200-epoch history", fontsize=14, fontweight="bold")
    save_figure(fig, f"{run['variant'].lower()}_learning_curves.png")


def confusion_matrix(run):
    fig, ax = plt.subplots(figsize=(6.5, 5.7), layout="constrained")
    im = ax.imshow(run["cm"], cmap="Blues", vmin=0, vmax=50)
    for i in range(5):
        for j in range(5):
            value = int(run["cm"][i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color="white" if value > 25 else "#0f172a", fontweight="bold" if value else "normal")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="Ground-truth class",
           title=f"{run['variant']}: {run['metrics']['top1']:.2f}% Top-1 ({np.trace(run['cm'])}/250)")
    fig.colorbar(im, ax=ax, label="Number of images", shrink=0.85)
    save_figure(fig, f"{run['variant'].lower()}_confusion_matrix.png")


def image_path(variant, row):
    filename = row["path"].replace("\\", "/").rsplit("/", 1)[-1]
    path = ROOT / "data" / variant / "test" / CLASSES[int(row["target"])] / filename
    assert path.is_file(), path
    SOURCES.add(path)
    return path


def error_examples(run):
    errors = [r for r in run["predictions"] if r["target"] != r["prediction"]]
    columns = min(4, len(errors))
    rows = math.ceil(len(errors) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.2 * rows), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors):
        with Image.open(image_path(run["variant"], row)) as image:
            ax.imshow(image.convert("RGB"), interpolation="nearest")
        ax.set_title(f"True: {CLASSES[int(row['target'])]}\nPredicted: {CLASSES[int(row['prediction'])]}", fontsize=10)
    fig.suptitle(f"{run['variant']}: all {len(errors)} test errors", fontweight="bold")
    save_figure(fig, f"{run['variant'].lower()}_errors.png")


def dataset_examples():
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.7), layout="constrained")
    for column, name in enumerate(CLASSES):
        filename = sorted((ROOT / "data/Mid224/train" / name).glob("*.jpeg"))[0].name
        for row, variant in enumerate(["Mid224", "Mid32"]):
            path = ROOT / "data" / variant / "train" / name / filename
            assert path.is_file()
            SOURCES.add(path)
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("RGB"), interpolation="nearest")
            axes[row, column].set_title(f"{name} | {variant}", fontsize=11)
            axes[row, column].axis("off")
    fig.suptitle("Paired training examples | same sample at two native resolutions", fontweight="bold")
    save_figure(fig, "dataset_examples.png")


def comparison_figures(runs, validation_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = np.arange(5)
    for ax, run in zip(axes, runs):
        for offset, key, color in [(-0.25, "precision", "#2563eb"), (0, "recall", "#d97706"), (0.25, "f1", "#059669")]:
            ax.bar(x + offset, run[key], width=0.24, label=key.upper() if key == "f1" else key.title(), color=color)
        ax.set(xticks=x, xticklabels=CLASSES, ylim=(0, 1.08), ylabel="Score", title=run["variant"])
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Per-class test metrics | 50 test images per class", fontweight="bold")
    save_figure(fig, "per_class_metrics.png")
    fig, ax = plt.subplots(figsize=(10, 4.1), layout="constrained")
    for run, color in zip(runs, ["#2563eb", "#059669"]):
        ax.plot([int(r["epoch"]) for r in run["epochs"]], [float(r["learning_rate"]) for r in run["epochs"]], color=color,
                label=f"{run['variant']}, initial LR={run['config']['learning_rate']}", linewidth=2)
    ax.set(xlabel="Epoch", ylabel="Learning rate used during epoch", xlim=(1, 200), ylim=(0, 0.16), title="Cosine learning-rate schedules")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, "learning_rate_schedule.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for index, (ax, variant) in enumerate(zip(axes, ["Mid32", "Mid224"])):
        labels = [r[0] for r in validation_rows]
        values = [float(r[1 + index]) for r in validation_rows]
        bars = ax.barh(labels, values, color="#2563eb", alpha=0.85)
        ax.invert_yaxis()
        ax.set(xlim=(0, 103), xlabel="Best validation Top-1 (%)", title=variant)
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Independent recipe comparisons | validation only | seed 42", fontweight="bold")
    save_figure(fig, "validation_comparison.png")


def epoch_table(run, full=False):
    chosen = {1, 25, 50, 75, 100, 125, 150, 175, 200, int(run["best"]["epoch"])}
    rows = []
    for r in run["epochs"]:
        if not full and int(r["epoch"]) not in chosen:
            continue
        label = f"**{r['epoch']}**" if r["epoch"] == run["best"]["epoch"] else r["epoch"]
        rows.append([label, f"{float(r['learning_rate']):.7f}", f"{float(r['train_loss']):.6f}",
                     f"{float(r['train_top1']):.2f}", f"{float(r['val_loss']):.6f}", f"{float(r['val_top1']):.2f}"])
    return markdown_table(["Epoch", "LR", "Train loss", "Train Top-1 (%)", "Val loss", "Val Top-1 (%)"], rows)


def historical_comparison_figures(history):
    """Keep every published copy of the Basic/L plots in sync with the CSV."""
    models = ("basic", "l")
    assert {(r["model"], r["variant"]) for r in history} == {
        (model, variant) for model in models for variant in ("Mid32", "Mid224")
    }
    values = {(r["model"], r["variant"]): r for r in history}
    comparison = RESULTS / "model_comparison"
    images = []
    for metric, filename, ylabel, scale in [
        ("test_top1_percent", "accuracy_by_dataset.png", "Test Top-1 (%)", 1),
        ("macro_f1", "macro_f1_by_dataset.png", "Test Macro-F1 (%)", 100),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        x = np.arange(2)
        for offset, model, color in [(-0.18, "basic", "#64748b"), (0.18, "l", "#2563eb")]:
            scores = [float(values[model, variant][metric]) * scale for variant in ("Mid32", "Mid224")]
            bars = ax.bar(x + offset, scores, width=0.34, label=model.title(), color=color)
            ax.bar_label(bars, labels=[f"{score:.2f}" for score in scores], padding=3)
        ax.set(xticks=x, xticklabels=("Mid32", "Mid224"), ylim=(0, 105), ylabel=ylabel,
               title="Historical Basic/L comparison | seed 42")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        destination = comparison / filename
        fig.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        images.append(destination)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for model, color in [("basic", "#64748b"), ("l", "#2563eb")]:
        row = values[model, "Mid224"]
        x, y = int(row["flops"]) / 1e9, float(row["test_top1_percent"])
        ax.scatter(x, y, s=100, color=color)
        ax.annotate(model.title(), (x, y), xytext=(8, 8), textcoords="offset points")
    ax.set(xlim=(0.72, 1.05), ylim=(89, 100), xlabel="Conv/Linear GFLOPs per image",
           ylabel="Test Top-1 (%)", title="Historical Mid224 accuracy and compute | seed 42")
    ax.grid(alpha=0.2)
    destination = comparison / "accuracy_vs_compute_mid224.png"
    fig.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    images.append(destination)
    for name in ("Model_Basic", "Model_L"):
        for source in list(images[:3]):
            target = RESULTS / name / "comparisons" / source.name
            shutil.copy2(source, target)
            images.append(target)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Run fresh inference on the two pinned checkpoints before rebuilding figures")
    args = parser.parse_args()
    SOURCES.add(Path(__file__).resolve())
    if args.evaluate:
        evaluate_pinned_checkpoints()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    runs = [load_run(*spec) for spec in RUNS]
    for run in runs:
        learning_curves(run)
        confusion_matrix(run)
        error_examples(run)
    dataset_examples()
    validations = csv_rows(DOSSIER / "selection/validation_summary.csv")
    tests = csv_rows(DOSSIER / "comparisons/test_metrics.csv")
    by_val = {(r["variant"], r["recipe"]): r for r in validations}
    by_test = {(r["variant"], r["recipe"]): r for r in tests}
    recipe_names = [("Baseline L", "baseline"), ("Smoothing 0.05", "smoothing005"), ("Smoothing 0.10", "smoothing010"), ("LR 0.05", "lr005"), ("LR 0.15", "lr015")]
    ablation = [[label, f"{float(by_val['Mid32', recipe]['validation_top1']):.2f}", f"{float(by_val['Mid224', recipe]['validation_top1']):.2f}",
                 f"{float(by_test['Mid32', recipe]['test_top1']):.2f}", f"{float(by_test['Mid224', recipe]['test_top1']):.2f}"] for label, recipe in recipe_names]
    stage4 = csv_rows(RESULTS / "experiment_midterm_stage4_depth/comparisons/validation_summary.csv")
    stage4_test = csv_rows(RESULTS / "experiment_midterm_stage4_depth/comparisons/metrics_summary.csv")
    s_val = {r["variant"]: r for r in stage4 if r["kind"] == "Stage4-depth3"}
    s_test = {r["variant"]: r for r in stage4_test if r["kind"] == "Stage4-depth3"}
    ablation.append(["Stage 4: 3 block", *(f"{float(s_val[v]['validation_top1']):.2f}" for v in ["Mid32", "Mid224"]), *(f"{float(s_test[v]['test_top1']):.2f}" for v in ["Mid32", "Mid224"])])
    comparison_figures(runs, ablation)
    text = REPORT.read_text(encoding="utf-8")
    summary = []
    for run in runs:
        m, cfg, best = run["metrics"], run["config"], run["best"]
        summary.append([run["variant"], best["epoch"], f"{float(best['val_top1']):.2f}", f"{m['top1']:.2f}",
                        f"{np.trace(run['cm'])}/250", f"{m['macro_f1']:.6f}", f"{m['loss']:.6f}", f"{cfg['complexity']['flops']/1e9:.6f}"])
        rows = [[name, f"{run['precision'][i]*100:.2f}", f"{run['recall'][i]*100:.2f}", f"{run['f1'][i]:.6f}", "50", str(run['cm'][i,i])] for i, name in enumerate(CLASSES)]
        text = replace_block(text, f"{run['variant'].upper()}_CLASSES", markdown_table(["Lớp", "Precision (%)", "Recall (%)", "F1", "Support", "Đúng"], rows))
        text = replace_block(text, f"{run['variant'].upper()}_MILESTONES", epoch_table(run))
        text = replace_block(text, f"{run['variant'].upper()}_FULL_EPOCHS", epoch_table(run, full=True))
    text = replace_block(text, "SELECTED_RESULTS", markdown_table(["Dataset", "Epoch chọn", "Val Top-1 (%)", "Test Top-1 (%)", "Đúng", "Macro-F1", "Test loss chạy lại", "GFLOPs"], summary))
    text = replace_block(text, "ABLATION", markdown_table(["Cấu hình", "Val Mid32 (%)", "Val Mid224 (%)", "Test Mid32 (%)", "Test Mid224 (%)"], ablation))
    history = csv_rows(RESULTS / "model_comparison/reported_metrics.csv")
    history_figures = historical_comparison_figures(history)
    text = replace_block(text, "HISTORICAL_RESULTS", markdown_table(["Mô hình", "Dataset", "Batch", "Test Top-1 (%)", "Macro-F1", "Tham số", "GFLOPs"],
        [[r["model"], r["variant"], r["batch_size"], f"{float(r['test_top1_percent']):.2f}", f"{float(r['macro_f1']):.6f}", f"{int(r['parameters']):,}", f"{int(r['flops'])/1e9:.6f}"] for r in history]))
    text = replace_block(text, "CHECKPOINT_HASHES", markdown_table(["Checkpoint", "SHA-256"], [[f"`{run['checkpoint'].relative_to(ROOT).as_posix()}`", f"`{sha256(run['checkpoint'])}`"] for run in runs]))
    text = replace_block(text, "CHECKPOINT_PARITY", markdown_table(["Dataset", "Tensor trọng số ghim khớp bản hồ sơ"], [[r["variant"], "Có" if r["metadata"]["pinned_and_archived_state_dict_equal"] else "Không"] for r in runs]))
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    with (TABLES / "selected_checkpoint_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "selected_epoch", "val_top1", "test_top1", "correct", "macro_f1", "fresh_test_loss", "gflops"])
        writer.writerows(summary)
    outputs = sorted([*FIGURES.glob("*.png"), *TABLES.glob("*.csv"), *history_figures])
    provenance = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": REPORT.relative_to(ROOT).as_posix(),
        "training_performed": False,
        "inference_performed_by_builder": args.evaluate,
        "validation": "200 epochs per selected run; predictions, confusion matrices, class metrics and saved pinned-entrypoint inference logs agree",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in sorted(SOURCES)],
        "outputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in outputs],
        "report_sha256": sha256(REPORT),
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "figures": len(list(FIGURES.glob('*.png'))), "tables": len(list(TABLES.glob('*.csv'))), "selected_runs": [r['name'] for r in runs], "validated": True}, indent=2))


if __name__ == "__main__":
    main()
