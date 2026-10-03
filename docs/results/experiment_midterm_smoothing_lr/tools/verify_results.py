#!/usr/bin/env python3
"""Independently verify the final smoothing/LR experiment dossier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote


CLASSES = ("bird", "cat", "dog", "frog", "horse")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
RECIPES = ("baseline", "smoothing005", "smoothing010", "lr005", "lr015")
EXPECTED_RUNS = tuple(
    f"l_{variant}_{recipe}"
    for variant in ("mid32", "mid224")
    for recipe in RECIPES
)
REQUIRED_RUN_FILES = (
    "best_val.pt",
    "last.pt",
    "config.json",
    "epochs.csv",
    "validation_split.json",
)
MODEL_FILES = {
    "README.md",
    "test_metrics.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "classification_report.csv",
    "confusion_matrix.png",
    "learning_curves.png",
}
SELECTION_FILES = {
    "selection_lock.json",
    "validation_summary.csv",
    "recipe_vs_baseline.csv",
    "validation_top1_by_recipe.png",
    "validation_loss_by_recipe.png",
    "best_epoch_by_recipe.png",
}
COMPARISON_FILES = {
    "validation_metrics.csv",
    "test_metrics.csv",
    "class_metrics.csv",
    "test_top1_by_recipe.png",
    "macro_f1_by_recipe.png",
    "learning_curves_mid32.png",
    "learning_curves_mid224.png",
    "confusion_matrices_mid32.png",
    "confusion_matrices_mid224.png",
}
EXPECTED_MANIFEST_SHA256 = {
    "Mid32": "5e012ceef22d363877407ce7a0ecc2d8b42dcc7cb891f981124eae2523b0ad8e",
    "Mid224": "028a2eda6f9c9c385875925e3cadbd83979ad3718df7652e3864a01df01a1929",
}
EXPECTED_WINNERS = {"Mid32": "l_mid32_lr015", "Mid224": "l_mid224_baseline"}
EXPECTED_SOURCE_FILES = {
    "train_mid_experiment.py",
    "models/SE_Attention.py",
    "models/TickNet.py",
    "models/common.py",
    "models/datasets.py",
    "models/mid_data.py",
    "models/mid_experiment_data.py",
    "models/mid_models.py",
    "models/model_profile.py",
    "models/ticknet_c.py",
    "models/ticknet_l.py",
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(
    checks: dict[str, dict[str, Any]], name: str, ok: bool, details: list[str]
) -> None:
    checks[name] = {"ok": bool(ok), "details": details}


def _verify_expected_tree(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    if not (result_root / "README.md").is_file():
        details.append("missing root README.md")
    for run_name in EXPECTED_RUNS:
        copied = result_root / "checkpoints" / run_name
        missing_run = [name for name in REQUIRED_RUN_FILES if not (copied / name).is_file()]
        if missing_run:
            details.append(f"{run_name} copied bundle missing {missing_run}")
        model = result_root / "models" / run_name
        observed_model = {path.name for path in model.iterdir() if path.is_file()} if model.is_dir() else set()
        if observed_model != MODEL_FILES:
            details.append(
                f"{run_name} model files differ: missing={sorted(MODEL_FILES-observed_model)}, "
                f"extra={sorted(observed_model-MODEL_FILES)}"
            )
    for directory, expected in (
        (result_root / "selection", SELECTION_FILES),
        (result_root / "comparisons", COMPARISON_FILES),
    ):
        observed = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
        if observed != expected:
            details.append(
                f"{directory.name} files differ: missing={sorted(expected-observed)}, "
                f"extra={sorted(observed-expected)}"
            )
    required_misc = (
        result_root / "checkpoints" / "checkpoint_manifest.csv",
        result_root / "provenance" / "run_manifest.csv",
        result_root / "provenance" / "source_checks.json",
        result_root / "provenance" / "evaluation_environment.json",
        result_root / "provenance" / "artifact_manifest.csv",
        result_root / "provenance" / "source_split_manifest_mid32.csv",
        result_root / "provenance" / "source_split_manifest_mid224.csv",
        result_root / "tools" / "build_experiment_results.py",
        result_root / "tools" / "verify_results.py",
    )
    for path in required_misc:
        if not path.is_file():
            details.append(f"missing {path.relative_to(result_root).as_posix()}")
    return not details, details


def _verify_checkpoints(source_runs: Path, result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    observed_rows: dict[tuple[str, str], dict[str, str]] = {}
    manifest_path = result_root / "checkpoints" / "checkpoint_manifest.csv"
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            observed_rows = {(row["run_name"], row["file"]): row for row in rows}
            if len(rows) != 50 or len(observed_rows) != 50:
                details.append(f"checkpoint manifest must contain 50 unique rows, found {len(rows)}")
        except (OSError, UnicodeError, csv.Error, KeyError) as error:
            details.append(f"cannot parse checkpoint manifest: {error}")
    else:
        details.append("checkpoint manifest is missing")
    for run_name in EXPECTED_RUNS:
        for filename in REQUIRED_RUN_FILES:
            source = source_runs / run_name / filename
            copied = result_root / "checkpoints" / run_name / filename
            if not source.is_file() or not copied.is_file():
                continue
            source_digest = file_sha256(source)
            copied_digest = file_sha256(copied)
            if source.stat().st_size != copied.stat().st_size or source_digest != copied_digest:
                details.append(f"{run_name}/{filename} differs from source")
            row = observed_rows.get((run_name, filename))
            if row is not None:
                if (
                    row.get("source_sha256") != source_digest
                    or row.get("copied_sha256") != copied_digest
                    or int(row.get("size_bytes", -1)) != source.stat().st_size
                ):
                    details.append(f"{run_name}/{filename} disagrees with checkpoint manifest")
    return not details, details


def _verify_selection(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    lock_path = result_root / "selection" / "selection_lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, [f"cannot parse selection lock: {error}"]
    if lock.get("selection_source") != "validation_only":
        details.append("selection source is not validation_only")
    if lock.get("winners") != EXPECTED_WINNERS:
        details.append(f"selection winners differ: {lock.get('winners')}")
    candidates = lock.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 10:
        details.append("selection lock must contain ten validation candidates")
    elif any(any("test" in key.lower() for key in candidate) for candidate in candidates):
        details.append("selection candidates contain test-derived fields")
    digest = file_sha256(lock_path)
    try:
        readme = (result_root / "README.md").read_text(encoding="utf-8")
        environment = json.loads(
            (result_root / "provenance" / "evaluation_environment.json").read_text(encoding="utf-8")
        )
        if digest not in readme:
            details.append("root report does not record the current selection-lock digest")
        if environment.get("selection_lock_sha256") != digest:
            details.append("evaluation environment selection-lock digest differs")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        details.append(f"cannot verify selection-lock references: {error}")
    return not details, details


def _verify_model_metrics(result_root: Path) -> tuple[dict[str, bool], dict[str, list[str]]]:
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    schema_details: list[str] = []
    metric_details: list[str] = []
    expected_columns = [
        "path",
        "target",
        "target_name",
        "prediction",
        "prediction_name",
        "correct",
    ]
    comparison_rows: dict[str, dict[str, str]] = {}
    comparison_path = result_root / "comparisons" / "test_metrics.csv"
    if comparison_path.is_file():
        try:
            with comparison_path.open(encoding="utf-8", newline="") as handle:
                comparison_rows = {row["run_name"]: row for row in csv.DictReader(handle)}
        except (OSError, UnicodeError, csv.Error, KeyError) as error:
            metric_details.append(f"cannot parse comparison test metrics: {error}")
    for run_name in EXPECTED_RUNS:
        model = result_root / "models" / run_name
        prediction_path = model / "predictions.csv"
        metrics_path = model / "test_metrics.json"
        confusion_path = model / "confusion_matrix.csv"
        report_path = model / "classification_report.csv"
        if not all(path.is_file() for path in (prediction_path, metrics_path, confusion_path, report_path)):
            continue
        try:
            predictions = pd.read_csv(prediction_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            confusion = pd.read_csv(confusion_path)
            class_report = pd.read_csv(report_path)
        except Exception as error:
            schema_details.append(f"{run_name} evidence cannot be parsed: {error}")
            continue
        if list(predictions.columns) != expected_columns:
            schema_details.append(f"{run_name} prediction schema differs")
            continue
        if len(predictions) != 250:
            schema_details.append(f"{run_name} prediction count is {len(predictions)}, not 250")
        inverse = {index: name for name, index in CLASS_TO_IDX.items()}
        targets = predictions["target"].astype(int).tolist()
        predicted = predictions["prediction"].astype(int).tolist()
        if not set(targets + predicted).issubset(inverse):
            schema_details.append(f"{run_name} contains unknown class indices")
            continue
        expected_target_names = [inverse[value] for value in targets]
        expected_prediction_names = [inverse[value] for value in predicted]
        expected_correct = [target == prediction for target, prediction in zip(targets, predicted)]
        observed_correct = predictions["correct"].map(
            lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
        ).tolist()
        if predictions["target_name"].tolist() != expected_target_names:
            schema_details.append(f"{run_name} target names disagree with indices")
        if predictions["prediction_name"].tolist() != expected_prediction_names:
            schema_details.append(f"{run_name} prediction names disagree with indices")
        if observed_correct != expected_correct:
            metric_details.append(f"{run_name} correctness flags disagree with predictions")

        matrix = confusion_matrix(targets, predicted, labels=list(range(5)))
        expected_confusion_columns = ["actual_class", *(f"predicted_{name}" for name in CLASSES)]
        if list(confusion.columns) != expected_confusion_columns:
            schema_details.append(f"{run_name} confusion-matrix schema differs")
        else:
            observed_matrix = confusion.drop(columns="actual_class").to_numpy(dtype=int)
            if confusion["actual_class"].tolist() != list(CLASSES) or not (observed_matrix == matrix).all():
                metric_details.append(f"{run_name} confusion matrix disagrees with predictions")
        top1 = 100.0 * sum(expected_correct) / len(expected_correct)
        macro_f1 = f1_score(targets, predicted, labels=list(range(5)), average="macro", zero_division=0)
        required_metric_keys = {"loss", "top1", "samples", "macro_f1"}
        if not required_metric_keys.issubset(metrics) or not math.isfinite(float(metrics.get("loss", math.nan))):
            metric_details.append(f"{run_name} test metrics are incomplete or loss is not finite")
        elif (
            int(metrics["samples"]) != 250
            or abs(float(metrics["top1"]) - top1) > 1e-12
            or abs(float(metrics["macro_f1"]) - macro_f1) > 1e-12
        ):
            metric_details.append(f"{run_name} test metrics disagree with predictions")
        expected_report = classification_report(
            targets,
            predicted,
            labels=list(range(5)),
            target_names=list(CLASSES),
            output_dict=True,
            zero_division=0,
        )
        report_by_name = {row["class_name"]: row for row in class_report.to_dict(orient="records")}
        for name in (*CLASSES, "macro avg", "weighted avg"):
            observed = report_by_name.get(name)
            expected = expected_report[name]
            if observed is None or any(
                abs(float(observed[field]) - float(expected[field])) > 1e-12
                for field in ("precision", "recall", "f1-score", "support")
            ):
                metric_details.append(f"{run_name} classification report differs at {name}")
                break
        comparison = comparison_rows.get(run_name)
        if comparison is not None and required_metric_keys.issubset(metrics):
            if any(
                abs(float(comparison[field]) - float(metrics[key])) > 1e-12
                for field, key in (
                    ("test_loss", "loss"),
                    ("test_top1", "top1"),
                    ("test_macro_f1", "macro_f1"),
                )
            ):
                metric_details.append(f"{run_name} comparison metrics differ from model metrics")
    return (
        {"prediction_schema": not schema_details, "recomputed_metrics": not metric_details},
        {"prediction_schema": schema_details, "recomputed_metrics": metric_details},
    )


def _verify_artifact_manifest(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    manifest = result_root / "provenance" / "artifact_manifest.csv"
    if not manifest.is_file():
        return False, ["artifact manifest is missing"]
    try:
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        recorded = {row["relative_path"]: row for row in rows}
    except (OSError, UnicodeError, csv.Error, KeyError) as error:
        return False, [f"cannot parse artifact manifest: {error}"]
    excluded = {
        "provenance/artifact_manifest.csv",
        "provenance/verification.json",
    }
    actual = {
        path.relative_to(result_root).as_posix()
        for path in result_root.rglob("*")
        if (
            path.is_file()
            and path.relative_to(result_root).as_posix() not in excluded
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    }
    if set(recorded) != actual:
        details.append(
            f"artifact inventory differs: missing={sorted(actual-set(recorded))}, "
            f"stale={sorted(set(recorded)-actual)}"
        )
    for relative, row in recorded.items():
        path = result_root / relative
        if not path.is_file():
            continue
        if int(row.get("size_bytes", -1)) != path.stat().st_size or row.get("sha256") != file_sha256(path):
            details.append(f"artifact hash/size differs: {relative}")
    return not details, details


def _verify_images(result_root: Path) -> tuple[bool, list[str]]:
    from PIL import Image

    details: list[str] = []
    pngs = list(result_root.rglob("*.png"))
    expected = {
        *(f"selection/{name}" for name in (
            "validation_top1_by_recipe.png",
            "validation_loss_by_recipe.png",
            "best_epoch_by_recipe.png",
        )),
        *(f"models/{run_name}/{name}" for run_name in EXPECTED_RUNS for name in (
            "confusion_matrix.png",
            "learning_curves.png",
        )),
        *(f"comparisons/{name}" for name in (
            "test_top1_by_recipe.png",
            "macro_f1_by_recipe.png",
            "learning_curves_mid32.png",
            "learning_curves_mid224.png",
            "confusion_matrices_mid32.png",
            "confusion_matrices_mid224.png",
        )),
    }
    observed = {path.relative_to(result_root).as_posix() for path in pngs}
    if observed != expected:
        details.append(
            f"PNG inventory differs: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    for path in pngs:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            details.append(f"unreadable image {path.relative_to(result_root)}: {error}")
    return not details, details


def _verify_markdown_links(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    for markdown in result_root.rglob("*.md"):
        try:
            text = markdown.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            details.append(f"cannot read {markdown.relative_to(result_root)}: {error}")
            continue
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = unquote(target.split("#", 1)[0])
            resolved = (markdown.parent / relative_target).resolve()
            try:
                resolved.relative_to(result_root.resolve())
            except ValueError:
                details.append(f"{markdown.relative_to(result_root)} links outside result tree: {target}")
                continue
            if not resolved.exists():
                details.append(f"{markdown.relative_to(result_root)} has broken link: {target}")
    return not details, details


def _verify_parseable_files(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    for path in result_root.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            details.append(f"invalid JSON {path.relative_to(result_root)}: {error}")
    for path in result_root.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                first = next(csv.reader(handle), None)
            if first is None:
                details.append(f"empty CSV {path.relative_to(result_root)}")
        except (OSError, UnicodeError, csv.Error) as error:
            details.append(f"invalid CSV {path.relative_to(result_root)}: {error}")
    return not details, details


def _verify_source_manifests(result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    for variant, expected in EXPECTED_MANIFEST_SHA256.items():
        path = result_root / "provenance" / f"source_split_manifest_{variant.lower()}.csv"
        if not path.is_file():
            details.append(f"missing {variant} source manifest")
        elif file_sha256(path) != expected:
            details.append(f"{variant} source manifest digest differs")
    return not details, details


def _verify_training_sources(source_runs: Path, result_root: Path) -> tuple[bool, list[str]]:
    details: list[str] = []
    repo_root = next(
        (candidate for candidate in (result_root, *result_root.parents) if (candidate / "train_mid_experiment.py").is_file()),
        None,
    )
    if repo_root is None:
        return False, ["cannot locate repository root containing train_mid_experiment.py"]
    checked = 0
    for run_name in EXPECTED_RUNS:
        config_path = source_runs / run_name / "config.json"
        if not config_path.is_file():
            details.append(f"{run_name} source config is missing")
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            recorded = config["source_hashes"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            details.append(f"{run_name} source hashes cannot be parsed: {error}")
            continue
        if not isinstance(recorded, dict) or set(recorded) != EXPECTED_SOURCE_FILES:
            details.append(f"{run_name} source hash set differs from the expected 11 files")
        for relative, expected_digest in recorded.items():
            source = repo_root / relative
            if not source.is_file():
                details.append(f"{run_name} source hash target is missing: {relative}")
            elif file_sha256(source) != expected_digest:
                details.append(f"{run_name} source hash mismatch: {relative}")
            checked += 1
    source_checks_path = result_root / "provenance" / "source_checks.json"
    if source_checks_path.is_file():
        try:
            source_checks = json.loads(source_checks_path.read_text(encoding="utf-8"))
            if source_checks.get("checked") != 110 or source_checks.get("matched") != 110:
                details.append("source_checks.json does not record 110/110 matches")
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            details.append(f"cannot parse source_checks.json: {error}")
    else:
        details.append("source_checks.json is missing")
    if checked != 110:
        details.append(f"recomputed source hash count is {checked}, not 110")
    return not details, details


def verify_result_tree(source_runs: Path, result_root: Path) -> dict[str, object]:
    source_runs = Path(source_runs).resolve()
    result_root = Path(result_root).resolve()
    checks: dict[str, dict[str, Any]] = {}
    for name, verifier in (
        ("expected_tree", lambda: _verify_expected_tree(result_root)),
        ("checkpoint_sha256", lambda: _verify_checkpoints(source_runs, result_root)),
        ("selection_lock", lambda: _verify_selection(result_root)),
        ("source_manifests", lambda: _verify_source_manifests(result_root)),
        ("source_hashes", lambda: _verify_training_sources(source_runs, result_root)),
        ("artifact_manifest", lambda: _verify_artifact_manifest(result_root)),
        ("images", lambda: _verify_images(result_root)),
        ("markdown_links", lambda: _verify_markdown_links(result_root)),
        ("parseable_files", lambda: _verify_parseable_files(result_root)),
    ):
        try:
            ok, details = verifier()
        except Exception as error:
            ok, details = False, [f"verifier raised {type(error).__name__}: {error}"]
        _record(checks, name, ok, details)
    try:
        metric_status, metric_details = _verify_model_metrics(result_root)
    except Exception as error:
        message = [f"verifier raised {type(error).__name__}: {error}"]
        metric_status = {"prediction_schema": False, "recomputed_metrics": False}
        metric_details = {"prediction_schema": message, "recomputed_metrics": message}
    for name in ("prediction_schema", "recomputed_metrics"):
        _record(checks, name, metric_status[name], metric_details[name])
    failed = [name for name, value in checks.items() if not value["ok"]]
    return {
        "ok": not failed,
        "source_runs": str(source_runs),
        "result_root": str(result_root),
        "checks": checks,
        "failed_checks": failed,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = verify_result_tree(args.source_runs, args.results)
    result["verified_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output = args.results / "provenance" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
