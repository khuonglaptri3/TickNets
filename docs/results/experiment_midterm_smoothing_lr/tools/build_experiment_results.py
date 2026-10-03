#!/usr/bin/env python3
"""Build the audited TickNet-L smoothing/learning-rate result dossier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


CLASSES = ("bird", "cat", "dog", "frog", "horse")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}
REQUIRED_RUN_FILES = (
    "best_val.pt",
    "last.pt",
    "config.json",
    "epochs.csv",
    "validation_split.json",
)
EXPECTED_MEMBERSHIP_SHA256 = "2dd7a3372b207ec9501d13e35edee79a6f3d2bb5f0dd10b5d3443c9326073358"
EXPECTED_VALIDATION_SHA256 = "55d56a207a40c4ad97e24cf2fd55f8620fe674f6727e47b63cb65c1290ddd062"
EXPECTED_MANIFEST_SHA256 = {
    "Mid32": "5e012ceef22d363877407ce7a0ecc2d8b42dcc7cb891f981124eae2523b0ad8e",
    "Mid224": "028a2eda6f9c9c385875925e3cadbd83979ad3718df7652e3864a01df01a1929",
}
EXPECTED_IMAGE_SIZE = {"Mid32": (32, 32), "Mid224": (224, 224)}
EXPECTED_SPLIT_COUNT = {"train": 5000, "test": 50}
EPOCH_FIELDS = (
    "epoch",
    "learning_rate",
    "train_loss",
    "train_top1",
    "train_samples",
    "val_loss",
    "val_top1",
    "val_samples",
)
REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = Path(__file__).resolve().parents[1]


class EvidenceError(RuntimeError):
    """Raised when source evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class Recipe:
    name: str
    learning_rate: float
    label_smoothing: float


RECIPES = (
    Recipe("baseline", 0.1, 0.0),
    Recipe("smoothing005", 0.1, 0.05),
    Recipe("smoothing010", 0.1, 0.10),
    Recipe("lr005", 0.05, 0.0),
    Recipe("lr015", 0.15, 0.0),
)
RUN_MATRIX = tuple(
    (f"l_{variant.lower()}_{recipe.name}", variant, recipe)
    for variant in ("Mid32", "Mid224")
    for recipe in RECIPES
)
RUN_EXPECTATIONS = {name: (variant, recipe) for name, variant, recipe in RUN_MATRIX}


@dataclass(frozen=True)
class RunRecord:
    name: str
    variant: str
    recipe: str
    path: Path
    config: dict[str, Any]
    epochs: list[dict[str, str]]
    best_epoch: int
    best_val_top1: float
    best_val_loss: float


def expected_run_names() -> tuple[str, ...]:
    return tuple(name for name, _, _ in RUN_MATRIX)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON evidence must be an object: {path}")
    return payload


def select_best_epoch(rows: list[dict[str, str]]) -> tuple[int, float, float]:
    if not rows:
        raise EvidenceError("epochs evidence is empty")
    try:
        candidates = [
            (int(row["epoch"]), float(row["val_top1"]), float(row["val_loss"]))
            for row in rows
            if row.get("val_top1") and row.get("val_loss")
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"invalid validation values in epochs evidence: {error}") from error
    if not candidates:
        raise EvidenceError("epochs evidence has no validation rows")
    epoch, top1, loss = max(candidates, key=lambda item: (item[1], -item[2]))
    return epoch, top1, loss


def _validate_config(
    run_name: str, config: dict[str, Any], repo_root: Path
) -> tuple[str, Recipe, int]:
    variant, recipe = RUN_EXPECTATIONS[run_name]
    expected = {
        "variant": variant,
        "model": "l",
        "seed": 42,
        "split_seed": 123,
        "val_fraction": 0.1,
        "epochs": 200,
        "learning_rate": recipe.learning_rate,
        "label_smoothing": recipe.label_smoothing,
        "trainer_revision": "mid-experiment-v2",
        "architecture_revision": "ticknet-l-v1",
        "train_samples": 22500,
        "val_samples": 2500,
        "test_samples": 250,
        "class_to_idx": CLASS_TO_IDX,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "validation_sha256": EXPECTED_VALIDATION_SHA256,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256[variant],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise EvidenceError(
                f"{run_name} config {key!r} mismatch: {config.get(key)!r} != {value!r}"
            )

    source_hashes = config.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise EvidenceError(f"{run_name} has no source hashes")
    expected_source_paths = {
        "train_mid_experiment.py",
        *(path.relative_to(repo_root).as_posix() for path in sorted((repo_root / "models").glob("*.py"))),
    }
    if set(source_hashes) != expected_source_paths:
        raise EvidenceError(
            f"{run_name} source hash set mismatch: "
            f"missing={sorted(expected_source_paths-set(source_hashes))}, "
            f"extra={sorted(set(source_hashes)-expected_source_paths)}"
        )
    checked = 0
    for relative, recorded_digest in source_hashes.items():
        source = repo_root / relative
        if not source.is_file():
            raise EvidenceError(f"{run_name} source hash target is missing: {relative}")
        current_digest = file_sha256(source)
        if current_digest != recorded_digest:
            raise EvidenceError(
                f"{run_name} source hash mismatch for {relative}: "
                f"{current_digest} != {recorded_digest}"
            )
        checked += 1
    return variant, recipe, checked


def _validate_split(run_name: str, path: Path, config: dict[str, Any]) -> None:
    split = _read_json(path)
    keys = {
        "variant": config["variant"],
        "class_to_idx": CLASS_TO_IDX,
        "split_seed": 123,
        "val_fraction": 0.1,
        "train_samples": 22500,
        "val_samples": 2500,
        "test_samples": 250,
        "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "validation_sha256": EXPECTED_VALIDATION_SHA256,
        "full_train": False,
    }
    for key, expected in keys.items():
        if split.get(key) != expected:
            raise EvidenceError(f"{run_name} validation split {key!r} mismatch")
    train_keys = split.get("train_keys")
    validation_keys = split.get("validation_keys")
    test_keys = split.get("test_keys")
    if not all(isinstance(values, list) for values in (train_keys, validation_keys, test_keys)):
        raise EvidenceError(f"{run_name} validation split key lists are invalid")
    if (len(train_keys), len(validation_keys), len(test_keys)) != (22500, 2500, 250):
        raise EvidenceError(f"{run_name} validation split sample counts are invalid")
    if set(train_keys) & set(validation_keys):
        raise EvidenceError(f"{run_name} train and validation memberships overlap")
    official_train = sorted([*train_keys, *validation_keys])
    membership = _compact_json_sha256(
        {
            "class_to_idx": CLASS_TO_IDX,
            "train_keys": official_train,
            "test_keys": sorted(test_keys),
        }
    )
    validation = _compact_json_sha256(sorted(validation_keys))
    if membership != EXPECTED_MEMBERSHIP_SHA256:
        raise EvidenceError(f"{run_name} recomputed membership hash mismatch")
    if validation != EXPECTED_VALIDATION_SHA256:
        raise EvidenceError(f"{run_name} recomputed validation hash mismatch")


def _read_epochs(run_name: str, path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EPOCH_FIELDS:
                raise EvidenceError(f"{run_name} epochs.csv fields are invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise EvidenceError(f"cannot read {run_name} epochs.csv: {error}") from error
    if len(rows) != 200:
        raise EvidenceError(f"{run_name} epochs.csv must contain exactly 200 rows")
    try:
        observed_epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"{run_name} has invalid epoch numbers") from error
    if observed_epochs != list(range(1, 201)):
        raise EvidenceError(f"{run_name} epochs must be exactly 1..200")
    return rows


def _validate_checkpoints(record: RunRecord) -> None:
    try:
        import torch

        best = torch.load(record.path / "best_val.pt", map_location="cpu", weights_only=True)
        last = torch.load(record.path / "last.pt", map_location="cpu", weights_only=True)
    except Exception as error:  # torch wraps malformed archives in several exception types.
        raise EvidenceError(f"{record.name} checkpoint cannot be loaded safely: {error}") from error
    if best.get("role") != "best_val" or int(best.get("epoch", -1)) != record.best_epoch:
        raise EvidenceError(f"{record.name} best_val.pt does not match recomputed best epoch")
    if last.get("role") != "last" or int(last.get("epoch", -1)) != 200:
        raise EvidenceError(f"{record.name} last.pt is not the completed epoch-200 state")
    last_best = last.get("best_validation") or {}
    if int(last_best.get("epoch", -1)) != record.best_epoch:
        raise EvidenceError(f"{record.name} last.pt best validation metadata is inconsistent")
    for checkpoint_name, checkpoint in (("best_val.pt", best), ("last.pt", last)):
        checkpoint_config = checkpoint.get("config") or {}
        for key in (
            "variant",
            "seed",
            "split_seed",
            "architecture_revision",
            "membership_sha256",
            "validation_sha256",
            "source_manifest_sha256",
        ):
            if checkpoint_config.get(key) != record.config.get(key):
                raise EvidenceError(f"{record.name} {checkpoint_name} config {key!r} mismatch")


def audit_runs(runs_root: Path, repo_root: Path) -> list[RunRecord]:
    runs_root = Path(runs_root)
    repo_root = Path(repo_root)
    expected = set(expected_run_names())
    present = {
        path.name
        for path in runs_root.glob("l_mid*")
        if path.is_dir() and (path.name.startswith("l_mid32_") or path.name.startswith("l_mid224_"))
    }
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing:
        raise EvidenceError(f"missing expected runs: {missing}")
    if unexpected:
        raise EvidenceError(f"unexpected experiment runs: {unexpected}")

    records: list[RunRecord] = []
    source_checks = 0
    for run_name in expected_run_names():
        run_path = runs_root / run_name
        missing_files = [name for name in REQUIRED_RUN_FILES if not (run_path / name).is_file()]
        if missing_files:
            raise EvidenceError(f"{run_name} is incomplete; missing {missing_files}")
        config = _read_json(run_path / "config.json")
        variant, recipe, checked = _validate_config(run_name, config, repo_root)
        source_checks += checked
        _validate_split(run_name, run_path / "validation_split.json", config)
        epochs = _read_epochs(run_name, run_path / "epochs.csv")
        best_epoch, best_top1, best_loss = select_best_epoch(epochs)
        records.append(
            RunRecord(
                name=run_name,
                variant=variant,
                recipe=recipe.name,
                path=run_path.resolve(),
                config=config,
                epochs=epochs,
                best_epoch=best_epoch,
                best_val_top1=best_top1,
                best_val_loss=best_loss,
            )
        )

    for record in records:
        _validate_checkpoints(record)
    expected_source_checks = sum(len(record.config["source_hashes"]) for record in records)
    if source_checks != expected_source_checks:
        raise EvidenceError("source hash check count is inconsistent")
    return records


def write_source_manifest(data_root: Path, variant: str, output_path: Path) -> str:
    from PIL import Image

    if variant not in EXPECTED_IMAGE_SIZE:
        raise EvidenceError(f"unsupported variant: {variant}")
    data_root = Path(data_root)
    variant_root = data_root / variant
    rows: list[dict[str, object]] = []
    memberships: dict[str, list[str]] = {}
    for split in ("train", "test"):
        memberships[split] = []
        for class_name in CLASSES:
            class_dir = variant_root / split / class_name
            if not class_dir.is_dir():
                raise EvidenceError(f"missing image directory: {class_dir}")
            images = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() in EXTENSIONS
            )
            for image_path in images:
                try:
                    with Image.open(image_path) as image:
                        actual_size = image.size
                        image.verify()
                except Exception as error:
                    raise EvidenceError(f"invalid image {image_path}: {error}") from error
                if actual_size != EXPECTED_IMAGE_SIZE[variant]:
                    raise EvidenceError(
                        f"wrong image dimensions for {image_path}: "
                        f"{actual_size} != {EXPECTED_IMAGE_SIZE[variant]}"
                    )
                memberships[split].append(f"{class_name}/{image_path.name}")
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "class_name": class_name,
                        "filename": image_path.name,
                        "size_bytes": image_path.stat().st_size,
                        "sha256": file_sha256(image_path),
                    }
                )
    overlap = set(memberships["train"]) & set(memberships["test"])
    if overlap:
        raise EvidenceError(f"{variant} train/test memberships overlap")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = ("variant", "split", "class_name", "filename", "size_bytes", "sha256")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    return file_sha256(output_path)


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as symlink_error:
        if os.name != "nt":
            raise EvidenceError(f"cannot create dataset symlink {link}: {symlink_error}") from symlink_error
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise EvidenceError(
            f"cannot create dataset junction {link}: {(result.stderr or result.stdout).strip()}"
        )


def prepare_evaluation_data_root(
    data_root: Path, variant: str, manifest_path: Path, workspace: Path
) -> Path:
    data_root = Path(data_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise EvidenceError(f"source manifest does not exist: {manifest_path}")
    root = Path(workspace).resolve() / variant.lower()
    root.mkdir(parents=True, exist_ok=True)
    destination_manifest = root / "split_manifest.csv"
    if destination_manifest.exists():
        if destination_manifest.read_bytes() != manifest_path.read_bytes():
            raise EvidenceError(f"evaluation manifest is inconsistent: {destination_manifest}")
    else:
        shutil.copy2(manifest_path, destination_manifest)
    source = data_root / variant
    link = root / variant
    if link.exists() or link.is_symlink():
        if link.resolve() != source.resolve():
            raise EvidenceError(f"evaluation data link points to a different source: {link}")
    else:
        _create_directory_link(link, source)
    return root


def build_selection_lock(records: list[RunRecord]) -> dict[str, object]:
    if {record.name for record in records} != set(expected_run_names()):
        raise EvidenceError("selection requires exactly the ten audited runs")
    candidates = [
        {
            "run_name": record.name,
            "variant": record.variant,
            "recipe": record.recipe,
            "seed": record.config["seed"],
            "best_epoch": record.best_epoch,
            "validation_top1": record.best_val_top1,
            "validation_loss": record.best_val_loss,
            "membership_sha256": record.config["membership_sha256"],
            "validation_sha256": record.config["validation_sha256"],
            "source_manifest_sha256": record.config["source_manifest_sha256"],
        }
        for record in records
    ]
    winners: dict[str, str] = {}
    for variant in ("Mid32", "Mid224"):
        variant_records = [record for record in records if record.variant == variant]
        winner = max(
            variant_records,
            key=lambda record: (record.best_val_top1, -record.best_val_loss),
        )
        winners[variant] = winner.name
    return {
        "schema_version": 1,
        "selection_source": "validation_only",
        "selection_rule": "maximum validation Top-1; lower validation loss breaks ties",
        "test_policy": (
            "Historical test metrics are computed only after this lock and cannot change winners."
        ),
        "winners": winners,
        "candidates": candidates,
    }


def copy_run_bundles(
    records: list[RunRecord], destination: Path
) -> list[dict[str, object]]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for record in records:
        target_dir = destination / record.name
        target_dir.mkdir(parents=True, exist_ok=True)
        extras = sorted(
            path.name
            for path in target_dir.iterdir()
            if path.name not in REQUIRED_RUN_FILES
        )
        if extras:
            raise EvidenceError(
                f"refusing to overwrite inconsistent destination {target_dir}; extra files: {extras}"
            )
        for filename in REQUIRED_RUN_FILES:
            source = record.path / filename
            target = target_dir / filename
            source_digest = file_sha256(source)
            if target.exists():
                copied_digest = file_sha256(target)
                if copied_digest != source_digest or target.stat().st_size != source.stat().st_size:
                    raise EvidenceError(
                        f"refusing to overwrite inconsistent destination file: {target}"
                    )
            else:
                temporary = target.with_name(f".{target.name}.copying")
                temporary.unlink(missing_ok=True)
                shutil.copy2(source, temporary)
                copied_digest = file_sha256(temporary)
                if copied_digest != source_digest:
                    temporary.unlink(missing_ok=True)
                    raise EvidenceError(f"copied file hash mismatch: {source} -> {target}")
                temporary.replace(target)
            manifest.append(
                {
                    "run_name": record.name,
                    "variant": record.variant,
                    "recipe": record.recipe,
                    "file": filename,
                    "size_bytes": source.stat().st_size,
                    "source_sha256": source_digest,
                    "copied_sha256": file_sha256(target),
                    "source_path": f"runs/{record.name}/{filename}",
                    "copied_path": f"checkpoints/{record.name}/{filename}",
                }
            )
    return manifest


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv_atomic(
    path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _persist_selection_lock(core: dict[str, object], path: Path) -> dict[str, object]:
    if path.exists():
        existing = _read_json(path)
        existing_core = {key: value for key, value in existing.items() if key != "created_at_utc"}
        if existing_core != core or not isinstance(existing.get("created_at_utc"), str):
            raise EvidenceError(f"selection lock is inconsistent and immutable: {path}")
        return existing
    payload = {
        **core,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json_atomic(path, payload)
    return payload


def _write_validation_plots(records: list[RunRecord], selection_dir: Path) -> None:
    matplotlib_cache = REPO_ROOT / "runs" / ".report-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("best_val_top1", "Validation Top-1 (%)", "validation_top1_by_recipe.png"),
        ("best_val_loss", "Validation loss", "validation_loss_by_recipe.png"),
        ("best_epoch", "Best epoch", "best_epoch_by_recipe.png"),
    )
    labels = [recipe.name for recipe in RECIPES]
    x = list(range(len(labels)))
    width = 0.36
    for attribute, ylabel, filename in metrics:
        figure, axis = plt.subplots(figsize=(9, 5))
        for offset, variant, color in ((-width / 2, "Mid32", "#4c78a8"), (width / 2, "Mid224", "#f58518")):
            values = [
                getattr(next(record for record in records if record.variant == variant and record.recipe == recipe), attribute)
                for recipe in labels
            ]
            bars = axis.bar([position + offset for position in x], values, width, label=variant, color=color)
            axis.bar_label(bars, fmt="%.2f" if attribute != "best_epoch" else "%d", padding=2, fontsize=8)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} by recipe (validation-only)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(selection_dir / filename, dpi=160)
        plt.close(figure)


def _write_selection_tables(records: list[RunRecord], selection_dir: Path) -> None:
    summary = [
        {
            "run_name": record.name,
            "variant": record.variant,
            "recipe": record.recipe,
            "seed": record.config["seed"],
            "best_epoch": record.best_epoch,
            "validation_top1": record.best_val_top1,
            "validation_loss": record.best_val_loss,
        }
        for record in records
    ]
    _write_csv_atomic(
        selection_dir / "validation_summary.csv",
        summary,
        (
            "run_name",
            "variant",
            "recipe",
            "seed",
            "best_epoch",
            "validation_top1",
            "validation_loss",
        ),
    )
    deltas: list[dict[str, object]] = []
    for record in records:
        baseline = next(
            item for item in records if item.variant == record.variant and item.recipe == "baseline"
        )
        deltas.append(
            {
                "run_name": record.name,
                "variant": record.variant,
                "recipe": record.recipe,
                "delta_validation_top1_vs_baseline": record.best_val_top1 - baseline.best_val_top1,
                "delta_validation_loss_vs_baseline": record.best_val_loss - baseline.best_val_loss,
                "delta_best_epoch_vs_baseline": record.best_epoch - baseline.best_epoch,
            }
        )
    _write_csv_atomic(
        selection_dir / "recipe_vs_baseline.csv",
        deltas,
        (
            "run_name",
            "variant",
            "recipe",
            "delta_validation_top1_vs_baseline",
            "delta_validation_loss_vs_baseline",
            "delta_best_epoch_vs_baseline",
        ),
    )


def _write_run_manifest(records: list[RunRecord], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        row: dict[str, object] = {
            "run_name": record.name,
            "variant": record.variant,
            "recipe": record.recipe,
            "seed": record.config["seed"],
            "split_seed": record.config["split_seed"],
            "epochs": len(record.epochs),
            "best_epoch": record.best_epoch,
            "validation_top1": record.best_val_top1,
            "validation_loss": record.best_val_loss,
            "membership_sha256": record.config["membership_sha256"],
            "validation_sha256": record.config["validation_sha256"],
            "source_manifest_sha256": record.config["source_manifest_sha256"],
            "git_revision": record.config.get("git_revision"),
            "git_dirty": record.config.get("git_dirty"),
        }
        for filename in REQUIRED_RUN_FILES:
            row[f"{filename}_sha256"] = file_sha256(record.path / filename)
        rows.append(row)
    fields = tuple(rows[0])
    _write_csv_atomic(path, rows, fields)


def run_select(runs_root: Path, data_root: Path) -> dict[str, object]:
    del data_root  # The audited manifests, not the combined local manifest, are authoritative here.
    records = audit_runs(runs_root, REPO_ROOT)
    for variant in ("Mid32", "Mid224"):
        manifest = RESULT_ROOT / "provenance" / f"source_split_manifest_{variant.lower()}.csv"
        if not manifest.is_file() or file_sha256(manifest) != EXPECTED_MANIFEST_SHA256[variant]:
            raise EvidenceError(f"run audit first; verified {variant} source manifest is missing")

    selection_dir = RESULT_ROOT / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    lock = _persist_selection_lock(
        build_selection_lock(records), selection_dir / "selection_lock.json"
    )
    _write_selection_tables(records, selection_dir)
    _write_validation_plots(records, selection_dir)

    checkpoint_rows = copy_run_bundles(records, RESULT_ROOT / "checkpoints")
    _write_csv_atomic(
        RESULT_ROOT / "checkpoints" / "checkpoint_manifest.csv",
        checkpoint_rows,
        (
            "run_name",
            "variant",
            "recipe",
            "file",
            "size_bytes",
            "source_sha256",
            "copied_sha256",
            "source_path",
            "copied_path",
        ),
    )
    _write_run_manifest(records, RESULT_ROOT / "provenance" / "run_manifest.csv")
    result = {
        "ok": True,
        "winners": lock["winners"],
        "copied_files": len(checkpoint_rows),
        "copied_bytes": sum(int(row["size_bytes"]) for row in checkpoint_rows),
        "selection_lock_sha256": file_sha256(selection_dir / "selection_lock.json"),
    }
    print(json.dumps(result, indent=2))
    return result


def build_evaluate_command(
    python_executable: Path,
    repo_root: Path,
    data_root: Path,
    record: RunRecord,
    output_dir: Path,
) -> list[str]:
    return [
        str(Path(python_executable).resolve()),
        str(Path(repo_root).resolve() / "train_mid_experiment.py"),
        "--data-root",
        str(Path(data_root).resolve()),
        "--variant",
        record.variant,
        "--model",
        "l",
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--evaluate",
        str((record.path / "best_val.pt").resolve()),
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--threads",
        "2",
    ]


def _verify_raw_evaluation(record: RunRecord, output_dir: Path) -> None:
    required = ("test_metrics.json", "test_predictions.csv", "confusion_matrix.csv")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise EvidenceError(f"{record.name} evaluation output is incomplete: {missing}")
    with (output_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != 250:
        raise EvidenceError(f"{record.name} evaluation must contain 250 predictions")
    with (output_dir / "confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = [[int(value) for value in row] for row in csv.reader(handle)]
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        raise EvidenceError(f"{record.name} confusion matrix must be 5x5")
    if sum(sum(row) for row in matrix) != 250:
        raise EvidenceError(f"{record.name} confusion matrix must sum to 250")
    metrics = _read_json(output_dir / "test_metrics.json")
    if metrics.get("samples") != 250:
        raise EvidenceError(f"{record.name} test metrics sample count must be 250")


def run_evaluations(
    records: list[RunRecord],
    result_root: Path,
    evaluation_data_roots: dict[str, Path],
    python_executable: Path,
    scope: str,
) -> None:
    result_root = Path(result_root)
    lock_path = result_root / "selection" / "selection_lock.json"
    if not lock_path.is_file():
        raise EvidenceError(f"selection lock is required before evaluation: {lock_path}")
    lock_bytes = lock_path.read_bytes()
    lock = _read_json(lock_path)
    expected_winners = {"Mid32": "l_mid32_lr015", "Mid224": "l_mid224_baseline"}
    if lock.get("selection_source") != "validation_only" or lock.get("winners") != expected_winners:
        raise EvidenceError("selection lock is invalid or was not derived from validation only")
    if scope != "all":
        raise EvidenceError("only --scope all is supported for the fixed ten-run dossier")

    evaluation_root = result_root / "eval"
    for record in records:
        output_dir = evaluation_root / record.name
        if output_dir.exists():
            raise EvidenceError(f"evaluation requires a fresh output directory: {output_dir}")
    for record in records:
        data_root = evaluation_data_roots.get(record.variant)
        if data_root is None or not Path(data_root).is_dir():
            raise EvidenceError(f"missing exact evaluation data root for {record.variant}")
        checkpoint = record.path / "best_val.pt"
        if not checkpoint.is_file():
            raise EvidenceError(f"copied best checkpoint is missing: {checkpoint}")

    evaluation_root.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        output_dir = evaluation_root / record.name
        command = build_evaluate_command(
            python_executable, REPO_ROOT, evaluation_data_roots[record.variant], record, output_dir
        )
        print(f"[{index}/{len(records)}] evaluating {record.name}", flush=True)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode:
            raise EvidenceError(
                f"{record.name} evaluation failed with exit {process.returncode}\n"
                f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
            )
        _verify_raw_evaluation(record, output_dir)
        print(process.stdout.strip(), flush=True)
    if lock_path.read_bytes() != lock_bytes:
        raise EvidenceError("selection lock changed during test evaluation")


def _stage_selection_lock(source: Path, evaluation_root: Path) -> Path:
    destination = evaluation_root / "selection" / "selection_lock.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise EvidenceError("staged selection lock differs from the immutable source lock")
    else:
        shutil.copy2(source, destination)
    return destination


def run_evaluate(runs_root: Path, data_root: Path, scope: str) -> dict[str, object]:
    records = audit_runs(runs_root, REPO_ROOT)
    source_lock = RESULT_ROOT / "selection" / "selection_lock.json"
    if not source_lock.is_file():
        raise EvidenceError("selection lock is missing; run select before evaluate")
    source_lock_digest = file_sha256(source_lock)
    expected_core = build_selection_lock(records)
    existing_lock = _read_json(source_lock)
    if {key: value for key, value in existing_lock.items() if key != "created_at_utc"} != expected_core:
        raise EvidenceError("selection lock no longer matches audited validation evidence")

    copied_records: list[RunRecord] = []
    for record in records:
        copied_path = RESULT_ROOT / "checkpoints" / record.name
        copied_best = copied_path / "best_val.pt"
        if not copied_best.is_file() or file_sha256(copied_best) != file_sha256(record.path / "best_val.pt"):
            raise EvidenceError(f"copied best checkpoint mismatch for {record.name}")
        copied_records.append(replace(record, path=copied_path.resolve()))

    evaluation_root = REPO_ROOT / "runs" / "experiment_midterm_eval"
    _stage_selection_lock(source_lock, evaluation_root)
    manifest_root = RESULT_ROOT / "provenance"
    data_workspace = REPO_ROOT / "runs" / "experiment_midterm_eval_data"
    evaluation_data_roots = {
        variant: prepare_evaluation_data_root(
            data_root,
            variant,
            manifest_root / f"source_split_manifest_{variant.lower()}.csv",
            data_workspace,
        )
        for variant in ("Mid32", "Mid224")
    }
    run_evaluations(
        copied_records,
        result_root=evaluation_root,
        evaluation_data_roots=evaluation_data_roots,
        python_executable=Path(sys.executable),
        scope=scope,
    )
    if file_sha256(source_lock) != source_lock_digest:
        raise EvidenceError("source selection lock changed during evaluation")
    evaluation_dirs = list((evaluation_root / "eval").iterdir())
    result = {
        "ok": True,
        "evaluated_runs": len(evaluation_dirs),
        "selection_lock_sha256": source_lock_digest,
        "evaluation_root": str(evaluation_root / "eval"),
    }
    print(json.dumps(result, indent=2))
    return result


def compute_classification_rows(
    targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> list[dict[str, object]]:
    from sklearn.metrics import classification_report

    if len(targets) != len(predictions) or not targets:
        raise EvidenceError("classification metrics require equal non-empty target/prediction lists")
    labels = list(range(len(class_names)))
    report = classification_report(
        list(targets),
        list(predictions),
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, object]] = []
    for class_name in class_names:
        values = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1-score": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    rows.append(
        {
            "class_name": "accuracy",
            "precision": "",
            "recall": "",
            "f1-score": report["accuracy"],
            "support": len(targets),
        }
    )
    for average in ("macro avg", "weighted avg"):
        values = report[average]
        rows.append(
            {
                "class_name": average,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1-score": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    return rows


def enrich_predictions(raw_predictions: Any, class_map: dict[str, int]) -> Any:
    import pandas as pd

    required = ["path", "target", "prediction"]
    if list(raw_predictions.columns) != required:
        raise EvidenceError(f"raw prediction fields must be exactly {required}")
    inverse = {index: name for name, index in class_map.items()}
    if set(inverse) != set(range(len(class_map))):
        raise EvidenceError("class map indices must be contiguous from zero")
    frame = raw_predictions.copy()
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="raise").astype(int)
    if not set(frame["target"]).issubset(inverse) or not set(frame["prediction"]).issubset(inverse):
        raise EvidenceError("prediction contains an unknown class index")
    frame.insert(2, "target_name", frame["target"].map(inverse))
    frame.insert(4, "prediction_name", frame["prediction"].map(inverse))
    frame["correct"] = frame["target"] == frame["prediction"]
    return frame[["path", "target", "target_name", "prediction", "prediction_name", "correct"]]


def _prepare_matplotlib() -> Any:
    matplotlib_cache = REPO_ROOT / "runs" / ".report-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _import_plotting() -> tuple[Any, Any]:
    plt = _prepare_matplotlib()
    import seaborn as sns

    return plt, sns


def _write_dataframe_atomic(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _read_labeled_confusion(path: Path) -> Any:
    import pandas as pd

    frame = pd.read_csv(path)
    expected = ["actual_class", *(f"predicted_{name}" for name in CLASSES)]
    if list(frame.columns) != expected or frame["actual_class"].tolist() != list(CLASSES):
        raise EvidenceError(f"invalid labeled confusion matrix: {path}")
    return frame


def _top_confusion(matrix: Any) -> tuple[str, str, int]:
    best = ("", "", -1)
    for actual_index, actual_name in enumerate(CLASSES):
        for predicted_index, predicted_name in enumerate(CLASSES):
            if actual_index == predicted_index:
                continue
            count = int(matrix[actual_index][predicted_index])
            if count > best[2]:
                best = (actual_name, predicted_name, count)
    return best


def write_model_report(record: RunRecord, evaluation_dir: Path, model_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    evaluation_dir = Path(evaluation_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    _verify_raw_evaluation(record, evaluation_dir)
    metrics = _read_json(evaluation_dir / "test_metrics.json")
    raw_predictions = pd.read_csv(evaluation_dir / "test_predictions.csv")
    predictions = enrich_predictions(raw_predictions, record.config["class_to_idx"])
    if len(predictions) != 250:
        raise EvidenceError(f"{record.name} must have 250 enriched predictions")
    targets = predictions["target"].tolist()
    predicted = predictions["prediction"].tolist()
    matrix = confusion_matrix(targets, predicted, labels=list(range(len(CLASSES))))
    raw_matrix = np.loadtxt(evaluation_dir / "confusion_matrix.csv", delimiter=",", dtype=int)
    if not np.array_equal(matrix, raw_matrix):
        raise EvidenceError(f"{record.name} raw confusion matrix disagrees with predictions")
    computed_top1 = 100.0 * float(predictions["correct"].mean())
    classification_rows = compute_classification_rows(targets, predicted, CLASSES)
    macro = next(row for row in classification_rows if row["class_name"] == "macro avg")
    if abs(computed_top1 - float(metrics["top1"])) > 1e-12:
        raise EvidenceError(f"{record.name} Top-1 disagrees with predictions")
    if abs(float(macro["f1-score"]) - float(metrics["macro_f1"])) > 1e-12:
        raise EvidenceError(f"{record.name} Macro F1 disagrees with predictions")

    _write_json_atomic(model_dir / "test_metrics.json", metrics)
    _write_dataframe_atomic(predictions, model_dir / "predictions.csv")
    confusion_frame = pd.DataFrame(
        matrix, columns=[f"predicted_{name}" for name in CLASSES]
    )
    confusion_frame.insert(0, "actual_class", CLASSES)
    _write_dataframe_atomic(confusion_frame, model_dir / "confusion_matrix.csv")
    classification_frame = pd.DataFrame(classification_rows)
    _write_dataframe_atomic(classification_frame, model_dir / "classification_report.csv")

    plt, sns = _import_plotting()
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(f"{record.name} — historical test confusion matrix")
    figure.tight_layout()
    figure.savefig(model_dir / "confusion_matrix.png", dpi=170)
    plt.close(figure)

    epochs = [int(row["epoch"]) for row in record.epochs]
    train_top1 = [float(row["train_top1"]) for row in record.epochs]
    val_top1 = [float(row["val_top1"]) for row in record.epochs]
    train_loss = [float(row["train_loss"]) for row in record.epochs]
    val_loss = [float(row["val_loss"]) for row in record.epochs]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(epochs, train_top1, label="train Top-1")
    axes[0].plot(epochs, val_top1, label="validation Top-1")
    axes[0].axvline(record.best_epoch, color="black", linestyle="--", alpha=0.6, label="best val epoch")
    axes[0].set(xlabel="Epoch", ylabel="Top-1 (%)", title="Accuracy curves")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, train_loss, label="train objective")
    axes[1].plot(epochs, val_loss, label="hard-label validation loss")
    axes[1].axvline(record.best_epoch, color="black", linestyle="--", alpha=0.6)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss curves")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle(record.name)
    figure.tight_layout()
    figure.savefig(model_dir / "learning_curves.png", dpi=170)
    plt.close(figure)

    per_class = [row for row in classification_rows if row["class_name"] in CLASSES]
    weakest = min(per_class, key=lambda row: float(row["f1-score"]))
    actual, confused_as, count = _top_confusion(matrix)
    result_root = model_dir.parents[1]
    lock_path = result_root / "selection" / "selection_lock.json"
    selection_text = ""
    if lock_path.is_file():
        lock = _read_json(lock_path)
        selected = lock.get("winners", {}).get(record.variant) == record.name
        selection_text = "Selected winner" if selected else "Not selected; reported as a post-selection diagnostic"
    checkpoint_dir = result_root / "checkpoints" / record.name
    if not checkpoint_dir.is_dir():
        checkpoint_dir = record.path
    best_digest = file_sha256(checkpoint_dir / "best_val.pt")
    last_digest = file_sha256(checkpoint_dir / "last.pt")
    lines = [
        f"# {record.name}",
        "",
        f"- Variant: `{record.variant}`",
        f"- Recipe: `{record.recipe}`",
        f"- Seed: `{record.config['seed']}`; split seed: `{record.config['split_seed']}`",
        f"- Validation role: {selection_text or 'See the root validation-only selection report.'}",
        f"- Best validation epoch: **{record.best_epoch}**",
        f"- Validation Top-1: **{record.best_val_top1:.2f}%**",
        f"- Hard-label validation loss: **{record.best_val_loss:.6f}**",
        f"- Historical test Top-1: **{float(metrics['top1']):.2f}%**",
        f"- Historical test Macro F1: **{float(metrics['macro_f1']):.6f}**",
        f"- Historical test loss: **{float(metrics['loss']):.6f}**",
        "",
        "## Class-level observations",
        "",
        f"The lowest per-class F1 is `{weakest['class_name']}` at {float(weakest['f1-score']):.4f}. "
        f"The largest off-diagonal cell is `{actual}` → `{confused_as}` ({count} images).",
        "",
        "These test results are post-selection descriptive evidence. They did not choose this recipe or checkpoint.",
        "",
        "## Evidence",
        "",
        "- [Test metrics](test_metrics.json)",
        "- [Predictions](predictions.csv)",
        "- [Classification report](classification_report.csv)",
        "- [Confusion matrix CSV](confusion_matrix.csv)",
        "- [Confusion matrix figure](confusion_matrix.png)",
        "- [Learning curves](learning_curves.png)",
        f"- [Copied best validation checkpoint](../../checkpoints/{record.name}/best_val.pt) — SHA-256 `{best_digest}`",
        f"- [Copied final checkpoint](../../checkpoints/{record.name}/last.pt) — SHA-256 `{last_digest}`",
        "",
        "## Limitations",
        "",
        "This is one seed on a historical 250-image test split. It provides no multi-seed uncertainty estimate. "
        "Training objectives differ when label smoothing changes, so training-loss values must not be compared across smoothing recipes.",
        "",
    ]
    (model_dir / "README.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _write_comparison_plots(
    records: list[RunRecord], result_root: Path, test_rows: list[dict[str, object]]
) -> None:
    plt, sns = _import_plotting()
    comparisons = result_root / "comparisons"
    labels = [recipe.name for recipe in RECIPES]
    x = list(range(len(labels)))
    width = 0.36
    for metric, ylabel, filename in (
        ("test_top1", "Historical test Top-1 (%)", "test_top1_by_recipe.png"),
        ("test_macro_f1", "Historical test Macro F1", "macro_f1_by_recipe.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        for offset, variant, color in ((-width / 2, "Mid32", "#4c78a8"), (width / 2, "Mid224", "#f58518")):
            values = [
                float(
                    next(
                        row
                        for row in test_rows
                        if row["variant"] == variant and row["recipe"] == recipe
                    )[metric]
                )
                for recipe in labels
            ]
            bars = axis.bar([position + offset for position in x], values, width, label=variant, color=color)
            axis.bar_label(bars, fmt="%.2f" if metric == "test_top1" else "%.3f", padding=2, fontsize=8)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} by recipe (post-selection descriptive)")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(comparisons / filename, dpi=170)
        plt.close(figure)

    for variant in ("Mid32", "Mid224"):
        figure, axis = plt.subplots(figsize=(10, 5.5))
        for record in [item for item in records if item.variant == variant]:
            if record.epochs:
                axis.plot(
                    [int(row["epoch"]) for row in record.epochs],
                    [float(row["val_top1"]) for row in record.epochs],
                    label=record.recipe,
                    linewidth=1.5,
                )
        axis.set(
            xlabel="Epoch",
            ylabel="Validation Top-1 (%)",
            title=f"{variant} validation learning curves",
        )
        if axis.lines:
            axis.legend(ncol=2)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(comparisons / f"learning_curves_{variant.lower()}.png", dpi=170)
        plt.close(figure)

        variant_records = [item for item in records if item.variant == variant]
        figure, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharex=True, sharey=True)
        for axis, record in zip(axes, variant_records):
            frame = _read_labeled_confusion(
                result_root / "models" / record.name / "confusion_matrix.csv"
            )
            matrix = frame.drop(columns="actual_class").to_numpy(dtype=int)
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=CLASSES,
                yticklabels=CLASSES,
                ax=axis,
            )
            axis.set_title(record.recipe)
            axis.tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Actual")
        for axis in axes:
            axis.set_xlabel("Predicted")
        figure.suptitle(f"{variant} historical test confusion matrices (descriptive)")
        figure.tight_layout()
        figure.savefig(comparisons / f"confusion_matrices_{variant.lower()}.png", dpi=170)
        plt.close(figure)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def write_comparison_report(
    records: list[RunRecord], result_root: Path, selection_lock: dict[str, object]
) -> None:
    import pandas as pd

    result_root = Path(result_root)
    comparisons = result_root / "comparisons"
    comparisons.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    winners = selection_lock["winners"]
    for record in records:
        validation_rows.append(
            {
                "run_name": record.name,
                "variant": record.variant,
                "recipe": record.recipe,
                "best_epoch": record.best_epoch,
                "validation_top1": record.best_val_top1,
                "validation_loss": record.best_val_loss,
                "selected_by_validation": winners[record.variant] == record.name,
            }
        )
        model_dir = result_root / "models" / record.name
        metrics = _read_json(model_dir / "test_metrics.json")
        test_rows.append(
            {
                "run_name": record.name,
                "variant": record.variant,
                "recipe": record.recipe,
                "selection_status": (
                    "validation_winner" if winners[record.variant] == record.name else "post_selection_diagnostic"
                ),
                "test_loss": metrics["loss"],
                "test_top1": metrics["top1"],
                "test_macro_f1": metrics["macro_f1"],
                "test_samples": metrics["samples"],
            }
        )
        report = pd.read_csv(model_dir / "classification_report.csv")
        for row in report.to_dict(orient="records"):
            class_rows.append(
                {
                    "run_name": record.name,
                    "variant": record.variant,
                    "recipe": record.recipe,
                    **row,
                }
            )
    _write_csv_atomic(
        comparisons / "validation_metrics.csv",
        validation_rows,
        tuple(validation_rows[0]),
    )
    _write_csv_atomic(comparisons / "test_metrics.csv", test_rows, tuple(test_rows[0]))
    _write_csv_atomic(comparisons / "class_metrics.csv", class_rows, tuple(class_rows[0]))
    _write_comparison_plots(records, result_root, test_rows)

    validation_table = [
        (
            record.variant,
            record.recipe,
            record.best_epoch,
            f"{record.best_val_top1:.2f}",
            f"{record.best_val_loss:.6f}",
            "yes" if winners[record.variant] == record.name else "no",
        )
        for record in records
    ]
    test_by_name = {str(row["run_name"]): row for row in test_rows}
    test_table = [
        (
            record.variant,
            record.recipe,
            f"{float(test_by_name[record.name]['test_top1']):.2f}",
            f"{float(test_by_name[record.name]['test_macro_f1']):.6f}",
            f"{float(test_by_name[record.name]['test_loss']):.6f}",
            test_by_name[record.name]["selection_status"],
        )
        for record in records
    ]
    delta_lines: list[str] = []
    for variant in ("Mid32", "Mid224"):
        baseline = next(item for item in records if item.variant == variant and item.recipe == "baseline")
        parts = []
        for record in [item for item in records if item.variant == variant and item.recipe != "baseline"]:
            delta = record.best_val_top1 - baseline.best_val_top1
            parts.append(f"`{record.recipe}` {delta:+.2f} pp")
        delta_lines.append(f"- **{variant}:** " + ", ".join(parts) + " versus its baseline.")

    confusion_lines: list[str] = []
    for variant in ("Mid32", "Mid224"):
        winner = next(record for record in records if record.name == winners[variant])
        frame = _read_labeled_confusion(result_root / "models" / winner.name / "confusion_matrix.csv")
        actual, predicted, count = _top_confusion(frame.drop(columns="actual_class").to_numpy(dtype=int))
        classification = pd.read_csv(result_root / "models" / winner.name / "classification_report.csv")
        per_class = classification[classification["class_name"].isin(CLASSES)]
        weakest = per_class.loc[per_class["f1-score"].astype(float).idxmin()]
        confusion_lines.append(
            f"- **{variant} validation winner `{winner.name}`:** lowest class F1 is "
            f"`{weakest['class_name']}` ({float(weakest['f1-score']):.4f}); largest confusion is "
            f"`{actual}` → `{predicted}` ({count}/250 images)."
        )

    representative = records[0].config.get("complexity", {})
    parameter_count = representative.get("learnable_parameters", "n/a")
    mid32_flops = next(
        (record.config.get("complexity", {}).get("flops") for record in records if record.variant == "Mid32"),
        "n/a",
    )
    mid224_flops = next(
        (record.config.get("complexity", {}).get("flops") for record in records if record.variant == "Mid224"),
        "n/a",
    )
    lock_sha = selection_lock.get("lock_sha256", "not-recorded")
    lines = [
        "# TickNet-L midterm smoothing and learning-rate experiment",
        "",
        "## Outcome",
        "",
        "The **validation-only selection** was frozen before any test inference. It selected "
        f"`{winners['Mid32']}` for Mid32 and `{winners['Mid224']}` for Mid224. "
        "All ten test evaluations below are post-selection descriptive diagnostics and cannot replace those winners.",
        "",
        f"Selection-lock SHA-256: `{lock_sha}`.",
        "",
        "## Experimental controls",
        "",
        "All runs use TickNet-L revision `ticknet-l-v1`, seed 42, split seed 123, 200 epochs, "
        "the same five-class mapping, and the same validation membership hash. The only intended changes are "
        "label smoothing (0.05/0.10) or initial SGD learning rate (0.05/0.15).",
        "",
        f"The architecture has {parameter_count} learnable parameters. Recorded FLOPs are {mid32_flops} "
        f"for Mid32 and {mid224_flops} for Mid224, both within the protocol limits.",
        "",
        "## Validation selection evidence",
        "",
        *_markdown_table(
            ("Variant", "Recipe", "Best epoch", "Val Top-1 (%)", "Val loss", "Selected"),
            validation_table,
        ),
        "",
        "Validation Top-1 changes relative to the same-resolution baseline:",
        "",
        *delta_lines,
        "",
        "Training loss is not compared across smoothing settings because label smoothing changes the training objective. "
        "Hard-label validation loss remains listed as comparable supporting evidence.",
        "",
        "## Historical test diagnostics (after selection)",
        "",
        *_markdown_table(
            ("Variant", "Recipe", "Test Top-1 (%)", "Macro F1", "Test loss", "Role"),
            test_table,
        ),
        "",
        "This table is intentionally descriptive rather than a test-based ranking. The 250-image historical test split "
        "was accessed only after the immutable validation lock was written.",
        "",
        "## Class-level behavior",
        "",
        *confusion_lines,
        "",
        "Full per-class precision, recall, F1, support, predictions, and confusion matrices are available in each model directory.",
        "",
        "## Reproduce and verify",
        "",
        "Create the ignored Python environment described in the implementation plan, attach the paired Mid32/Mid224 data, "
        "and run the phases in order. Evaluation requires a fresh `runs/experiment_midterm_eval/` directory. "
        "If the original training `runs/` directory is unavailable, use "
        "`docs/results/experiment_midterm_smoothing_lr/checkpoints` as `--runs`.",
        "",
        "```powershell",
        "$python = 'runs/.report-venv/Scripts/python.exe'",
        "& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py audit --runs runs --data-root data",
        "& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py select --runs runs --data-root data",
        "& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py evaluate --runs runs --data-root data --scope all",
        "& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py report --runs runs --data-root data",
        "& $python docs/results/experiment_midterm_smoothing_lr/tools/verify_results.py --source-runs runs --results docs/results/experiment_midterm_smoothing_lr",
        "```",
        "",
        "## Evidence index",
        "",
        "- [Selection lock and validation plots](selection/)",
        "- [Cross-run CSVs and figures](comparisons/)",
        "- [Checkpoint copy manifest](checkpoints/checkpoint_manifest.csv)",
        "- [Run and source provenance](provenance/)",
        "- Model reports:",
        *[f"  - [{record.name}](models/{record.name}/)" for record in records],
        "",
        "## Provenance and limitations",
        "",
        "The Kaggle configs record `git_revision = null` and `git_dirty = null`. Therefore, the 110 recorded source-file "
        "hashes (all matched) are the primary code-provenance evidence. The local combined manifest was not used for "
        "inference; exact Mid32/Mid224 manifests were rebuilt and matched the checkpoint hashes.",
        "",
        "This experiment has one seed per recipe, so it cannot estimate variance or statistical significance. The test "
        "split has only 250 images and is historical; its metrics should not be treated as fresh benchmark estimates. "
        "No retraining or full-train phase was run for this dossier.",
        "",
    ]
    (result_root / "README.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _write_provenance(records: list[RunRecord], result_root: Path, lock_sha: str) -> None:
    from importlib.metadata import version

    source_checks = {
        "ok": True,
        "checked": sum(len(record.config["source_hashes"]) for record in records),
        "matched": sum(len(record.config["source_hashes"]) for record in records),
        "git_revision_null_runs": sum(record.config.get("git_revision") is None for record in records),
        "git_dirty_null_runs": sum(record.config.get("git_dirty") is None for record in records),
        "manifests": EXPECTED_MANIFEST_SHA256,
        "runs": [
            {
                "run_name": record.name,
                "source_hashes": record.config["source_hashes"],
                "all_match": True,
            }
            for record in records
        ],
    }
    _write_json_atomic(result_root / "provenance" / "source_checks.json", source_checks)
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "evaluation_entrypoint": "train_mid_experiment.py --evaluate",
        "checkpoint_role": "best_val",
        "selection_lock_sha256": lock_sha,
        "packages": {
            package: version(package)
            for package in (
                "torch",
                "torchvision",
                "numpy",
                "pillow",
                "pandas",
                "matplotlib",
                "seaborn",
                "scikit-learn",
            )
        },
    }
    _write_json_atomic(
        result_root / "provenance" / "evaluation_environment.json", environment
    )


def _write_artifact_manifest(result_root: Path) -> list[dict[str, object]]:
    manifest_path = result_root / "provenance" / "artifact_manifest.csv"
    excluded = {
        manifest_path.resolve(),
        (result_root / "provenance" / "verification.json").resolve(),
    }
    rows = [
        {
            "relative_path": path.relative_to(result_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(result_root.rglob("*"))
        if (
            path.is_file()
            and path.resolve() not in excluded
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
    ]
    _write_csv_atomic(
        manifest_path, rows, ("relative_path", "size_bytes", "sha256")
    )
    return rows


def run_report(runs_root: Path, data_root: Path) -> dict[str, object]:
    del data_root
    records = audit_runs(runs_root, REPO_ROOT)
    lock_path = RESULT_ROOT / "selection" / "selection_lock.json"
    if not lock_path.is_file():
        raise EvidenceError("selection lock is missing; run select before report")
    lock = _read_json(lock_path)
    expected_core = build_selection_lock(records)
    if {key: value for key, value in lock.items() if key != "created_at_utc"} != expected_core:
        raise EvidenceError("selection lock does not match validation evidence")
    lock_sha = file_sha256(lock_path)
    lock_for_report = {**lock, "lock_sha256": lock_sha}
    evaluation_root = REPO_ROOT / "runs" / "experiment_midterm_eval" / "eval"
    observed = {path.name for path in evaluation_root.iterdir() if path.is_dir()} if evaluation_root.is_dir() else set()
    if observed != set(expected_run_names()):
        raise EvidenceError("raw evaluation outputs are missing or contain unexpected runs")

    for record in records:
        write_model_report(
            record,
            evaluation_root / record.name,
            RESULT_ROOT / "models" / record.name,
        )
    write_comparison_report(records, RESULT_ROOT, lock_for_report)
    _write_provenance(records, RESULT_ROOT, lock_sha)
    artifact_rows = _write_artifact_manifest(RESULT_ROOT)
    result = {
        "ok": True,
        "model_reports": len(records),
        "artifact_manifest_rows": len(artifact_rows),
        "selection_lock_sha256": lock_sha,
    }
    print(json.dumps(result, indent=2))
    return result


def _validate_manifest_counts(path: Path, variant: str) -> None:
    counts = {(split, class_name): 0 for split in ("train", "test") for class_name in CLASSES}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["split"], row["class_name"])
            if row["variant"] != variant or key not in counts:
                raise EvidenceError(f"unexpected manifest row in {path}: {row}")
            counts[key] += 1
    for (split, class_name), observed in counts.items():
        expected = EXPECTED_SPLIT_COUNT[split]
        if observed != expected:
            raise EvidenceError(
                f"{variant} {split}/{class_name} image count {observed} != {expected}"
            )


def run_audit(runs_root: Path, data_root: Path) -> dict[str, object]:
    records = audit_runs(runs_root, REPO_ROOT)
    provenance = RESULT_ROOT / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, str]] = {}
    for variant in ("Mid32", "Mid224"):
        output = provenance / f"source_split_manifest_{variant.lower()}.csv"
        digest = write_source_manifest(data_root, variant, output)
        _validate_manifest_counts(output, variant)
        expected = EXPECTED_MANIFEST_SHA256[variant]
        if digest != expected:
            raise EvidenceError(f"{variant} source manifest hash mismatch: {digest} != {expected}")
        manifests[variant] = {"path": str(output), "sha256": digest}
    source_checks = sum(len(record.config["source_hashes"]) for record in records)
    result = {
        "ok": True,
        "runs": len(records),
        "source_hashes": f"{source_checks}/{source_checks}",
        "manifests": manifests,
    }
    print(json.dumps(result, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    audit = subparsers.add_parser("audit", help="audit runs and rebuild source manifests")
    audit.add_argument("--runs", type=Path, default=Path("runs"))
    audit.add_argument("--data-root", type=Path, default=Path("data"))
    select = subparsers.add_parser("select", help="freeze validation winners and copy runs")
    select.add_argument("--runs", type=Path, default=Path("runs"))
    select.add_argument("--data-root", type=Path, default=Path("data"))
    evaluate = subparsers.add_parser("evaluate", help="run post-selection test inference")
    evaluate.add_argument("--runs", type=Path, default=Path("runs"))
    evaluate.add_argument("--data-root", type=Path, default=Path("data"))
    evaluate.add_argument("--scope", choices=("all",), default="all")
    report = subparsers.add_parser("report", help="generate model and comparison evidence")
    report.add_argument("--runs", type=Path, default=Path("runs"))
    report.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.phase == "audit":
        return run_audit(args.runs.resolve(), args.data_root.resolve())
    if args.phase == "select":
        return run_select(args.runs.resolve(), args.data_root.resolve())
    if args.phase == "evaluate":
        return run_evaluate(args.runs.resolve(), args.data_root.resolve(), args.scope)
    if args.phase == "report":
        return run_report(args.runs.resolve(), args.data_root.resolve())
    raise EvidenceError(f"unsupported phase: {args.phase}")


if __name__ == "__main__":
    try:
        main()
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
