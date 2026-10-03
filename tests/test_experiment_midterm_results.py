from __future__ import annotations

import base64
import builtins
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    REPO_ROOT
    / "docs"
    / "results"
    / "experiment_midterm_smoothing_lr"
    / "tools"
    / "build_experiment_results.py"
)
VERIFIER_PATH = BUILDER_PATH.with_name("verify_results.py")

spec = importlib.util.spec_from_file_location("experiment_midterm_results_builder", BUILDER_PATH)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)

verifier_spec = importlib.util.spec_from_file_location(
    "experiment_midterm_results_verifier", VERIFIER_PATH
)
assert verifier_spec is not None and verifier_spec.loader is not None
verifier = importlib.util.module_from_spec(verifier_spec)
sys.modules[verifier_spec.name] = verifier
verifier_spec.loader.exec_module(verifier)

EvidenceError = builder.EvidenceError
RunRecord = builder.RunRecord
audit_runs = builder.audit_runs
build_selection_lock = builder.build_selection_lock
build_evaluate_command = builder.build_evaluate_command
compute_classification_rows = builder.compute_classification_rows
copy_run_bundles = builder.copy_run_bundles
enrich_predictions = builder.enrich_predictions
expected_run_names = builder.expected_run_names
prepare_evaluation_data_root = builder.prepare_evaluation_data_root
select_best_epoch = builder.select_best_epoch
run_evaluations = builder.run_evaluations
write_comparison_report = builder.write_comparison_report
write_source_manifest = builder.write_source_manifest
verify_result_tree = verifier.verify_result_tree


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAN0lEQVR4nO3RQQ0AAAjD"
    "wAH+PSOhfPj1BIykVPXkU7+uxwMH/gCZCJkImQiZCJkImQiZCJkoZAHY4QBGrYv0nAAAAABJRU5ErkJggg=="
)
EXPECTED_FIXTURE_BYTES = (
    b"variant,split,class_name,filename,size_bytes,sha256\r\n"
    b"Mid32,train,bird,sample.png,112,"
    b"672f604c70fdf14a06dc3a32e927a3636b45d8d982cb0c7c87830cc60d54bd94\r\n"
)
EXPECTED_FIXTURE_SHA256 = "f51e4c122bc902ee2664522f5f50e321c75248662097e214b4f0303ec9d614e4"
CLASSES = ("bird", "cat", "dog", "frog", "horse")


def epoch_row(epoch: int, top1: float, loss: float) -> dict[str, str]:
    return {"epoch": str(epoch), "val_top1": str(top1), "val_loss": str(loss)}


@pytest.fixture
def tiny_mid_data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for split in ("train", "test"):
        for class_name in CLASSES:
            (root / "Mid32" / split / class_name).mkdir(parents=True)
    (root / "Mid32" / "train" / "bird" / "sample.png").write_bytes(PNG_BYTES)
    (root / "split_manifest.csv").write_bytes(b"original-combined-manifest\r\n")
    return root


@pytest.fixture
def complete_runs_with_tampered_source_hash(tmp_path: Path) -> Path:
    source_root = REPO_ROOT / "runs"
    target_root = tmp_path / "runs"
    target_root.mkdir()
    for run_name in expected_run_names():
        source = source_root / run_name
        target = target_root / run_name
        target.mkdir()
        for filename in ("config.json", "epochs.csv", "validation_split.json"):
            shutil.copy2(source / filename, target / filename)
        (target / "best_val.pt").write_bytes(b"fixture-best")
        (target / "last.pt").write_bytes(b"fixture-last")

    config_path = target_root / expected_run_names()[0] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["source_hashes"]["train_mid_experiment.py"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return target_root


@pytest.fixture
def real_summary_fixture(tmp_path: Path) -> list[object]:
    summaries = {
        "l_mid32_baseline": (173, 90.40, 0.500444689655304),
        "l_mid32_smoothing005": (167, 90.64, 0.3353641463279724),
        "l_mid32_smoothing010": (191, 90.72, 0.35884732117652896),
        "l_mid32_lr005": (197, 90.56, 0.4916291002988815),
        "l_mid32_lr015": (186, 91.04, 0.4418038331210613),
        "l_mid224_baseline": (177, 93.88, 0.2763168884038925),
        "l_mid224_smoothing005": (160, 93.44, 0.2367592604637146),
        "l_mid224_smoothing010": (176, 93.64, 0.2633008383989334),
        "l_mid224_lr005": (188, 93.32, 0.2965208834171295),
        "l_mid224_lr015": (198, 93.84, 0.254891860127449),
    }
    records = []
    for name in expected_run_names():
        variant = "Mid32" if "mid32" in name else "Mid224"
        recipe = name.split(f"l_{variant.lower()}_", 1)[1]
        epoch, top1, loss = summaries[name]
        records.append(
            RunRecord(
                name=name,
                variant=variant,
                recipe=recipe,
                path=tmp_path / name,
                config={
                    "seed": 42,
                    "split_seed": 123,
                    "membership_sha256": builder.EXPECTED_MEMBERSHIP_SHA256,
                    "validation_sha256": builder.EXPECTED_VALIDATION_SHA256,
                    "source_manifest_sha256": builder.EXPECTED_MANIFEST_SHA256[variant],
                },
                epochs=[],
                best_epoch=epoch,
                best_val_top1=top1,
                best_val_loss=loss,
            )
        )
    return records


@pytest.fixture
def run_fixture(tmp_path: Path) -> object:
    run_path = tmp_path / "source" / "l_mid32_baseline"
    run_path.mkdir(parents=True)
    for index, filename in enumerate(builder.REQUIRED_RUN_FILES):
        (run_path / filename).write_bytes(f"fixture-{index}-{filename}".encode())
    return RunRecord(
        name="l_mid32_baseline",
        variant="Mid32",
        recipe="baseline",
        path=run_path,
        config={"seed": 42},
        epochs=[],
        best_epoch=1,
        best_val_top1=90.0,
        best_val_loss=0.5,
    )


@pytest.fixture
def complete_lock(tmp_path: Path) -> Path:
    path = tmp_path / "selection" / "selection_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_source": "validation_only",
                "winners": {
                    "Mid32": "l_mid32_lr015",
                    "Mid224": "l_mid224_baseline",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def complete_result_fixture(tmp_path: Path) -> SimpleNamespace:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    run_name = "l_mid32_baseline"
    source = runs / run_name
    copied = results / "checkpoints" / run_name
    source.mkdir(parents=True)
    copied.mkdir(parents=True)
    for index, filename in enumerate(builder.REQUIRED_RUN_FILES):
        payload = f"fixture-{index}-{filename}".encode()
        (source / filename).write_bytes(payload)
        (copied / filename).write_bytes(payload)

    model = results / "models" / run_name
    model.mkdir(parents=True)
    rows = []
    for target in range(5):
        for index in range(50):
            rows.append(
                {
                    "path": f"{CLASSES[target]}/{index}.png",
                    "target": target,
                    "target_name": CLASSES[target],
                    "prediction": target,
                    "prediction_name": CLASSES[target],
                    "correct": True,
                }
            )
    pd.DataFrame(rows).to_csv(model / "predictions.csv", index=False)
    (model / "test_metrics.json").write_text(
        json.dumps({"loss": 0.1, "top1": 100.0, "samples": 250, "macro_f1": 1.0}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [[name, *([50 if row == column else 0 for column in range(5)])] for row, name in enumerate(CLASSES)],
        columns=["actual_class", *(f"predicted_{name}" for name in CLASSES)],
    ).to_csv(model / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        [
            {
                "class_name": name,
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 50 if name in CLASSES else 250,
            }
            for name in (*CLASSES, "accuracy", "macro avg", "weighted avg")
        ]
    ).to_csv(model / "classification_report.csv", index=False)
    (model / "README.md").write_text("[predictions](predictions.csv)\n", encoding="utf-8")
    (results / "README.md").write_text(
        f"[model predictions](models/{run_name}/predictions.csv)\n", encoding="utf-8"
    )
    selection = results / "selection"
    selection.mkdir()
    (selection / "selection_lock.json").write_text(
        json.dumps(
            {
                "selection_source": "validation_only",
                "winners": {"Mid32": "l_mid32_lr015", "Mid224": "l_mid224_baseline"},
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(runs=runs, results=results, run_name=run_name)


def test_audit_requires_exact_ten_complete_200_epoch_runs(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="missing expected runs"):
        audit_runs(tmp_path / "runs", REPO_ROOT)


def test_select_best_epoch_uses_top1_then_lower_loss() -> None:
    rows = [epoch_row(1, 90.0, 0.4), epoch_row(2, 90.0, 0.3)]
    assert select_best_epoch(rows) == (2, 90.0, 0.3)


def test_source_manifest_matches_kaggle_csv_contract(
    tiny_mid_data: Path, tmp_path: Path
) -> None:
    output = tmp_path / "manifest.csv"
    digest = write_source_manifest(tiny_mid_data, "Mid32", output)
    assert digest == EXPECTED_FIXTURE_SHA256
    assert output.read_bytes() == EXPECTED_FIXTURE_BYTES


def test_evaluation_data_root_links_images_without_mutating_source(
    tiny_mid_data: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "source_manifest.csv"
    write_source_manifest(tiny_mid_data, "Mid32", manifest)
    original = (tiny_mid_data / "split_manifest.csv").read_bytes()

    root = prepare_evaluation_data_root(
        tiny_mid_data, "Mid32", manifest, tmp_path / "workspace"
    )

    linked = root / "Mid32" / "train" / "bird" / "sample.png"
    assert linked.samefile(tiny_mid_data / "Mid32" / "train" / "bird" / "sample.png")
    assert (root / "split_manifest.csv").read_bytes() == EXPECTED_FIXTURE_BYTES
    assert (tiny_mid_data / "split_manifest.csv").read_bytes() == original


def test_audit_rejects_source_hash_mismatch(
    complete_runs_with_tampered_source_hash: Path,
) -> None:
    with pytest.raises(EvidenceError, match="source hash"):
        audit_runs(complete_runs_with_tampered_source_hash, REPO_ROOT)


def test_audit_rejects_missing_recorded_source_hash() -> None:
    config = json.loads(
        (REPO_ROOT / "runs" / "l_mid32_baseline" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    config["source_hashes"].pop("models/common.py")
    with pytest.raises(EvidenceError, match="source hash set"):
        builder._validate_config("l_mid32_baseline", config, REPO_ROOT)


def test_selection_lock_freezes_expected_validation_winners(
    real_summary_fixture: list[object],
) -> None:
    lock = build_selection_lock(real_summary_fixture)
    assert lock["winners"] == {
        "Mid32": "l_mid32_lr015",
        "Mid224": "l_mid224_baseline",
    }
    assert lock["selection_source"] == "validation_only"
    assert all("test" not in key for candidate in lock["candidates"] for key in candidate)


def test_copy_run_bundles_preserves_every_file_and_sha256(
    run_fixture: object, tmp_path: Path
) -> None:
    manifest = copy_run_bundles([run_fixture], tmp_path / "checkpoints")
    assert {row["file"] for row in manifest} == set(builder.REQUIRED_RUN_FILES)
    assert all(row["source_sha256"] == row["copied_sha256"] for row in manifest)
    assert all(not Path(str(row["source_path"])).is_absolute() for row in manifest)
    assert all(not Path(str(row["copied_path"])).is_absolute() for row in manifest)


def test_copy_refuses_inconsistent_existing_destination(
    run_fixture: object, tmp_path: Path
) -> None:
    destination = tmp_path / "checkpoints" / run_fixture.name
    destination.mkdir(parents=True)
    (destination / "config.json").write_bytes(b"tampered")
    with pytest.raises(EvidenceError, match="refusing to overwrite"):
        copy_run_bundles([run_fixture], tmp_path / "checkpoints")


def test_validation_plots_use_repo_local_writable_matplotlib_cache(
    real_summary_fixture: list[object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    builder._write_validation_plots(real_summary_fixture, tmp_path)
    cache = Path(os.environ["MPLCONFIGDIR"])
    assert cache.is_dir()
    assert cache.is_relative_to(REPO_ROOT / "runs")


def test_evaluate_requires_existing_validation_lock(
    run_fixture: object, tmp_path: Path
) -> None:
    with pytest.raises(EvidenceError, match="selection lock"):
        run_evaluations(
            [run_fixture],
            result_root=tmp_path,
            evaluation_data_roots={"Mid32": tmp_path / "data"},
            python_executable=Path(sys.executable),
            scope="all",
        )


def test_evaluate_command_uses_best_checkpoint_and_matching_variant(
    run_fixture: object, tmp_path: Path
) -> None:
    command = build_evaluate_command(
        Path(sys.executable), REPO_ROOT, tmp_path / "data", run_fixture, tmp_path / "eval"
    )
    assert command[command.index("--evaluate") + 1].endswith("best_val.pt")
    assert command[command.index("--variant") + 1] == run_fixture.variant


def test_evaluate_refuses_nonfresh_output(
    run_fixture: object, complete_lock: Path, tmp_path: Path
) -> None:
    del complete_lock
    (tmp_path / "eval" / run_fixture.name).mkdir(parents=True)
    with pytest.raises(EvidenceError, match="fresh output"):
        run_evaluations(
            [run_fixture],
            result_root=tmp_path,
            evaluation_data_roots={"Mid32": tmp_path / "data"},
            python_executable=Path(sys.executable),
            scope="all",
        )


def test_class_metrics_match_known_confusion_matrix() -> None:
    rows = compute_classification_rows(
        targets=[0, 0, 1, 1], predictions=[0, 1, 1, 1], class_names=["bird", "cat"]
    )
    by_name = {row["class_name"]: row for row in rows}
    assert by_name["macro avg"]["f1-score"] == pytest.approx(0.7333333333333334)
    assert by_name["bird"]["recall"] == pytest.approx(0.5)


def test_predictions_include_names_and_correctness() -> None:
    raw = pd.DataFrame(
        [
            {"path": "a.png", "target": 0, "prediction": 0},
            {"path": "b.png", "target": 1, "prediction": 0},
        ]
    )
    frame = enrich_predictions(raw, {"bird": 0, "cat": 1})
    assert list(frame.columns) == [
        "path",
        "target",
        "target_name",
        "prediction",
        "prediction_name",
        "correct",
    ]
    assert frame["correct"].tolist() == [True, False]
    assert frame["target_name"].tolist() == ["bird", "cat"]


def test_report_does_not_reselect_from_test(
    real_summary_fixture: list[object], tmp_path: Path
) -> None:
    for index, record in enumerate(real_summary_fixture):
        model_dir = tmp_path / "models" / record.name
        model_dir.mkdir(parents=True)
        top1 = 99.9 if record.name == "l_mid32_baseline" else 50.0 + index
        (model_dir / "test_metrics.json").write_text(
            json.dumps({"loss": 0.2, "top1": top1, "samples": 250, "macro_f1": 0.8}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "class_name": name,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 50,
                }
                for name in (*CLASSES, "accuracy", "macro avg", "weighted avg")
            ]
        ).to_csv(model_dir / "classification_report.csv", index=False)
        matrix = pd.DataFrame(
            [[name, *([50 if row == column else 0 for column in range(5)])] for row, name in enumerate(CLASSES)],
            columns=["actual_class", *(f"predicted_{name}" for name in CLASSES)],
        )
        matrix.to_csv(model_dir / "confusion_matrix.csv", index=False)

    lock = build_selection_lock(real_summary_fixture)
    lock["lock_sha256"] = "fixture-lock-sha256"
    write_comparison_report(real_summary_fixture, tmp_path, lock)

    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "validation-only selection" in text
    assert "fixture-lock-sha256" in text
    assert "l_mid32_lr015" in text
    assert "## Reproduce and verify" in text
    assert "build_experiment_results.py audit" in text
    assert "verify_results.py --source-runs" in text


def test_plot_stack_configures_cache_before_importing_seaborn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "seaborn":
            assert "MPLCONFIGDIR" in os.environ
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    builder._import_plotting()


def test_verifier_detects_tampered_checkpoint(
    complete_result_fixture: SimpleNamespace,
) -> None:
    checkpoint = (
        complete_result_fixture.results
        / "checkpoints"
        / complete_result_fixture.run_name
        / "best_val.pt"
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    result = verify_result_tree(
        complete_result_fixture.runs, complete_result_fixture.results
    )
    assert not result["ok"]
    assert "checkpoint_sha256" in result["failed_checks"]


def test_verifier_detects_metric_or_prediction_tampering(
    complete_result_fixture: SimpleNamespace,
) -> None:
    predictions_path = (
        complete_result_fixture.results
        / "models"
        / complete_result_fixture.run_name
        / "predictions.csv"
    )
    predictions = pd.read_csv(predictions_path)
    predictions.loc[0, ["prediction", "prediction_name", "correct"]] = [1, "cat", False]
    predictions.to_csv(predictions_path, index=False)
    result = verify_result_tree(
        complete_result_fixture.runs, complete_result_fixture.results
    )
    assert not result["ok"]
    assert "recomputed_metrics" in result["failed_checks"]


def test_verifier_detects_broken_markdown_link(
    complete_result_fixture: SimpleNamespace,
) -> None:
    (complete_result_fixture.results / "README.md").write_text(
        "[missing](models/does-not-exist/predictions.csv)\n", encoding="utf-8"
    )
    result = verify_result_tree(
        complete_result_fixture.runs, complete_result_fixture.results
    )
    assert "markdown_links" in result["failed_checks"]


def test_verifier_reports_empty_predictions_without_crashing(
    complete_result_fixture: SimpleNamespace,
) -> None:
    prediction_path = (
        complete_result_fixture.results
        / "models"
        / complete_result_fixture.run_name
        / "predictions.csv"
    )
    pd.DataFrame(
        columns=[
            "path",
            "target",
            "target_name",
            "prediction",
            "prediction_name",
            "correct",
        ]
    ).to_csv(prediction_path, index=False)
    result = verify_result_tree(
        complete_result_fixture.runs, complete_result_fixture.results
    )
    assert not result["ok"]
    assert "recomputed_metrics" in result["failed_checks"]


def test_independent_verifier_recomputes_training_source_hashes(tmp_path: Path) -> None:
    result_root = tmp_path / "docs" / "results" / "experiment"
    source_runs = tmp_path / "runs"
    result_root.mkdir(parents=True)
    source = tmp_path / "train_mid_experiment.py"
    source.write_text("original\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    run = source_runs / "l_mid32_baseline"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps({"source_hashes": {"train_mid_experiment.py": digest}}),
        encoding="utf-8",
    )
    source.write_text("tampered\n", encoding="utf-8")
    ok, details = verifier._verify_training_sources(source_runs, result_root)
    assert not ok
    assert any("source hash mismatch" in detail for detail in details)


def test_artifact_manifest_excludes_python_cache(tmp_path: Path) -> None:
    cache = tmp_path / "tools" / "__pycache__" / "tool.cpython-312.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"generated-cache")
    (tmp_path / "README.md").write_text("report\n", encoding="utf-8")
    rows = builder._write_artifact_manifest(tmp_path)
    assert [row["relative_path"] for row in rows] == ["README.md"]


def test_verifier_accepts_complete_expected_png_inventory(tmp_path: Path) -> None:
    expected = [
        *(f"selection/{name}" for name in (
            "validation_top1_by_recipe.png",
            "validation_loss_by_recipe.png",
            "best_epoch_by_recipe.png",
        )),
        *(f"models/{run}/{name}" for run in expected_run_names() for name in (
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
    ]
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_BYTES)
    ok, details = verifier._verify_images(tmp_path)
    assert ok, details


def test_copied_run_bundles_are_declared_binary_for_git() -> None:
    protected = (
        "docs/results/experiment_midterm_smoothing_lr/"
        "checkpoints/l_mid32_baseline/config.json"
    )
    result = subprocess.run(
        ["git", "check-attr", "text", "--", protected],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("text: unset")


def test_generated_text_artifacts_use_lf_line_endings(tmp_path: Path) -> None:
    json_path = tmp_path / "artifact.json"
    csv_path = tmp_path / "artifact.csv"
    frame_path = tmp_path / "frame.csv"
    builder._write_json_atomic(json_path, {"ok": True})
    builder._write_csv_atomic(csv_path, [{"value": 1}], ("value",))
    builder._write_dataframe_atomic(pd.DataFrame([{"value": 1}]), frame_path)
    assert b"\r\n" not in json_path.read_bytes()
    assert b"\r\n" not in csv_path.read_bytes()
    assert b"\r\n" not in frame_path.read_bytes()
