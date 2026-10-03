# Smoothing/Learning-Rate Experiment Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified, reproducible dossier for all ten completed TickNet-L smoothing/LR runs, including full copied run bundles, post-selection test inference, per-model evidence, figures, and an overall comparison report.

**Architecture:** A single builder CLI owns ordered `audit`, `select`, `evaluate`, and `report` phases; an independent verifier recomputes hashes and metrics from final artifacts. The pipeline reuses `train_mid_experiment.py --evaluate` for inference, freezes validation winners before test access, and aborts on manifest, source, checkpoint, or selection-lock inconsistencies.

**Tech Stack:** Python 3.12, PyTorch/torchvision, NumPy, Pillow, pandas, Matplotlib, seaborn, scikit-learn, pytest, PowerShell, SHA-256.

**Spec:** `docs/superpowers/specs/2026-10-03-smoothing-lr-results-design.md`

## Global Constraints

- Process exactly ten seed-42 run directories: five recipes for `Mid32` and five for `Mid224`.
- Select from validation only: `l_mid32_lr015` and `l_mid224_baseline`; evaluate all ten tests only after persisting the lock.
- Evaluate `best_val.pt`; preserve both `best_val.pt` and `last.pt` in the copied run bundle.
- Copy every source run file byte-for-byte to `docs/results/experiment_midterm_smoothing_lr/checkpoints/<run_name>/` and verify SHA-256.
- Rebuild variant-specific source manifests exactly and require the recorded checkpoint/config hash before inference.
- Require manifest SHA-256 `5e012ceef22d363877407ce7a0ecc2d8b42dcc7cb891f981124eae2523b0ad8e` for Mid32 and `028a2eda6f9c9c385875925e3cadbd83979ad3718df7652e3864a01df01a1929` for Mid224.
- Do not modify `train_mid_experiment.py`, models, configs, training recipes, checkpoints, or original `runs/` content.
- Do not compare training loss across smoothing levels or select/rank recipes from test metrics.
- Treat the test split as historical and all test metrics as post-selection descriptive evidence.
- Keep the temporary environment under ignored `runs/.report-venv/`.
- Do not stage the existing Kaggle notebook/README working-tree changes.
- Do not commit task-by-task: the already-created design commit plus one final result commit is the two-commit limit.

## Review Focus

1. A missing/partial/tampered run must fail audit before selection or copying; covered in Task 1 tests.
2. A locally rebuilt manifest that differs by rows, ordering, line endings, or hash must block inference; covered in Task 1 tests.
3. Missing, altered, or test-derived selection lock must block evaluation/report generation; covered in Task 2 and Task 3 tests.
4. Evaluation must always use `best_val.pt`, the matching variant, and a fresh output directory; covered in Task 3 tests.
5. Copied binaries, predictions, confusion matrices, derived metrics, and Markdown links must survive independent tamper detection; covered in Task 5 tests.

---

### Task 1: Run audit and exact manifest reconstruction

**Files:**
- Create: `docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py`
- Create: `tests/test_experiment_midterm_results.py`

**Interfaces:**
- Produces: `RunRecord`; `expected_run_names() -> tuple[str, ...]`; `select_best_epoch(rows: list[dict[str, str]]) -> tuple[int, float, float]`; `audit_runs(runs_root: Path, repo_root: Path) -> list[RunRecord]`; `write_source_manifest(data_root: Path, variant: str, output_path: Path) -> str`; `prepare_evaluation_data_root(data_root: Path, variant: str, manifest_path: Path, workspace: Path) -> Path`; and CLI phase `audit`.
- `RunRecord` fields: `name`, `variant`, `recipe`, `path`, `config`, `epochs`, `best_epoch`, `best_val_top1`, `best_val_loss`.
- Later tasks consume the ordered `list[RunRecord]` and the two verified manifest paths.

- [ ] **Step 1: Create the ignored report environment**

Run:

```powershell
uv venv runs/.report-venv --python 3.12
uv pip install --python runs/.report-venv/Scripts/python.exe torch torchvision numpy pillow pandas matplotlib seaborn scikit-learn pytest
```

Expected: the environment Python imports every listed package; no environment file appears in `git status` because `runs/` is ignored.

- [ ] **Step 2: Write failing audit and manifest tests**

```python
def test_audit_requires_exact_ten_complete_200_epoch_runs(tmp_path):
    with pytest.raises(EvidenceError, match="missing expected runs"):
        audit_runs(tmp_path / "runs", REPO_ROOT)

def test_select_best_epoch_uses_top1_then_lower_loss():
    rows = [epoch_row(1, 90.0, 0.4), epoch_row(2, 90.0, 0.3)]
    assert select_best_epoch(rows) == (2, 90.0, 0.3)

def test_source_manifest_matches_kaggle_csv_contract(tiny_mid_data, tmp_path):
    digest = write_source_manifest(tiny_mid_data, "Mid32", tmp_path / "manifest.csv")
    assert digest == EXPECTED_FIXTURE_SHA256
    assert (tmp_path / "manifest.csv").read_bytes() == EXPECTED_FIXTURE_BYTES

def test_evaluation_data_root_links_images_without_mutating_source(tiny_mid_data, tmp_path):
    root = prepare_evaluation_data_root(tiny_mid_data, "Mid32", MANIFEST, tmp_path)
    assert (root / "Mid32" / "train" / "bird" / "sample.png").samefile(
        tiny_mid_data / "Mid32" / "train" / "bird" / "sample.png"
    )
    assert (tiny_mid_data / "split_manifest.csv").read_bytes() == ORIGINAL_MANIFEST_BYTES

def test_audit_rejects_source_hash_mismatch(complete_run_fixture):
    complete_run_fixture.config["source_hashes"]["train_mid_experiment.py"] = "0" * 64
    with pytest.raises(EvidenceError, match="source hash"):
        audit_runs(complete_run_fixture.root, REPO_ROOT)
```

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
```

Expected: FAIL because the builder module and interfaces do not exist.

- [ ] **Step 4: Implement audit interfaces and `audit` CLI phase**

Implement the exact interfaces above. Validate required files, run identity, epochs `1..200`, recipe matrix, config fields, source hashes, split hashes, recomputed best epoch, and image dimensions/counts. Serialize manifests with fields `variant,split,class_name,filename,size_bytes,sha256`, Kaggle ordering, UTF-8 without BOM, and CSV `\r\n` records. Build an ignored evaluation data root using a directory symlink on Unix or an NTFS junction on Windows; never copy 50,500 images or modify `data/split_manifest.csv`.

- [ ] **Step 5: Run audit tests and the real audit**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
runs/.report-venv/Scripts/python.exe docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py audit --runs runs --data-root data
```

Expected: tests PASS; real audit reports ten complete runs, 110/110 source hashes, and manifest hashes equal the two hashes recorded by the configs.

### Task 2: Validation lock and complete run-bundle copying

**Files:**
- Modify: `docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py`
- Modify: `tests/test_experiment_midterm_results.py`
- Generate: `docs/results/experiment_midterm_smoothing_lr/selection/*`
- Generate: `docs/results/experiment_midterm_smoothing_lr/checkpoints/*`
- Generate: `docs/results/experiment_midterm_smoothing_lr/provenance/run_manifest.csv`

**Interfaces:**
- Consumes: audited `RunRecord` values from Task 1.
- Produces: `build_selection_lock(records) -> dict`, `copy_run_bundles(records, destination) -> list[dict]`, CLI phase `select`, `selection_lock.json`, and checkpoint manifest rows.

- [ ] **Step 1: Write failing lock and copy tests**

```python
def test_selection_lock_freezes_expected_validation_winners(real_summary_fixture):
    lock = build_selection_lock(real_summary_fixture)
    assert lock["winners"] == {
        "Mid32": "l_mid32_lr015",
        "Mid224": "l_mid224_baseline",
    }
    assert lock["selection_source"] == "validation_only"

def test_copy_run_bundles_preserves_every_file_and_sha256(run_fixture, tmp_path):
    manifest = copy_run_bundles([run_fixture], tmp_path / "checkpoints")
    assert {row["file"] for row in manifest} == {
        "best_val.pt", "last.pt", "config.json", "epochs.csv", "validation_split.json"
    }
    assert all(row["source_sha256"] == row["copied_sha256"] for row in manifest)

def test_copy_refuses_inconsistent_existing_destination(run_fixture, tmp_path):
    tamper_destination(tmp_path)
    with pytest.raises(EvidenceError, match="refusing to overwrite"):
        copy_run_bundles([run_fixture], tmp_path / "checkpoints")
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q -k "selection_lock or copy_run_bundles or copy_refuses"
```

Expected: FAIL because selection/copy interfaces do not exist.

- [ ] **Step 3: Implement selection and exact copy behavior**

Write validation summaries/deltas and plots, persist the lock atomically, copy all five files from every run, and write size/hash/provenance rows. Never derive a winner from test fields.

- [ ] **Step 4: Run tests and real `select` phase**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
runs/.report-venv/Scripts/python.exe docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py select --runs runs --data-root data
```

Expected: tests PASS; lock names the two fixed winners; 50 copied files match their sources; no test metrics exist yet.

### Task 3: Post-selection inference for all ten best checkpoints

**Files:**
- Modify: `docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py`
- Modify: `tests/test_experiment_midterm_results.py`
- Generate: temporary evaluation outputs below ignored `runs/experiment_midterm_eval/`

**Interfaces:**
- Consumes: audited records, exact data roots, immutable selection lock, and copied `best_val.pt` files.
- Produces: `build_evaluate_command(python_executable: Path, repo_root: Path, data_root: Path, record: RunRecord, output_dir: Path) -> list[str]`; `run_evaluations(records: list[RunRecord], result_root: Path, evaluation_data_roots: dict[str, Path], python_executable: Path, scope: str) -> None`; and CLI phase `evaluate --scope all`.

- [ ] **Step 1: Write failing inference-policy tests**

```python
def test_evaluate_requires_existing_validation_lock(tmp_path, audited_records):
    with pytest.raises(EvidenceError, match="selection lock"):
        run_evaluations(audited_records, result_root=tmp_path, scope="all")

def test_evaluate_command_uses_best_checkpoint_and_matching_variant(record, tmp_path):
    command = build_evaluate_command(PYTHON, REPO_ROOT, tmp_path / "data", record, tmp_path / "eval")
    assert command[command.index("--evaluate") + 1].endswith("best_val.pt")
    assert command[command.index("--variant") + 1] == record.variant

def test_evaluate_refuses_nonfresh_output(record, complete_lock, tmp_path):
    (tmp_path / "eval" / record.name).mkdir(parents=True)
    with pytest.raises(EvidenceError, match="fresh output"):
        run_evaluations([record], result_root=tmp_path, scope="all")
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q -k "evaluate_requires or evaluate_command or evaluate_refuses"
```

Expected: FAIL before inference orchestration exists.

- [ ] **Step 3: Implement evaluation orchestration**

Build exact commands using `train_mid_experiment.py --evaluate`, reconstructed manifest roots, recorded variant, copied `best_val.pt`, CPU/auto device, and fresh per-run output. Verify each output has `test_metrics.json`, `test_predictions.csv`, `confusion_matrix.csv`, and 250 predictions before continuing.

- [ ] **Step 4: Run tests, then all ten real evaluations**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
runs/.report-venv/Scripts/python.exe docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py evaluate --runs runs --data-root data --scope all
```

Expected: tests PASS; ten evaluation directories complete; no selection-lock byte changes; every confusion-matrix cell sum is 250.

### Task 4: Per-model evidence and aggregate report

**Files:**
- Modify: `docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py`
- Modify: `tests/test_experiment_midterm_results.py`
- Generate: `docs/results/experiment_midterm_smoothing_lr/models/<run_name>/*`
- Generate: `docs/results/experiment_midterm_smoothing_lr/comparisons/*`
- Generate: `docs/results/experiment_midterm_smoothing_lr/README.md`
- Generate: `docs/results/experiment_midterm_smoothing_lr/provenance/{source_checks.json,evaluation_environment.json,artifact_manifest.csv}`

**Interfaces:**
- Consumes: frozen selection and ten raw evaluation outputs.
- Produces: `compute_classification_rows(targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]) -> list[dict[str, object]]`; `enrich_predictions(raw_predictions: pandas.DataFrame, class_map: dict[str, int]) -> pandas.DataFrame`; `write_model_report(record: RunRecord, evaluation_dir: Path, model_dir: Path) -> None`; `write_comparison_report(records: list[RunRecord], result_root: Path, selection_lock: dict[str, object]) -> None`; and CLI phase `report`.

- [ ] **Step 1: Write failing metric/report tests**

```python
def test_class_metrics_match_known_confusion_matrix():
    rows = compute_classification_rows(KNOWN_TARGETS, KNOWN_PREDICTIONS, CLASS_NAMES)
    assert rows["macro avg"]["f1-score"] == pytest.approx(KNOWN_MACRO_F1)

def test_predictions_include_names_and_correctness(raw_predictions):
    frame = enrich_predictions(raw_predictions, CLASS_MAP)
    assert list(frame.columns) == [
        "path", "target", "target_name", "prediction", "prediction_name", "correct"
    ]

def test_report_does_not_reselect_from_test(tmp_path, locked_results):
    write_comparison_report(locked_results, tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "validation-only selection" in text
    assert locked_results.lock_sha256 in text
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q -k "class_metrics or predictions_include or report_does_not"
```

Expected: FAIL before derived metric/report functions exist.

- [ ] **Step 3: Implement model and comparison artifact generation**

Create labeled predictions, labeled confusion matrices, classification reports, per-model heatmaps/learning curves/READMEs, cross-recipe tables/plots, root report, source checks, environment capture, and artifact manifest. Keep validation selection and post-selection test analysis in separate sections.

- [ ] **Step 4: Run tests and real `report` phase**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
runs/.report-venv/Scripts/python.exe docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py report --runs runs --data-root data
```

Expected: tests PASS; ten model directories each contain the seven specified files; all comparison/provenance files from the spec exist.

### Task 5: Independent verification and tamper tests

**Files:**
- Create: `docs/results/experiment_midterm_smoothing_lr/tools/verify_results.py`
- Modify: `tests/test_experiment_midterm_results.py`
- Generate: `docs/results/experiment_midterm_smoothing_lr/provenance/verification.json`

**Interfaces:**
- Consumes: source `runs/`, final result tree, selection lock, copied bundles, and predictions.
- Produces: `verify_result_tree(source_runs: Path, result_root: Path) -> dict[str, object]` and process exit 0 only when every check passes.

- [ ] **Step 1: Write failing independent-verifier tests**

```python
def test_verifier_detects_tampered_checkpoint(complete_result_fixture):
    tamper_checkpoint(complete_result_fixture)
    result = verify_result_tree(complete_result_fixture.runs, complete_result_fixture.results)
    assert not result["ok"]
    assert "checkpoint_sha256" in result["failed_checks"]

def test_verifier_detects_metric_or_prediction_tampering(complete_result_fixture):
    tamper_prediction(complete_result_fixture)
    result = verify_result_tree(complete_result_fixture.runs, complete_result_fixture.results)
    assert not result["ok"]
    assert "recomputed_metrics" in result["failed_checks"]

def test_verifier_detects_broken_markdown_link(complete_result_fixture):
    remove_link_target(complete_result_fixture)
    result = verify_result_tree(complete_result_fixture.runs, complete_result_fixture.results)
    assert "markdown_links" in result["failed_checks"]
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q -k "verifier_detects"
```

Expected: FAIL because the independent verifier does not exist.

- [ ] **Step 3: Implement independent verification**

Recompute source/copy hashes, prediction counts, confusion matrices, Top-1, loss presence, Macro F1, class reports, selection-lock digest, expected tree contents, image readability, CSV/JSON parsing, and relative Markdown link targets without importing builder conclusions.

- [ ] **Step 4: Run the complete test and verification suite**

Run:

```powershell
runs/.report-venv/Scripts/python.exe -m pytest tests/test_experiment_midterm_results.py -q
runs/.report-venv/Scripts/python.exe -m pytest tests -q
runs/.report-venv/Scripts/python.exe docs/results/experiment_midterm_smoothing_lr/tools/verify_results.py --source-runs runs --results docs/results/experiment_midterm_smoothing_lr
```

Expected: all tests PASS; verifier prints `ok: true`; `verification.json` has no failed checks.

### Task 6: Final review and second/final commit

**Files:**
- Modify if evidence requires corrections: `docs/results/experiment_midterm_smoothing_lr/README.md`
- Include: `docs/superpowers/plans/2026-10-03-smoothing-lr-results.md`
- Include: `tests/test_experiment_midterm_results.py`
- Include: all generated files under `docs/results/experiment_midterm_smoothing_lr/`

**Interfaces:**
- Consumes: verified final dossier from Tasks 1–5.
- Produces: one final result commit and a concise execution handoff.

- [ ] **Step 1: Review conclusions against raw tables**

Confirm every numerical statement in the root/model READMEs is backed by a CSV/JSON file, every winner is validation-selected, every test statement is labeled post-selection, and single-seed limitations are explicit.

- [ ] **Step 2: Re-run fresh completion gates**

Run the full pytest suite, independent verifier, `git diff --check`, generated-file inventory, total-byte summary, and SHA-256 comparison once more. Expected: zero test/verifier/diff failures and 50/50 copied run files exact.

- [ ] **Step 3: Stage only in-scope files**

Stage the plan, test file, and `docs/results/experiment_midterm_smoothing_lr/`. Confirm `git diff --cached --name-only` excludes `docs/kaggle/README.md` and all untracked Kaggle notebooks.

- [ ] **Step 4: Create the second and final commit**

```powershell
git commit -m "results(midterm): add smoothing and LR experiment report"
```

- [ ] **Step 5: Report final evidence**

Provide commit hash, test counts, verifier status, copied checkpoint bytes, validation winners, post-selection test summary, and links to the root report and model directories.
