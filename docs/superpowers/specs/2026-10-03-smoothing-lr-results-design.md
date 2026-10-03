# Smoothing/Learning-Rate Experiment Results Design

## Objective

Build a reproducible result dossier for the ten completed TickNet-L midterm
runs under `runs/`, covering five recipes for each of `Mid32` and `Mid224`.
The dossier lives at
`docs/results/experiment_midterm_smoothing_lr/` and follows the evidence style
of `docs/results/Model_Basic/` while preserving the branch protocol: recipe
selection is frozen from validation before test inference is run.

## Scope and fixed decisions

- Process exactly ten run directories: baseline, smoothing 0.05, smoothing
  0.10, LR 0.05, and LR 0.15 for both resolutions.
- Treat all runs as seed-42, single-seed evidence. Do not claim statistical
  significance or stability across seeds.
- Lock the validation-selected recipe before test evaluation:
  - `Mid32`: `l_mid32_lr015` at validation Top-1 91.04%.
  - `Mid224`: `l_mid224_baseline` at validation Top-1 93.88%.
- Evaluate all ten `best_val.pt` checkpoints on the historical test split for
  descriptive reporting only. Test metrics must not alter the locked winners.
- Do not retrain, run `--full-train`, or modify trainer/model/config logic.
- Do not compare training loss across smoothing levels because their training
  objectives differ. Hard-label validation loss remains comparable.
- Copy each complete run directory into the result dossier, including both
  `best_val.pt` and `last.pt`, rather than keeping only selected weights.

## Input audit and provenance

Before inference, the result builder must verify for all ten runs:

1. `epochs.csv` contains exactly epochs 1 through 200.
2. `config.json`, `validation_split.json`, `best_val.pt`, and `last.pt` exist.
3. Variant, recipe, seed, split seed, class map, sample counts, architecture
   revision, trainer revision, membership hash, and validation hash agree with
   the expected experiment matrix.
4. The best epoch recomputed from `epochs.csv` agrees with the checkpoint
   selection rule: maximum validation Top-1, then minimum validation loss.
5. Every source hash recorded by the Kaggle config matches the current source.
6. Every copied artifact has the same byte size and SHA-256 as its source.

The Kaggle configs record `git_revision = null`; the report must disclose this
and use the recorded source hashes as the primary code-provenance evidence.

The local combined `data/split_manifest.csv` does not match the per-variant
manifest hashes recorded by the Kaggle checkpoints. The result builder must
recreate the Mid32 and Mid224 source manifests using the same ordering, fields,
image validation, SHA-256 calculation, and CSV serialization as the Kaggle
notebooks. Inference may proceed only if their SHA-256 values equal the values
stored in the corresponding checkpoints/configs.

## Evaluation policy

Create `selection/selection_lock.json` from validation data before any test
command. It records candidates, selection rule, winners, validation metrics,
split hashes, and a timestamp.

For each run, invoke the existing explicit evaluation path against
`best_val.pt` using the reconstructed variant-specific data root. Evaluation
must use hard-label cross-entropy and load checkpoints with `weights_only=True`.
The raw evaluation produces test loss, Top-1, Macro F1, predictions, and a
confusion matrix. Derived artifacts add class names, per-class precision,
recall, F1, support, readable heatmaps, and learning curves.

All ten test evaluations are post-selection diagnostics. The overall report
must distinguish validation selection tables from test-description tables and
must not rank or replace winners using test results.

## Result layout

```text
docs/results/experiment_midterm_smoothing_lr/
├── README.md
├── selection/
│   ├── selection_lock.json
│   ├── validation_summary.csv
│   ├── recipe_vs_baseline.csv
│   ├── validation_top1_by_recipe.png
│   ├── validation_loss_by_recipe.png
│   └── best_epoch_by_recipe.png
├── models/<run_name>/
│   ├── README.md
│   ├── test_metrics.json
│   ├── predictions.csv
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   └── learning_curves.png
├── checkpoints/<run_name>/
│   ├── best_val.pt
│   ├── last.pt
│   ├── config.json
│   ├── epochs.csv
│   └── validation_split.json
├── checkpoints/checkpoint_manifest.csv
├── comparisons/
│   ├── validation_metrics.csv
│   ├── test_metrics.csv
│   ├── class_metrics.csv
│   ├── test_top1_by_recipe.png
│   ├── macro_f1_by_recipe.png
│   ├── learning_curves_mid32.png
│   ├── learning_curves_mid224.png
│   ├── confusion_matrices_mid32.png
│   └── confusion_matrices_mid224.png
├── provenance/
│   ├── run_manifest.csv
│   ├── source_split_manifest_mid32.csv
│   ├── source_split_manifest_mid224.csv
│   ├── source_checks.json
│   ├── evaluation_environment.json
│   └── artifact_manifest.csv
└── tools/
    ├── build_experiment_results.py
    └── verify_results.py
```

The root `README.md` is the complete experiment report, not merely an index.
Each model README records its recipe, best validation epoch, validation and
test metrics, class-level behavior, checkpoint hashes, linked figures, and
limitations. `predictions.csv` includes path, numeric and named target/predicted
classes, and a correctness flag.

## Analysis content

The final report will cover:

- smoothing 0.05/0.10 deltas from the same-variant baseline;
- LR 0.05/0.15 deltas from the same-variant baseline;
- convergence speed, best epoch, validation Top-1, and validation loss;
- post-selection test Top-1, loss, Macro F1, per-class precision/recall/F1,
  confusion patterns, and resolution-specific failure modes;
- parameter/FLOP compliance and constant architecture across recipes;
- provenance limitations (`git_revision = null`), historical-test limitation,
  and lack of multi-seed uncertainty estimates.

## Tooling and execution

Use one result builder and one independent verifier. A temporary Python
environment may be created below the ignored `runs/` directory; it is not part
of the result commit. Required packages are PyTorch/torchvision, NumPy, Pillow,
pandas, Matplotlib, seaborn, and scikit-learn.

The builder exposes ordered phases:

1. `audit`: validate runs and reconstruct exact manifests.
2. `select`: generate and freeze validation-only selection evidence.
3. `evaluate --scope all`: run all ten post-selection test inferences.
4. `report`: generate per-model and cross-recipe artifacts.
5. `verify`: independently reload files, recompute metrics/hashes, and reject
   missing, inconsistent, or test-selected evidence.

Each phase is idempotent or refuses to overwrite inconsistent evidence.
Evaluation output directories are fresh; existing evidence is never silently
overwritten.

## Verification gates

Completion requires:

- unit tests for run discovery, best-epoch selection, manifest serialization,
  class metrics, and selection-lock enforcement;
- successful loading of every `best_val.pt` and `last.pt` with
  `weights_only=True`;
- exact 250-row predictions and 5x5 confusion matrix for every evaluated run;
- equality between metrics recomputed from predictions and reported JSON/CSV;
- exact SHA-256 equality for all files copied from `runs/`;
- all Markdown links resolving and all expected PNG/CSV/JSON files present;
- a clean independent verification report with no failures.

## Commit strategy

Keep the final branch history to at most two concise commits. Existing Kaggle
notebook working-tree changes are outside this task and must not be staged.

1. Design/provenance commit for this approved specification.
2. One result commit containing tools, tests, copied run evidence, inference
   artifacts, model READMEs, comparisons, and the final report.
