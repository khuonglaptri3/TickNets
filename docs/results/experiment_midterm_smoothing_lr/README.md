# TickNet-L midterm smoothing and learning-rate experiment

## Outcome

The **validation-only selection** was frozen before any test inference. It selected `l_mid32_lr015` for Mid32 and `l_mid224_baseline` for Mid224. All ten test evaluations below are post-selection descriptive diagnostics and cannot replace those winners.

Selection-lock SHA-256: `0d8a5e720f511f69442dca303f9a9e9a03dd12e2995d6dc7bec287710d962704`.

## Experimental controls

All runs use TickNet-L revision `ticknet-l-v1`, seed 42, split seed 123, 200 epochs, the same five-class mapping, and the same validation membership hash. The only intended changes are label smoothing (0.05/0.10) or initial SGD learning rate (0.05/0.15).

The architecture has 1096260 learnable parameters. Recorded FLOPs are 157820864 for Mid32 and 796760240 for Mid224, both within the protocol limits.

## Validation selection evidence

| Variant | Recipe | Best epoch | Val Top-1 (%) | Val loss | Selected |
| --- | --- | --- | --- | --- | --- |
| Mid32 | baseline | 173 | 90.40 | 0.500445 | no |
| Mid32 | smoothing005 | 167 | 90.64 | 0.335364 | no |
| Mid32 | smoothing010 | 191 | 90.72 | 0.358847 | no |
| Mid32 | lr005 | 197 | 90.56 | 0.491629 | no |
| Mid32 | lr015 | 186 | 91.04 | 0.441804 | yes |
| Mid224 | baseline | 177 | 93.88 | 0.276317 | yes |
| Mid224 | smoothing005 | 160 | 93.44 | 0.236759 | no |
| Mid224 | smoothing010 | 176 | 93.64 | 0.263301 | no |
| Mid224 | lr005 | 188 | 93.32 | 0.296521 | no |
| Mid224 | lr015 | 198 | 93.84 | 0.254892 | no |

Validation Top-1 changes relative to the same-resolution baseline:

- **Mid32:** `smoothing005` +0.24 pp, `smoothing010` +0.32 pp, `lr005` +0.16 pp, `lr015` +0.64 pp versus its baseline.
- **Mid224:** `smoothing005` -0.44 pp, `smoothing010` -0.24 pp, `lr005` -0.56 pp, `lr015` -0.04 pp versus its baseline.

Training loss is not compared across smoothing settings because label smoothing changes the training objective. Hard-label validation loss remains listed as comparable supporting evidence.

## Historical test diagnostics (after selection)

| Variant | Recipe | Test Top-1 (%) | Macro F1 | Test loss | Role |
| --- | --- | --- | --- | --- | --- |
| Mid32 | baseline | 90.40 | 0.903817 | 0.405454 | post_selection_diagnostic |
| Mid32 | smoothing005 | 89.60 | 0.895510 | 0.375396 | post_selection_diagnostic |
| Mid32 | smoothing010 | 91.20 | 0.911626 | 0.347748 | post_selection_diagnostic |
| Mid32 | lr005 | 92.00 | 0.919466 | 0.340118 | post_selection_diagnostic |
| Mid32 | lr015 | 92.40 | 0.923933 | 0.305484 | validation_winner |
| Mid224 | baseline | 95.20 | 0.951956 | 0.224384 | validation_winner |
| Mid224 | smoothing005 | 96.00 | 0.960132 | 0.169884 | post_selection_diagnostic |
| Mid224 | smoothing010 | 97.20 | 0.971999 | 0.177574 | post_selection_diagnostic |
| Mid224 | lr005 | 94.00 | 0.940221 | 0.217440 | post_selection_diagnostic |
| Mid224 | lr015 | 94.80 | 0.947906 | 0.185778 | post_selection_diagnostic |

This table is intentionally descriptive rather than a test-based ranking. The 250-image historical test split was accessed only after the immutable validation lock was written.

## Class-level behavior

- **Mid32 validation winner `l_mid32_lr015`:** lowest class F1 is `dog` (0.8750); largest confusion is `dog` → `cat` (7/250 images).
- **Mid224 validation winner `l_mid224_baseline`:** lowest class F1 is `dog` (0.9293); largest confusion is `dog` → `cat` (4/250 images).

Full per-class precision, recall, F1, support, predictions, and confusion matrices are available in each model directory.

## Reproduce and verify

Create the ignored Python environment described in the implementation plan, attach the paired Mid32/Mid224 data, and run the phases in order. Evaluation requires a fresh `runs/experiment_midterm_eval/` directory. If the original training `runs/` directory is unavailable, use `docs/results/experiment_midterm_smoothing_lr/checkpoints` as `--runs`.

```powershell
$python = 'runs/.report-venv/Scripts/python.exe'
& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py audit --runs runs --data-root data
& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py select --runs runs --data-root data
& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py evaluate --runs runs --data-root data --scope all
& $python docs/results/experiment_midterm_smoothing_lr/tools/build_experiment_results.py report --runs runs --data-root data
& $python docs/results/experiment_midterm_smoothing_lr/tools/verify_results.py --source-runs runs --results docs/results/experiment_midterm_smoothing_lr
```

## Evidence index

- [Selection lock and validation plots](selection/)
- [Cross-run CSVs and figures](comparisons/)
- [Checkpoint copy manifest](checkpoints/checkpoint_manifest.csv)
- [Run and source provenance](provenance/)
- Model reports:
  - [l_mid32_baseline](models/l_mid32_baseline/)
  - [l_mid32_smoothing005](models/l_mid32_smoothing005/)
  - [l_mid32_smoothing010](models/l_mid32_smoothing010/)
  - [l_mid32_lr005](models/l_mid32_lr005/)
  - [l_mid32_lr015](models/l_mid32_lr015/)
  - [l_mid224_baseline](models/l_mid224_baseline/)
  - [l_mid224_smoothing005](models/l_mid224_smoothing005/)
  - [l_mid224_smoothing010](models/l_mid224_smoothing010/)
  - [l_mid224_lr005](models/l_mid224_lr005/)
  - [l_mid224_lr015](models/l_mid224_lr015/)

## Provenance and limitations

The Kaggle configs record `git_revision = null` and `git_dirty = null`. Therefore, the 110 recorded source-file hashes (all matched) are the primary code-provenance evidence. The local combined manifest was not used for inference; exact Mid32/Mid224 manifests were rebuilt and matched the checkpoint hashes.

This experiment has one seed per recipe, so it cannot estimate variance or statistical significance. The test split has only 250 images and is historical; its metrics should not be treated as fresh benchmark estimates. No retraining or full-train phase was run for this dossier.
