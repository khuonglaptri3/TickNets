# TickNet-L v1 — Midterm results

This folder contains the reproducibility and reporting artifacts from the
full Kaggle training run of **TickNet-L v1** (`--model l`).  The source code is
the `feature/giua-ki-model-l` branch; this folder does not introduce a new
architecture or a separate training implementation.

| Variant | Train / test | Epochs | Test Top-1 | Macro F1 | Parameters | FLOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mid32 | 25,000 / 250 | 200 | 91.6% | 0.9157 | 1,096,260 | 0.157821 G |
| Mid224 | 25,000 / 250 | 200 | 95.6% | 0.9559 | 1,096,260 | 0.796760 G |

Both variants have five classes (`bird`, `cat`, `dog`, `frog`, `horse`), with
5,000 training images and 50 test images per class.  The runs use seed 42,
SGD (learning rate 0.1, momentum 0.9, weight decay 1e-4) and cosine annealing.
See the L directories under [`../training_logs/`](../training_logs/) for the
complete settings and `epochs.csv` for the 200 per-epoch records.

## Contents

- `midterm_report_assets/`: dataset-count evidence, model profile, learning
  curves, confusion matrices, classification reports, and a summary CSV.
- `../training_logs/l_mid*_seed42_20260928_103209/`: run configuration, all
  epoch logs, final checkpoints and held-out test metrics for each resolution.

The binary `last.pt` checkpoints are now stored under `../training_logs/` and
indexed by [`../checkpoints/checkpoint_manifest.csv`](../checkpoints/checkpoint_manifest.csv).
See the [2026-10-01 audit](../AUDIT_TICKNET_2026-10-01.md) for architecture,
split-membership verification and limitations of the experimental comparison.
