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
See each `runs/*/config.json` for the complete settings and `epochs.csv` for
the 200 per-epoch records.

## Contents

- `midterm_report_assets/`: dataset-count evidence, model profile, learning
  curves, confusion matrices, classification reports, and a summary CSV.
- `runs/*/`: run configuration, all epoch logs, and final held-out test
  metrics for each resolution.

The binary `last.pt` checkpoints are deliberately not tracked in Git to keep
the repository lightweight.  They remain in the original Kaggle artifact
export and can be supplied when checkpoint-based reproduction is required.
