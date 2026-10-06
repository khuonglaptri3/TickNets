# Four Experimental Directions for Midterm Model L

All experiment branches diverge from `feature/model-l-midterm-final`, commit `2bcbdb4105c5c8576829af568d6c31bb89a52c8f`:

| Branch | Verified Changes |
|---|---|
| `experiment/lv1-midterm-mixup` | Mixup regularization; preserves core TickNet-L v1 architecture |
| `experiment/lv1-midterm-cutmix` | CutMix regularization; preserves core TickNet-L v1 architecture |
| `experiment/lv1-midterm-smoothing-lr` | Label smoothing (0.05 / 0.10) and learning rate (0.05 / 0.15) evaluated in separate runs |
| `experiment/lv1-midterm-stage4-depth` | Stage 4 depth scaled from 2 to 3 blocks; baseline training recipe preserved |

Refer to `docs/experiments/BRANCH_GUIDE.md` on your assigned branch. `configs/midterm/experiment.json` serves as the primary configuration; `baseline.json` serves as the shared reference. Do not arbitrarily combine multiple modifications without isolated ablation.

---

## Environment Setup & Data Preparation

Requires Python 3.10+, with compatible PyTorch/TorchVision (profiling requires torch $\ge$ 2.5):

```bash
python -m pip install torch torchvision numpy pillow pytest scipy
```

Data layout: `DATA_ROOT/Mid32/train|test/{bird,cat,dog,frog,horse}` and similarly for `Mid224`. Both spatial resolutions must share identical file names, class mappings, and split assignments, retaining original 32 or 224 pixel sizes. When utilizing local git worktrees, point `--data-root` to the `data/` directory in the primary checkout. Never duplicate or alter the official test split.

Standard dataset breakdown: 5,000 training images and 50 test images per class. The experimental trainer partitions 500 images per class for validation: yielding 22,500 train, 2,500 validation, and 250 test samples. The `split_seed=123` is decoupled from training `seed=42`; validation membership is assigned deterministically by class/filename, maintaining cross-resolution consistency. Checksums recorded in `validation_split.json` verify sample membership.

---

## Running Baselines and Experimental Candidates

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --output-dir runs/mid32_experiment_seed42
```

Repeat for `--variant Mid224` targeting distinct output directories. JSON files supply default parameters; CLI flags override defaults. Maintain batch size 64, SGD with momentum 0.9, weight decay 1e-4, 200 epochs of Cosine Annealing, and standard crop/flip augmentations; modify strictly the single technique under study. Worker counts may be increased for Kaggle but must be matched between baseline and candidate.

Each run logs `config.json` (git revision, environment, complexity profiles), `validation_split.json`, `epochs.csv`, `last.pt`, and `best_val.pt`. Optimal checkpoints are selected based on validation Top-1, breaking ties with lower loss. Validation uses unaugmented images and hard-label cross-entropy. Training Top-1 under Mixup/CutMix represents sample-weighted accuracy across paired targets and should not be compared directly with hard-label training accuracy.

---

## Resuming Across Kaggle Sessions

Pre-specifying 200 epochs allows early session termination at epoch 100:

```bash
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid224 --data-root data --output-dir runs/mid224_experiment_seed42 --stop-after-epoch 100
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid224 --data-root data --output-dir runs/mid224_experiment_seed42 --resume runs/mid224_experiment_seed42/last.pt
```

Restore the complete run directory from the session artifact ZIP into the new session before executing resume. Retain the identical recipe, total epochs, and environment dependencies. Model, optimizer, scheduler, and RNG states (Python, NumPy, PyTorch, CUDA, DataLoader) are saved at epoch boundaries. Cosine schedules continue smoothly without resetting. `best_val.pt` governs model selection; `last.pt` facilitates resumption.

---

## Final Evaluation & Full-Data Retraining

The training pipeline does not evaluate test sets automatically. Once configurations are locked via validation metrics:

```bash
python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/mid32_experiment_seed42/best_val.pt --output-dir runs/mid32_selected_test
```

Exports Top-1 Accuracy, Loss, Macro F1, `test_predictions.csv`, and `confusion_matrix.csv`, validating mappings and architecture before loading weights.

To report figures at the prompt's 5,000 train/class scale, models can be retrained across all 25,000 images once the recipe and epoch count are locked by validation:

```bash
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --full-train --epochs 200 --output-dir runs/mid32_final_full_seed42
python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/mid32_final_full_seed42/last.pt --output-dir runs/mid32_final_full_test
```

Substitute `200` with the validated epoch count; `--full-train` operates without a validation split and emits only `last.pt`.

---

## Experimental Reporting Standards

Record branch, commit hash, dataset, seed, configuration, train/val/test sample counts, validation Top-1/loss at optimal epoch, parameter counts, GFLOPs, epoch duration, and GPU architecture. Confirm choices across seeds 42, 43, 44 under split seed 123; report mean ± standard deviation.

FLOP convention: $1\text{ MAC} = 2\text{ FLOPs}$, Conv2d/Linear only, single image in eval mode.

Run test suites before executing on GPU: `python -m pytest tests -q`.

---

## Artifact Recovery from Interrupted Sessions

Trainer revision `mid-experiment-v2` commits `last.pt` as the authoritative atomic state checkpoint for each epoch. It embeds epoch log history and a replica of best validation weights to reconstruct `epochs.csv` and `best_val.pt` should logging be interrupted. Resuming from `last.pt` re-executes uncommitted epochs; intra-batch resumption is not supported. Configuration logs capture the Git revision, dirty status, and Python source hashes; code modifications will trigger validation errors upon resuming.
