# Guide to Midterm TickNet-L Experiments

## 1. Branch L: Label Smoothing and Learning Rate

This branch preserves the core TickNet-L architecture (`ticknet-l-v1`, 5 classes). It only investigates training loss modifications or initial learning rates; validation and testing strictly evaluate standard CrossEntropyLoss with hard labels (smoothing = 0). The parameter `--label-smoothing` defaults to 0 and accepts finite values strictly in `0 <= value < 1`. CLI values override JSON configurations.

## Experimental Formats & Recipes

All recipes use batch size 64, 200 epochs, SGD, momentum 0.9, weight decay 1e-4, training seed 42, split seed 123, validation fraction 0.1, default augmentation, and no Mixup/CutMix. Scheduler is CosineAnnealingLR with `T_max=200`; LR values denote initial learning rates.

| Config | Label Smoothing | Initial LR | Comparison with Baseline |
| --- | ---: | ---: | --- |
| `configs/midterm/baseline.json` | 0 (CLI default) | 0.1 | Reference Baseline |
| `configs/midterm/experiment.json` | 0.05 | 0.1 | Branch default recipe |
| `configs/midterm/smoothing_005.json` | 0.05 | 0.1 | Smoothing variant only |
| `configs/midterm/smoothing_010.json` | 0.10 | 0.1 | Smoothing variant only |
| `configs/midterm/lr_005.json` | 0 | 0.05 | Learning rate variant only |
| `configs/midterm/lr_015.json` | 0 | 0.15 | Learning rate variant only |

`experiment.json` and `smoothing_005.json` represent identical configurations. The two LR configurations enforce zero label smoothing to isolate the impact of learning rate. Avoid confounding multi-factor conclusions from combined runs.

## Execution on Windows (PowerShell)

Open PowerShell at the worktree root. Prepared data must follow the layout `Mid32/train`, `Mid32/test`, `Mid224/train`, `Mid224/test` with 5 balanced classes (`bird`, `cat`, `dog`, `frog`, `horse`); sharing identical manifests and splits across all recipes. Modify `$dataRoot` to your data path. Each execution requires a distinct output directory; the trainer automatically selects CUDA when available.

```powershell
$pythonExe = 'python'
$dataRoot = 'data'
```

Commands for Mid32:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/baseline.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_baseline
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoothing005
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_010.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoothing010
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_005.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_lr005
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_015.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_lr015
```

Commands for Mid224:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/baseline.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_baseline
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_smoothing005
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_010.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_smoothing010
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_005.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_lr005
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_015.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_lr015
```

Using branch default recipe instead of `smoothing_005.json`:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_experiment
```

To run baseline using branch default config, add `--label-smoothing 0`. To isolate LR, use `lr_005.json` or `lr_015.json`; do not merely override LR on `experiment.json` because that configuration retains 0.05 smoothing.

## Execution on Kaggle

Within the repository directory in Kaggle, invoke the environment Python and configure data/output paths accordingly:

```python
!python train_mid_experiment.py --config configs/midterm/experiment.json --data-root /kaggle/input/midterm-prepared --variant Mid32 --output-dir /kaggle/working/l_mid32_smoothing005
```

Execute each configuration into a separate output directory. Maintain identical `--variant`, manifest, seed, split seed, and total epoch count across comparisons.

## Single-Epoch Smoke Testing & Resumption

`--stop-after-epoch 1` halts execution at the epoch boundary while preserving the 200-epoch cosine annealing decay schedule, ideal for preflight validation:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoke --stop-after-epoch 1
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoke --resume runs/l_mid32_smoke/last.pt
```

Resumption requires `last.pt`, the prior run directory with `epochs.csv`, and identical configurations (including smoothing, total epochs, augmentations, and split). Altering smoothing from 0.05 to 0 or 0.1 is rejected before writing results. Checkpoints serialize model, optimizer, scheduler, RNG, and DataLoader generator states. `--epochs 1` also executes a single epoch but establishes a different cosine decay trajectory, suitable only for standalone unit tests.

## Model Selection and Reporting

Validation partitions are isolated using split seed 123; the test set never influences configuration selection or checkpoint choices. Check `validation_sha256` and `membership_sha256` in `config.json` to verify sample alignment across runs. Checkpoint `best_val.pt` is selected by highest validation Top-1 accuracy, breaking ties with lower hard-label validation loss. `last.pt` is reserved for resumption. `--full-train` is prohibited during comparative rounds as it bypasses validation.

Compare recipes against baselines separately for Mid32 and Mid224. Record validation Top-1, validation loss, selected epoch, recipe, seed, and split hash. When testing stability across seeds, re-evaluate baselines and candidates with identical seeds under split seed 123. Avoid direct train loss comparisons across smoothing levels due to objective disparities; hard-label validation loss remains directly comparable.

Following validation-based recipe selection, evaluate the held-out test set once using the selected checkpoint:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid32 --evaluate runs/l_mid32_smoothing005/best_val.pt --output-dir runs/l_mid32_smoothing005_test
```

Evaluation enforces hard-label loss even if the configuration specifies smoothing. Emits `test_metrics.json`, `test_predictions.csv`, and `confusion_matrix.csv`. Never retroactively tune hyperparameters based on test scores.

## Hypotheses Under Evaluation

Label smoothing (0.05 or 0.10) can temper overconfidence and improve validation metrics; excessive smoothing may impair class discrimination. For 5 classes, the target distribution is $(1 - \epsilon) \cdot \text{one\_hot} + \epsilon / 5$.

LR 0.05 may yield more stable updates at the cost of slower convergence; LR 0.15 may accelerate early progress but exhibit heightened initial variance. These represent empirical hypotheses to be validated through full 200-epoch runs.

## Testing Suite

```powershell
$env:TORCH_HOME = 'runs/torch-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $pythonExe -m pytest tests/test_mid_smoothing.py -p no:cacheprovider
```

Verifies loss/gradient computations against analytical formulas and PyTorch references, baseline defaults, CLI/JSON ranges, CLI overrides, full single-epoch execution, hard-label validation/test metrics, resumption rejection upon smoothing mismatches, and bit-for-bit resumption continuity.

The notebook `docs/kaggle/experiments/Kaggle_Midterm_Experiment.ipynb` executes baseline and smoothing 0.05; refer to the [General Protocol](MIDTERM_PROTOCOL.md) for complete guidelines.

---

## 2. TickNet-L Stage 4 Depth Scaling Experiment

The branch `experiment/lv1-midterm-stage4-depth` derives from shared commit `2256a1d3`. It evaluates the empirical impact of appending an extra residual block in Stage 4 under identical training recipes.

## Baseline and Variant Specifications

Baseline uses `configs/midterm/baseline.json`, model `l`, revision `ticknet-l-v1`. The variant uses `configs/midterm/experiment.json`, model `l_stage4`, revision `ticknet-l-v1-stage4-depth3`. The two configurations differ solely in `model`.
The specialized model inherits from L v1 and appends `backbone.stage4.unit3`: a CompressedPDPBlock with 288 → 216 → 288 channels, parallel 3×3 and 5×5 depthwise kernels, stride 1, and identity shortcut. Stage depths shift from `(1, 1, 2, 2, 1)` to `(1, 1, 2, 3, 1)`. Conv2d layers use standard L v1 initialization. Instantiated via `build_ticknet_l_stage4(num_classes=5, *, cifar=False)`.

| Model | Block Count | Learnable Params (5 classes) | FLOPs Mid32 | FLOPs Mid224 |
| --- | ---: | ---: | ---: | ---: |
| L v1 | 7 | 1,096,260 | 157,820,864 | 796,760,240 |
| L Stage4-depth3 | 8 | 1,236,030 | 174,236,864 | 846,991,472 |
| Delta | +1 | +139,770 | +16,416,000 | +50,231,232 |

Examination Budgets: $\le 6,000,000$ parameters and **$< 1,000,000,000$ FLOPs**. Both configurations strictly satisfy constraints. FLOPs follow batch 1, eval mode, Conv2d/Linear only ($1\text{ MAC} = 2\text{ FLOPs}$), cross-verified with `torch.utils.flop_counter.FlopCounterMode`.

## Controlled Comparative Execution

Execute from the worktree root using ImageFolder datasets at `data/Mid32` and `data/Mid224`:

```powershell
$python = 'python'
$env:TORCH_HOME = 'runs/torch-cache'
& $python train_mid_experiment.py --config configs/midterm/baseline.json --data-root data --variant Mid32 --output-dir runs/stage4/baseline-Mid32
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid32 --output-dir runs/stage4/depth3-Mid32
& $python train_mid_experiment.py --config configs/midterm/baseline.json --data-root data --variant Mid224 --output-dir runs/stage4/baseline-Mid224
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid224 --output-dir runs/stage4/depth3-Mid224
```

Both models share training seed 42, split seed 123, 10% stratified validation, batch size 64, 200 epochs, SGD LR 0.1, momentum 0.9, weight decay 1e-4, and CosineAnnealingLR without mixing or label smoothing. Cross-check `validation_sha256` and `membership_sha256` in `config.json` to verify dataset alignment.

## Checkpoints and Evaluation

`epochs.csv` tracks train/validation metrics; `best_val.pt` is selected via validation Top-1, breaking ties with lower loss. `last.pt` serializes optimizer/scheduler/RNG state for seamless resumption:

```powershell
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid32 --output-dir runs/stage4/depth3-Mid32 --resume runs/stage4/depth3-Mid32/last.pt
```

Evaluate held-out test sets only after fixing configurations:

```powershell
& $python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/stage4/baseline-Mid32/best_val.pt --output-dir runs/stage4/test-baseline-Mid32
& $python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/stage4/depth3-Mid32/best_val.pt --output-dir runs/stage4/test-depth3-Mid32
```

Outputs include `test_metrics.json` (Top-1, loss, macro-F1), `test_predictions.csv`, and `confusion_matrix.csv`.

## Verification Suite

```powershell
& $python -m pytest -p no:cacheprovider tests/test_ticknet_l_stage4.py -q
```

Verifies baseline L, 8 blocks of the variant, initialization, parameter/FLOP counts, finite gradients across 32/224, revision rejection on resume/eval, and end-to-end execution.

---

## 3. TickNet-L v1 + Mixup Regularization

The branch `experiment/lv1-midterm-mixup` preserves L v1 (1,096,260 parameters), modifying batch-level training augmentation.

Mixup interpolates pairs of images and labels using mixing ratio $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha=0.2$. Loss is computed as $\lambda \cdot \text{CE}(y_A) + (1 - \lambda) \cdot \text{CE}(y_B)$.

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --mixing mixup --variant Mid32 --data-root data --output-dir runs/mid32_mixup_seed42
```

---

## 4. TickNet-L v1 + CutMix Regularization

The branch `experiment/lv1-midterm-cutmix` preserves L v1 (1,096,260 parameters), modifying batch-level training augmentation.

CutMix replaces rectangular image patches with patches from alternative samples. Sets $\alpha=1.0$, applied with probability 0.5; $\lambda$ is recalculated based on clipped patch bounding boxes.

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --mixing cutmix --variant Mid32 --data-root data --output-dir runs/mid32_cutmix_seed42
```

Configure variant to Mid224 with separate output directories. Select configurations based on validation Top-1/loss, evaluating final models with `--evaluate`. Training Top-1 accuracy under mixing represents sample-weighted target metrics and should not be compared directly with hard-label training accuracy.

Refer to the [General Protocol](MIDTERM_PROTOCOL.md) for full guidelines on multi-seed evaluations and full-train execution.
