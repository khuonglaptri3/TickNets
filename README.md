# TickNets: Efficient Lightweight Networks with Full-Residual PDP Blocks for CIFAR-10 & CIFAR-100

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.15%2B-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neucom.2024.127942-blue.svg)](https://doi.org/10.1016/j.neucom.2024.127942)
[![License](<https://img.shields.io/badge/License-Academic%20Final%20Exam-green.svg>)](LICENSE)

> **Original Research Paper:** *Efficient tick-shape networks of full-residual point-depth-point blocks for image classification*
> **Authors:** Thanh Tuan Nguyen and Thanh Phuong Nguyen (*Neurocomputing*, Volume 596, 2024, Article 127942)
> **Paper DOI:** [https://doi.org/10.1016/j.neucom.2024.127942](https://doi.org/10.1016/j.neucom.2024.127942) — See [BibTeX Citation](#7-original-paper-abstract--citation)

---

## 1. Project Overview & Final Exam Results

Final Kaggle results for **TickNet-L Large v2 (14 blocks)**:
- **CIFAR-10 Test Top-1:** **96.04%** (Validation Top-1: **96.90%**, SGD + Nesterov, `lr=0.175`)
- **CIFAR-100 Test Top-1:** **81.23%** (Validation Top-1: **80.58%**, SGD + Nesterov, `lr=0.175`)

See the full technical report in [`docs/experiments/TRAINING_AND_MODEL_REPORT.md`](docs/experiments/TRAINING_AND_MODEL_REPORT.md), the [experiment analysis and figures](docs/experiments/README.md), and the [selected checkpoint archive](checkpoints/README.md).

### 1.1. Completed Final Experiments (200 Epochs, Seed 42)

| Dataset | Optimizer | Initial LR | Weight Decay | Best Val Epoch | Validation Top-1 | Test Top-1 | Test Loss | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | **SGD + Nesterov** | `0.175` | `0.0001` | 187 | **96.90%** | **96.04%** | 0.172538 | 0.960355 |
| **CIFAR-10** | Adam | `0.001` | `0.0001` | 199 | 95.66% | 95.01% | 0.202167 | 0.950145 |
| **CIFAR-100** | **SGD + Nesterov** | `0.175` | `0.0005` | 198 | **80.58%** | **81.23%** | 0.700414 | 0.812142 |
| **CIFAR-100** | Adam | `0.001` | `0.0005` | 200 | 77.20% | 78.04% | 0.827812 | 0.780109 |

### 1.2. Core Examination Requirements & Constraints

* **Model Budget Compliance:**
  * **Learnable Parameters:** Strictly capped at $\le 6,000,000$ (6M parameters).
  * **Computational Cost (FLOPs forward):** Strictly capped at $< 1,000,000,000$ (1G FLOPs per $32 \times 32$ image).
* **Datasets:** Full evaluation on **CIFAR-10** (10 classes) and **CIFAR-100** (100 classes), using a stratified split of 45,000 training images, 5,000 validation images, and 10,000 official test images.
* **Training Protocol:** Train from scratch (200 epochs, batch size 128, CosineAnnealingLR to $\eta_{\min}=0$) comparing **SGD + Nesterov** and **Adam**.

---

## 2. Architecture: TickNet-L Large v2 (14 Blocks)

The custom **TickNet-L Large v2** model ([`models/ticknet_l.py`](models/ticknet_l.py)) expands the author's original **TickNet-Basic** ([`models/TickNet.py`](models/TickNet.py)) from 5 blocks to **14 Compressed PDP blocks** across 5 stages (`[1, 2, 5, 5, 1]`) with stage channels `[160, 128, 256, 512, 768]`:

```text
Input (Cin) ──→ Linear PW 1×1 compression (hidden)
            ──→ Depthwise Conv (3×3 in Stages 1–2; split 3×3 + 5×5 in Stages 3–5)
            ──→ BN + ReLU
            ──→ PW 1×1 projection (Cout) + BN + ReLU
            ──→ SE Attention (reduction=16)
            ──→ Residual Shortcut Add (+) ──→ Output (Cout)
```

### 2.1. Complexity Profile & Budget Verification

Evaluated with [`models/model_profile.py`](models/model_profile.py) and [`checkmodel.py`](checkmodel.py) ($1 \text{ MAC} = 2 \text{ FLOPs}$, batch size = 1, shape = `(1, 3, 32, 32)`):

| Architecture | Dataset | Learnable Params | Param Budget ($\le 6\text{M}$) | Forward FLOPs | FLOPs Budget ($< 1\text{G}$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Author TickNet-Basic** | **CIFAR-10** | 1,067,348 | 17.8% | 0.1584 GFLOPs | 15.8% | **PASSED** |
| **Author TickNet-Basic** | **CIFAR-100** | 1,159,598 | 19.3% | 0.1586 GFLOPs | 15.9% | **PASSED** |
| **TickNet-L Large v2 (14 blocks)** | **CIFAR-10** | **4,453,650** | **74.2%** | **0.7886 GFLOPs** | **78.9%** | **PASSED** |
| **TickNet-L Large v2 (14 blocks)** | **CIFAR-100** | **4,545,900** | **75.8%** | **0.7888 GFLOPs** | **78.9%** | **PASSED** |

### 2.2. Stage-by-Stage Breakdown (`python checkmodel.py`)

| Stage / Submodule | Output Shape | PDP Blocks | Learnable Params (CIFAR-10) | Forward FLOPs | FLOP % |
| :--- | :--- | :---: | ---: | ---: | ---: |
| `backbone.data_bn` | `(1, 3, 32, 32)` | — | 6 | 0 | 0.00% |
| `backbone.init_conv` | `(1, 32, 32, 32)` | — | 928 | 1,769,472 | 0.22% |
| `backbone.stage1` | `(1, 160, 32, 32)` | 1 | 15,626 | 23,664,896 | 3.00% |
| `backbone.stage2` | `(1, 128, 32, 32)` | 2 | 87,128 | 167,043,072 | 21.18% |
| `backbone.stage3` | `(1, 256, 16, 16)` | 5 | 524,656 | 263,454,720 | 33.41% |
| `backbone.stage4` | `(1, 512, 8, 8)` | 5 | 2,056,928 | 259,940,352 | 32.96% |
| `backbone.stage5` | `(1, 768, 4, 4)` | 1 | 969,648 | 47,542,272 | 6.03% |
| `backbone.final_conv` | `(1, 1024, 4, 4)` | — | 788,480 | 25,165,824 | 3.19% |
| `backbone.global_pool` | `(1, 1024, 1, 1)` | — | 0 | 0 | 0.00% |
| `classifier` | `(1, 10)` / `(1, 100)` | — | 10,250 (102,500 on C100) | 20,480 (204,800 on C100) | 0.00% (0.03%) |
| **TOTAL (CIFAR-10)** | `(1, 3, 32, 32) -> (1, 10)` | **14** | **4,453,650** | **788,601,088** | **100.00%** |

---

## 3. Final Experiment Configurations & Notebooks

| Experiment | Config JSON (`configs/final/`) | Kaggle Notebook (`docs/kaggle/`) | Executed Notebook (`docs/experiments/notebooks/`) |
| :--- | :--- | :--- | :--- |
| **CIFAR-10 (SGD)** | [`cifar10_ticknet_l_large_v2_14blocks_sgd_lr0175.json`](configs/final/cifar10_ticknet_l_large_v2_14blocks_sgd_lr0175.json) | [`Phase14_CIFAR10_TickNetL_LargeV2_14Blocks.ipynb`](docs/kaggle/Phase14_CIFAR10_TickNetL_LargeV2_14Blocks.ipynb) | [`phase-14-cifar-10-v-i-ticknet-l-large-v2-14-kh-i.ipynb`](docs/experiments/notebooks/phase-14-cifar-10-v-i-ticknet-l-large-v2-14-kh-i.ipynb) |
| **CIFAR-10 (Adam)** | [`cifar10_ticknet_l_large_v2_14blocks_adam_lr0001.json`](configs/final/cifar10_ticknet_l_large_v2_14blocks_adam_lr0001.json) | [`Phase17_CIFAR10_TickNetL_LargeV2_14Blocks_Adam.ipynb`](docs/kaggle/Phase17_CIFAR10_TickNetL_LargeV2_14Blocks_Adam.ipynb) | [`ticknet-l-large-v2-14-kh-i-adam.ipynb`](docs/experiments/notebooks/ticknet-l-large-v2-14-kh-i-adam.ipynb) |
| **CIFAR-100 (SGD)** | [`cifar100_ticknet_l_large_v2_cutmix_a10_200.json`](configs/final/cifar100_ticknet_l_large_v2_cutmix_a10_200.json) | [`Phase13_CIFAR100_TickNetL_LargeV2_CutMix_A10.ipynb`](docs/kaggle/Phase13_CIFAR100_TickNetL_LargeV2_CutMix_A10.ipynb) | [`phase13-cifar100-ticknetl-largev2-cutmix-a10.ipynb`](docs/experiments/notebooks/phase13-cifar100-ticknetl-largev2-cutmix-a10.ipynb) |
| **CIFAR-100 (Adam)** | [`cifar100_ticknet_l_large_v2_cutmix_a10_adam_lr0001.json`](configs/final/cifar100_ticknet_l_large_v2_cutmix_a10_adam_lr0001.json) | [`Phase18_CIFAR100_TickNetL_LargeV2_14Blocks_Adam.ipynb`](docs/kaggle/Phase18_CIFAR100_TickNetL_LargeV2_14Blocks_Adam.ipynb) | [`phase18-cifar100-ticknetl-largev2-14blocks-adam.ipynb`](docs/experiments/notebooks/phase18-cifar100-ticknetl-largev2-14blocks-adam.ipynb) |

---

## 4. Quickstart & Command-Line Instructions

### 4.1. Inspect Model Architecture, Parameters & FLOPs

```bash
# Inspect TickNet-L (CIFAR-10 default, or CIFAR-100 with --classes 100)
python checkmodel.py
python checkmodel.py --classes 100

# Detailed layer & forward/backward gradient verification
python checkmodel_l.py
python checkmodel_l.py --num-classes 100 --details
```

### 4.2. Download CIFAR-10 & CIFAR-100

```bash
python download_cifar.py --data-root data --dataset all
```

### 4.3. Training & Evaluation

```bash
# Train CIFAR-10 (14-block TickNet-L Large v2, SGD lr=0.175 + Cutout 16)
python train_cifar10_large_v2.py --config configs/final/cifar10_ticknet_l_large_v2_14blocks_sgd_lr0175.json --data-root data --output-dir runs/cifar10_sgd

# Train CIFAR-100 (14-block TickNet-L Large v2, SGD lr=0.175 + RandAugment + CutMix)
python train_cifar_cutmix_large_v2.py --config configs/final/cifar100_ticknet_l_large_v2_cutmix_a10_200.json --data-root data --output-dir runs/cifar100_sgd

# Evaluate a saved checkpoint on the official test set
python train_cifar.py --dataset cifar10 --evaluate checkpoints/cifar10_ticknet_l_large_v2_14blocks_sgd_lr0175_cutout_200ep/best_val.pt --output-dir runs/eval_cifar10 --allow-eval-source-change
```

---

## 5. Documentation & Reports

* [`docs/experiments/TRAINING_AND_MODEL_REPORT.md`](docs/experiments/TRAINING_AND_MODEL_REPORT.md): Comprehensive training settings, optimizer comparison, data augmentation, and architectural development report.
* [`docs/experiments/README.md`](docs/experiments/README.md): Summary of the 4 completed 14-block experiments, learning curves, confusion matrices, and per-class comparisons.
* [`checkpoints/README.md`](checkpoints/README.md): Documentation for the two validation-selected final checkpoints (`CIFAR-10` and `CIFAR-100`).

---

## 6. Output Artifacts & Metrics

Each completed run directory in [`docs/experiments/checkpoints_4_phase/`](docs/experiments/checkpoints_4_phase/) and [`checkpoints/`](checkpoints/) contains:

* `config.json`: Full training configuration, hardware info, parameter count, and FLOP count.
* `epochs.csv`: Per-epoch log tracking `train_loss`, `train_top1`, `val_loss`, `val_top1`, and learning rate.
* `best_val.pt`: Model weights from the epoch with highest validation accuracy.
* `last.pt`: Resumable checkpoint at epoch 200.
* `test_metrics.json`: Unbiased test performance: Top-1 Accuracy (%), Average Loss, and Macro F1 Score.
* `confusion_matrix.csv`: Class confusion matrix ($10 \times 10$ for CIFAR-10, $100 \times 100$ for CIFAR-100).
* `test_predictions.csv`: Per-sample targets, predictions, and negative log-likelihood.
* `completion.json` & `progress.json`: SHA-256 manifest and completion status.

---

## 7. Original Paper, Abstract & Citation

This repository builds upon the foundational architecture introduced by **Thanh Tuan Nguyen** and **Thanh Phuong Nguyen** in *Neurocomputing (2024)*:

```bibtex
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  volume       = {596},
  pages        = {127942},
  year         = {2024},
  url          = {https://doi.org/10.1016/j.neucom.2024.127942}
}
```
