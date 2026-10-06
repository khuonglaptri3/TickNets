# Hyperparameter Grid Search System Design for the Final Examination

**Course:** Deep Learning Final Examination  
**Date:** 2026-10-05  
**Target Architecture:** `TickNet-L v1` ([`models/ticknet_l.py`](../../../models/ticknet_l.py))  
**Datasets:** CIFAR-10 (10 classes) and CIFAR-100 (100 classes)  

---

## 1. Objectives & Experimental Framework

The final examination prompt ([`.doc/Final exam.docx`](../../../.doc/Final%20exam.docx)) mandates training and testing on CIFAR benchmarks, adhering to model complexity budgets, and investigating optimizer and learning rate spaces. The team's experimental protocol specifies:
1. Train Model $L$ (`TickNet-L`) from scratch on **CIFAR-10** and **CIFAR-100**; 100% train-from-scratch is the team's experimental choice rather than an exam mandate.
2. Hardware Constraints: **Learnable Parameters $\le 6M$**, **FLOPs $< 1G$**.
3. Systematic exploration of initial Learning Rates (e.g., `0.10`, `0.15`,...) across two optimizers: **SGD** and **Adam**.
4. Comprehensive reporting of hyperparameter settings (momentum, learning rate, epochs, weight decay, loss, accuracy).

### Experimental Matrix (8 Grid Search Configurations)

| Config ID | Dataset | Optimizer | Initial Learning Rate | Momentum / Betas | Weight Decay | Scheduler (200 Epochs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cifar10_sgd_lr010` | CIFAR-10 | SGD | `0.10` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar10_sgd_lr015` | CIFAR-10 | SGD | `0.15` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar10_adam_lr0001` | CIFAR-10 | Adam | `0.001` (1e-3) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar10_adam_lr00003` | CIFAR-10 | Adam | `0.0003` (3e-4) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar100_sgd_lr010` | CIFAR-100 | SGD | `0.10` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar100_sgd_lr015` | CIFAR-100 | SGD | `0.15` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar100_adam_lr0001` | CIFAR-100 | Adam | `0.001` (1e-3) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar100_adam_lr00003` | CIFAR-100 | Adam | `0.0003` (3e-4) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |

---

## 2. Technical Architecture of Training Module `train_cifar.py`

### 2.1. Hyperparameter Management & CLI / JSON Configuration
- Supports configuration loading via JSON (`--config configs/final/<id>.json`).
- CLI parameter override: `--dataset`, `--optimizer`, `--learning-rate`, `--epochs`, `--batch-size`, `--val-fraction`, `--seed`, `--output-dir`.

### 2.2. Dataset Partitioning Policy
- Stratified hold-out split: `val_fraction=0.1` (45,000 train / 5,000 val on CIFAR-10; 450 train / 50 val per class on CIFAR-100).
- Checkpoints are selected purely based on the validation set (`best_val.pt` selected by highest `val_top1`, breaking ties with lower `val_loss`).
- The official test set (10,000 images) is never evaluated during epoch selection or hyperparameter tuning.

### 2.3. Output Artifact Tree
Each experiment output directory (`--output-dir`) emits standardized artifacts:
1. `config.json`: Full configuration snapshot, git commit hash, model complexity (parameters, FLOPs), library versions.
2. `epochs.csv`: Per-epoch metrics history (epoch, learning_rate, train_loss, train_top1, val_loss, val_top1).
3. `best_val.pt`: Checkpoint weights achieving best validation accuracy.
4. `last.pt`: Complete state checkpoint (model, optimizer, scheduler, RNG) supporting session resumption.
5. `test_metrics.json`: Unbiased final evaluation on the 10,000 test images (Top-1, loss, macro_f1).
6. `confusion_matrix.csv`: Full confusion matrix ($10 \times 10$ or $100 \times 100$).

---

## 3. Recovery & Multi-Session Kaggle Execution

- Flag `--resume <path/to/last.pt>`: Restores exact model, optimizer, scheduler, and generator state at the interrupted epoch boundary.
- Flag `--stop-after-epoch <N>`: Supports smoke testing or phased execution within restricted GPU session timeouts.
- Flag `--evaluate <checkpoint>`: Runs standalone inference on the 10,000-image test set and exports confusion matrices.
