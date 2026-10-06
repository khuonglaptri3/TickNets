# CIFAR-10 & CIFAR-100 Hyperparameter Grid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete training and hyperparameter grid search infrastructure for `TickNet-L` on both CIFAR-10 and CIFAR-100, investigating SGD and Adam optimizers across learning rate bands, supporting local workstations and Kaggle GPU execution.

**Architecture:** Create script `train_cifar.py` adhering to TickNets engineering standards (seed management, per-epoch logging, validation-based `best_val.pt` selection, resumable `last.pt` checkpoints, model complexity profiling). Generate 8 JSON configuration files for the 8 experiments. Build unit tests and Kaggle execution notebooks.

**Tech Stack:** Python 3.11, PyTorch, TorchVision, PyTest.

**Specification Reference:** `docs/superpowers/specs/2026-10-05-cifar-grid-search-design.md`

## Global Constraints
- Architecture: `TickNet-L v1` (`cifar=True`) across both datasets.
- Parameter Budget: $\le 6,000,000$ params (1,100,105 on CIFAR-10, 1,169,315 on CIFAR-100).
- FLOP Budget: $< 1,000,000,000$ FLOPs (~0.158 GFLOPs).
- Optimizers: Full support for `sgd` and `adam`.
- Scheduler: `CosineAnnealingLR` over 200 epochs.
- Partitioning: `val_fraction=0.1` stratified split, validation checkpoint selection, held-out test evaluation.

---

### Task 1: Build Training Module `train_cifar.py`

**Files:**
- Create: `train_cifar.py`

- [x] **Step 1.1**: Author `parse_args` supporting both CLI arguments and JSON configurations.
- [x] **Step 1.2**: Author `run_epoch` computing loss, Top-1 accuracy, handling train and eval modes.
- [x] **Step 1.3**: Implement artifact persistence: `epochs.csv`, `best_val.pt`, `last.pt`, `test_metrics.json`, `confusion_matrix.csv`, `config.json`.
- [x] **Step 1.4**: Implement `evaluate` routine and checkpoint resumption.

---

### Task 2: Create Configuration Files in `configs/final/*.json`

**Files:**
- Create: `configs/final/cifar10_sgd_lr010.json`
- Create: `configs/final/cifar10_sgd_lr015.json`
- Create: `configs/final/cifar10_adam_lr0001.json`
- Create: `configs/final/cifar10_adam_lr00003.json`
- Create: `configs/final/cifar100_sgd_lr010.json`
- Create: `configs/final/cifar100_sgd_lr015.json`
- Create: `configs/final/cifar100_adam_lr0001.json`
- Create: `configs/final/cifar100_adam_lr00003.json`

- [x] **Step 2.1**: Create 4 configurations for CIFAR-10 (SGD lr 0.10, 0.15; Adam lr 1e-3, 3e-4).
- [x] **Step 2.2**: Create 4 configurations for CIFAR-100 (SGD lr 0.10, 0.15; Adam lr 1e-3, 3e-4).

---

### Task 3: Author Unit Tests in `tests/test_train_cifar.py`

**Files:**
- Create: `tests/test_train_cifar.py`

- [x] **Step 3.1**: Test optimizer instantiation (SGD vs Adam with momentum, betas, weight decay).
- [x] **Step 3.2**: Smoke test 1 training epoch on small dummy datasets for CIFAR-10 and CIFAR-100.
- [x] **Step 3.3**: Test `--evaluate` mode verifying export of standard $10 \times 10$ and $100 \times 100$ confusion matrices.

---

### Task 4: Author Kaggle Execution Notebooks

**Files:**
- Create: `docs/kaggle/Kaggle_Final_Exam_Grid_Search.ipynb`

- [x] **Step 4.1**: Build standalone GPU notebook for Kaggle execution on Tesla T4/P100 hardware.
- [x] **Step 4.2**: Integrate automated execution of 8 configurations saving to `/kaggle/working/runs/`.

---

### Task 5: Run Verification, Confirm, and Commit

- [x] **Step 5.1**: Execute `python3 -m pytest tests/test_train_cifar.py -v`.
- [x] **Step 5.2**: Execute complete test suite `python3 -m pytest tests -q`.
- [x] **Step 5.3**: Commit changes to git branch `feature/final-exam-model-l`.
