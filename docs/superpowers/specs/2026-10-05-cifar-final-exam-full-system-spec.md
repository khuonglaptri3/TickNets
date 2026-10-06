# Comprehensive Final Examination Technical Specification: TickNet-L & Author Baseline

**Project:** Deep Learning Final Examination  
**Date:** 2026-10-05  
**Git Branch:** `feature/final-exam-model-l`  
**Objective:** Fully specify all architectural modifications, preprocessing methodologies, optimization strategies, Kaggle modularization, critical documentation, and academic citations for model training and evaluation across CIFAR-10 & CIFAR-100.

---

## 1. Project Context & Final Examination Constraints

Examination requirements and team experimental design choices. The active specification for Kaggle execution and recovery is documented in [Reliability v2](2026-10-05-cifar-kaggle-reliability.md):
1. **Team Design Protocol — Train from Scratch 100%:** No pretrained weights loaded when initiating runs. (The DOCX prompt does not mandate this constraint). Backbone Conv layers use Kaiming Uniform; classifier weights use Xavier Normal; SE Linear layers retain PyTorch defaults.
2. **Hardware & Resource Ceilings:**
   - **Learnable Parameters:** Strictly capped at $\le 6,000,000$ (6M).
   - **Computational Complexity (Forward FLOPs):** Strictly capped at $< 1,000,000,000$ (1G FLOPs per $32 \times 32 \times 3$ image). Computational convention: $1 \text{ MAC} = 2 \text{ FLOPs}$.
3. **Benchmark Datasets:** Evaluated simultaneously on two standard benchmarks:
   - **CIFAR-10:** 10 classes, 60,000 images ($32 \times 32$).
   - **CIFAR-100:** 100 fine-grained classes, 60,000 images ($32 \times 32$).
4. **Zero Data Leakage Dataset Partitioning:**
   - Training Set: 45,000 images (90% stratified split of official training data).
   - Validation Set: 5,000 images (10% stratified split of official training data). Used for checkpoint selection (`best_val.pt`).
   - Official Test Set: 10,000 images. Evaluated strictly once upon completion on `best_val.pt`.

---

## 2. Preprocessing & Training Pipeline Enhancements

### 2.1. Regularization: Cutout Augmentation (DeVries & Taylor, 2017)
- **Source File:** [`models/cifar_data.py`](../../../models/cifar_data.py).
- **Mathematical Specification:** The `Cutout(n_holes=1, length=16)` transformation masks a $16 \times 16$ square at random coordinate $(y, x)$ on a $32 \times 32$ image, zeroing all values within the patch (corresponding to mean values post-normalization):
  $$M_{i,j} = \begin{cases} 0 & \text{if } |i - y| \le 8 \text{ and } |j - x| \le 8 \\ 1 & \text{otherwise} \end{cases}$$
- **Complete Training Transform Pipeline:**
  1. `RandomCrop(32, padding=4, padding_mode='reflect')`
  2. `RandomHorizontalFlip(p=0.5)`
  3. `ToTensor()` (converts pixel values to $[0.0, 1.0]$)
  4. `Normalize(mean, std)` (CIFAR-10: `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`; CIFAR-100: `mean=[0.5071, 0.4867, 0.4408]`, `std=[0.2675, 0.2565, 0.2761]`)
  5. `Cutout(n_holes=1, length=16)`

### 2.2. Optimizer Upgrades: Nesterov Momentum for SGD
- **Source File:** [`train_cifar.py`](../../../train_cifar.py).
- **Algorithmic Mechanics:** When configuring `sgd` with `momentum > 0`, the flag `nesterov=True` is enabled by default:
  $$v_{t} = \mu v_{t-1} + g_t$$
  $$\theta_t = \theta_{t-1} - \eta (g_t + \mu v_t)$$
  Where $\mu = 0.9$, $\eta \in \{0.10, 0.15\}$, and weight decay $\lambda = 1 \times 10^{-4}$.
- **Adam Configuration:** $\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\lambda = 1 \times 10^{-4}$ with $\eta \in \{0.001, 0.0003\}$.
- **Learning Rate Schedule:** Cosine Annealing Decay to 0 across 200 epochs:
  $$\eta_t = \frac{1}{2} \eta_{0} \left(1 + \cos\left(\frac{t \pi}{T_{\max}}\right)\right)$$

---

## 3. Experimental Configuration Matrix (10 Configurations)

Structured within [`configs/final/`](../../../configs/final) across two distinct groups:

### 3.1. Author Baseline Group (TickNet-Basic Baseline)
1. `baseline_cifar10_sgd_lr010.json`: Model `basic`, CIFAR-10, SGD (lr=0.10, momentum=0.9, nesterov=True, Cutout=16, 200 epochs).
2. `baseline_cifar100_sgd_lr010.json`: Model `basic`, CIFAR-100, SGD (lr=0.10, momentum=0.9, nesterov=True, Cutout=16, 200 epochs).

### 3.2. Proposed Model Grid Search Group (TickNet-L v1)
1. `cifar10_sgd_lr010.json`: Model `l`, CIFAR-10, SGD (lr=0.10, momentum=0.9, nesterov=True).
2. `cifar10_sgd_lr015.json`: Model `l`, CIFAR-10, SGD (lr=0.15, momentum=0.9, nesterov=True).
3. `cifar10_adam_lr0001.json`: Model `l`, CIFAR-10, Adam (lr=0.001).
4. `cifar10_adam_lr00003.json`: Model `l`, CIFAR-10, Adam (lr=0.0003).
5. `cifar100_sgd_lr010.json`: Model `l`, CIFAR-100, SGD (lr=0.10, momentum=0.9, nesterov=True).
6. `cifar100_sgd_lr015.json`: Model `l`, CIFAR-100, SGD (lr=0.15, momentum=0.9, nesterov=True).
7. `cifar100_adam_lr0001.json`: Model `l`, CIFAR-100, Adam (lr=0.001).
8. `cifar100_adam_lr00003.json`: Model `l`, CIFAR-100, Adam (lr=0.0003).

---

## 4. Kaggle 4-Phase & Baseline Notebook Architecture

To support resource-constrained execution, the experimental suite is partitioned into focused notebooks. Phases 1–4 embed audited source snapshots with multi-session recovery:

1. [`docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`](../../kaggle/Kaggle_Author_TickNet_Baseline.ipynb):
   - Executes the author's original `TickNet-Basic` on CIFAR-10 and CIFAR-100 (SGD lr=0.10).
   - Packages artifacts into `author_ticknet_baseline_results.zip`.
2. [`docs/kaggle/Phase1_CIFAR10_SGD.ipynb`](../../kaggle/Phase1_CIFAR10_SGD.ipynb):
   - Executes `cifar10_sgd_lr010` and `cifar10_sgd_lr015`.
   - Packages artifacts into `phase1_cifar10_sgd_results.zip`.
3. [`docs/kaggle/Phase2_CIFAR10_Adam.ipynb`](../../kaggle/Phase2_CIFAR10_Adam.ipynb):
   - Executes `cifar10_adam_lr0001` and `cifar10_adam_lr00003`.
   - Packages artifacts into `phase2_cifar10_adam_results.zip`.
4. [`docs/kaggle/Phase3_CIFAR100_SGD.ipynb`](../../kaggle/Phase3_CIFAR100_SGD.ipynb):
   - Executes `cifar100_sgd_lr010` and `cifar100_sgd_lr015`.
   - Packages artifacts into `phase3_cifar100_sgd_results.zip`.
5. [`docs/kaggle/Phase4_CIFAR100_Adam.ipynb`](../../kaggle/Phase4_CIFAR100_Adam.ipynb):
   - Executes `cifar100_adam_lr0001` and `cifar100_adam_lr00003`.
   - Packages artifacts into `phase4_cifar100_adam_results.zip`.

---

## 5. Critical Technical Documentation Restructuring (`docs/critical/`)

The 11 core technical guides provide full architectural traceability:
1. `COMMANDS_GUIDE.md`: Handbook of CLI execution commands for train, eval, resume, profiling, and Kaggle.
2. `MODEL_L.md`: TickNet-L v1 architecture, FLOPs/parameter complexity proofs, and budget compliance.
3. `DATALOADER.md`: CIFAR DataLoader mechanics, 90/10 stratified splitting, multi-process isolation, and DMA transfers.
4. `PREPROCESSING.md`: Cutout 16×16 mechanics (DeVries & Taylor, 2017) and color normalization constants.
5. `HYPERPARAMETER_TUNING.md`: Mathematical convergence comparison between SGD Nesterov and Adam.
6. `MODEL_INITIALIZATION.md`: Kaiming Uniform/He and Xavier initialization strategies for 100% train-from-scratch.
7. `TRAINING_AND_ARCHITECTURE.md`: Detailed comparative matrix between author baseline and TickNet-L v1.
8. `VALIDATION_AND_METRICS.md`: Mathematical formulations for Top-1, Macro-F1, Cross-Entropy Loss, and Confusion Matrices.
9. `DATASET_SPLIT.md`: Per-class sample allocation tables for CIFAR-10 and CIFAR-100.
10. `DATASET_SPLITTING.md`: Zero data leakage proofs and stratified partitioning logic.
11. `DATASET_CLEANING.md`: Cryptographic MD5 checksum verification and dataset hygiene profiles.

---

## 6. Academic Attribution & Original Author Citation

The primary `README.md` rigorously preserves academic attribution to the original research:
- **Authors:** Thanh Tuan Nguyen, Thanh Phuong Nguyen.
- **Title:** *Efficient tick-shape networks of full-residual point-depth-point blocks for image classification*.
- **Journal:** *Neurocomputing*, Volume 596, 2024, Article 127942.
- **DOI:** `10.1016/j.neucom.2024.127942`.
- **BibTeX Entry:** `@article{neucoTickNetNguyen23, ...}`.
- **Abstract:** Preserved in Section 8 of `README.md` for citation in the final examination technical report.
