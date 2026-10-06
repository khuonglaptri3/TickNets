# Implementation Plan: Final Examination System Evolution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comprehensively evolve the training, testing, data analysis, modular Kaggle execution, and technical documentation infrastructure for `TickNet-L v1` and the baseline `TickNet-Basic` on CIFAR-10 & CIFAR-100, while preserving academic attribution and citations to the original authors.

**Architecture:** Expand preprocessing with Cutout ($16\times16$ following DeVries & Taylor 2017), enable Nesterov momentum for SGD in `train_cifar.py`, configure the 10-experiment matrix, partition Kaggle execution into 5 standalone notebooks (1 Baseline + 4 Phases), author 11 in-depth technical documents in `docs/critical/`, overhaul `README.md` with complete CLI execution guidelines, and restore the BibTeX/Abstract citation of the original Neurocomputing 2024 paper.

**Tech Stack:** Python 3.11, PyTorch 2.0+, TorchVision, NumPy, Pandas, PyTest, Jupyter Notebook, GitHub Flavored Markdown.

**Specification Reference:** [`docs/superpowers/specs/2026-10-05-cifar-final-exam-full-system-spec.md`](../specs/2026-10-05-cifar-final-exam-full-system-spec.md)

## Global Constraints

- Parameter Constraint: $\le 6,000,000$ params (TickNet-L v1 uses 1,100,105 on CIFAR-10 and 1,169,315 on CIFAR-100).
- FLOP Constraint: $< 1,000,000,000$ FLOPs (TickNet-L v1 uses ~0.158 GFLOPs with $1\text{ MAC} = 2\text{ FLOPs}$).
- 100% Train from Scratch (Kaiming Uniform/Normal; no pretrained weights loaded).
- 90% train / 10% validation stratified partitioning with zero data leakage.
- Strict preservation of academic citations for authors Thanh Tuan Nguyen & Thanh Phuong Nguyen (Neurocomputing 2024).

## Review Focus

1. Verify isolation and tensor shape invariance of `Cutout(16x16)`.
2. Confirm `nesterov=True` activates strictly when `optimizer == 'sgd'` and `momentum > 0`.
3. Verify feasibility of 4 standalone Kaggle phases avoiding 12h GPU timeouts.
4. Verify consistency across 10 JSON configuration files and documentation descriptions.
5. Ensure 100% test pass rate across the automated test suite.

---

### Task 1: Integrate Cutout Regularization (DeVries & Taylor, 2017) into `models/cifar_data.py`

**Files:**
- Modify: `models/cifar_data.py`
- Test: `tests/test_cifar_data.py`

- [x] **Step 1.1**: Define `Cutout(n_holes=1, length=16)` generating random zero masks over $16\times16$ patches on $3\times32\times32$ image tensors.
- [x] **Step 1.2**: Integrate `use_cutout=True` and `cutout_length=16` parameters into `build_cifar_transforms` for CIFAR-10 and CIFAR-100 training pipelines.
- [x] **Step 1.3**: Add `cutout` configuration flags to `get_cifar_dataloaders`.
- [x] **Step 1.4**: Author unit tests verifying `Cutout` mask dimensions and tensor shape invariance.

---

### Task 2: Enable Nesterov Momentum for SGD in `train_cifar.py`

**Files:**
- Modify: `train_cifar.py`
- Test: `tests/test_train_cifar.py`

- [x] **Step 2.1**: Update `build_optimizer` in `train_cifar.py` to enable `nesterov=(args.momentum > 0)` when `optimizer == 'sgd'`.
- [x] **Step 2.2**: Log `nesterov` state into `config.json` per run.
- [x] **Step 2.3**: Author unit tests verifying `torch.optim.SGD` instances exhibit `param_groups[0]['nesterov'] == True`.

---

### Task 3: Standardize 10 Final Examination Configurations in `configs/final/*.json`

**Files:**
- Create/Modify: `configs/final/*.json`

- [x] **Step 3.1**: Update 8 Grid Search configuration files for `TickNet-L` (`cifar10_sgd_lr010.json`, `cifar10_sgd_lr015.json`, `cifar10_adam_lr0001.json`, `cifar10_adam_lr00003.json`, `cifar100_sgd_lr010.json`, `cifar100_sgd_lr015.json`, `cifar100_adam_lr0001.json`, `cifar100_adam_lr00003.json`) adding `"cutout": true, "cutout_length": 16`.
- [x] **Step 3.2**: Create 2 baseline configuration files for the author's model: `configs/final/baseline_cifar10_sgd_lr010.json` and `configs/final/baseline_cifar100_sgd_lr010.json` with `"model": "basic"`, `"optimizer": "sgd"`, `"learning_rate": 0.10`, `"momentum": 0.9`, `"weight_decay": 0.0001`, `"cutout": true`.

---

### Task 4: Modularize Kaggle Notebooks (Author Baseline + 4 Standalone Phases)

**Files:**
- Delete: `docs/kaggle/baseline_Basic/`
- Create: `docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`
- Create: `docs/kaggle/Phase1_CIFAR10_SGD.ipynb`
- Create: `docs/kaggle/Phase2_CIFAR10_Adam.ipynb`
- Create: `docs/kaggle/Phase3_CIFAR100_SGD.ipynb`
- Create: `docs/kaggle/Phase4_CIFAR100_Adam.ipynb`

- [x] **Step 4.1**: Remove legacy midterm folder `docs/kaggle/baseline_Basic/`.
- [x] **Step 4.2**: Build notebook `Kaggle_Author_TickNet_Baseline.ipynb` training `TickNet-Basic` on CIFAR-10 and CIFAR-100 with SGD Nesterov lr=0.10, Cutout 16, auto-packaging results into `author_ticknet_baseline_results.zip`.
- [x] **Step 4.3**: Partition the grid search matrix into 4 standalone phase notebooks:
  - `Phase1_CIFAR10_SGD.ipynb`: 2 CIFAR-10 runs (SGD lr=0.10, lr=0.15) $\to$ `phase1_cifar10_sgd_results.zip`.
  - `Phase2_CIFAR10_Adam.ipynb`: 2 CIFAR-10 runs (Adam lr=0.001, lr=0.0003) $\to$ `phase2_cifar10_adam_results.zip`.
  - `Phase3_CIFAR100_SGD.ipynb`: 2 CIFAR-100 runs (SGD lr=0.10, lr=0.15) $\to$ `phase3_cifar100_sgd_results.zip`.
  - `Phase4_CIFAR100_Adam.ipynb`: 2 CIFAR-100 runs (Adam lr=0.001, lr=0.0003) $\to$ `phase4_cifar100_adam_results.zip`.

---

### Task 5: Rewrite and Synchronize 11 Core Technical Documents in `docs/critical/*.md`

**Files:**
- Overwrite: `docs/critical/*.md` (11 files)

- [x] **Step 5.1**: Synchronize `COMMANDS_GUIDE.md` with CLI flags for train, eval, resume, profiling, and Kaggle phases.
- [x] **Step 5.2**: Synchronize `MODEL_L.md` with 7-block architecture, channel elasticity `[112, 64, 144, 288, 512]`, and budget proofs ($\le 6$M params, $< 1$G FLOPs).
- [x] **Step 5.3**: Synchronize `DATALOADER.md` with 90/10 stratified splitting on 45,000 train / 5,000 val for CIFAR.
- [x] **Step 5.4**: Synchronize `PREPROCESSING.md` with mathematical specifications for Cutout 16×16 and dataset normalization constants.
- [x] **Step 5.5**: Synchronize `HYPERPARAMETER_TUNING.md` with comparative analysis of SGD Nesterov vs Adam and Cosine Annealing.
- [x] **Step 5.6**: Synchronize `MODEL_INITIALIZATION.md` with Kaiming Uniform/He and Xavier initialization for 100% train-from-scratch.
- [x] **Step 5.7**: Synchronize `TRAINING_AND_ARCHITECTURE.md` with head-to-head architectural comparison between TickNet-L v1 and TickNet-Basic.
- [x] **Step 5.8**: Synchronize `VALIDATION_AND_METRICS.md` with formulas for Top-1, Loss, Macro-F1, and Confusion Matrix.
- [x] **Step 5.9**: Synchronize `DATASET_SPLIT.md` and `DATASET_SPLITTING.md` with detailed sample allocations and zero leakage proofs.
- [x] **Step 5.10**: Synchronize `DATASET_CLEANING.md` with MD5 checksum verification and dataset hygiene profiles.

---

### Task 6: Overhaul `README.md` & Restore Original Author Citations

**Files:**
- Overwrite: `README.md`

- [x] **Step 6.1**: Overhaul `README.md` structure: examination overview, architectural methodology, Cutout training, Grid Search matrix, CLI & Kaggle guides, output artifact trees, and document sitemap.
- [x] **Step 6.2**: Restore academic citation block for the original research:
  - DOI badge `10.1016/j.neucom.2024.127942` and paper attribution banner.
  - Dedicated section `## 8. Original Paper, Abstract & Citation` preserving paper title, verbatim abstract, and standard BibTeX entry:
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

---

### Task 7: Expand Automated Test Suite

**Files:**
- Update: `tests/test_cifar_data.py`
- Update: `tests/test_train_cifar.py`

- [x] **Step 7.1**: Test Cutout pipeline, stratified splitting, and normalization.
- [x] **Step 7.2**: Test FLOP and parameter profiling for `basic` and `l` across CIFAR-10 and CIFAR-100.
- [x] **Step 7.3**: Test 1-epoch smoke training for SGD Nesterov and Adam, verifying output artifact integrity.
- [x] **Step 7.4**: Execute complete test suite `PYTHONPATH=. pytest -v` $\to$ **Confirmed all tests passed (100%)**.

---

### Task 8: Git Branch Management & Remote Synchronization

- [x] **Step 8.1**: Commit changes to branch `feature/final-exam-model-l`.
- [x] **Step 8.2**: Push to remote repository `origin/feature/final-exam-model-l` and confirm latest commit head.
