# CIFAR-10 & CIFAR-100 DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standardized DataLoader module for CIFAR-10 and CIFAR-100 supporting automated downloading, standard preprocessing pipelines (RandomCrop with padding 4, RandomHorizontalFlip, Normalization), stratified train/validation partitioning, and comprehensive unit test verification.

**Architecture:** Create module `models/cifar_data.py` supplying dataset loaders and transformation pipelines for CIFAR-10 and CIFAR-100. Integrates canonical preprocessing (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`) with per-dataset benchmark mean/std statistics. Generates reproducible `train_loader`, `val_loader`, and `test_loader` with configurable batch sizes, worker counts, pin memory, and RNG seeds.

**Tech Stack:** Python 3.11+, PyTorch 2.x, TorchVision, PyTest.

**Specification Reference:** Requirements from `.doc/Final exam.docx` (train and test Model L on CIFAR-10 and CIFAR-100).

## Global Constraints

- Automated downloading via `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100` with `download=True`.
- Mandatory training preprocessing: `transforms.RandomCrop(32, padding=4)`, `transforms.RandomHorizontalFlip()`, `transforms.ToTensor()`, `transforms.Normalize(mean, std)`.
- Mandatory evaluation preprocessing: `transforms.ToTensor()`, `transforms.Normalize(mean, std)`.
- Canonical normalization statistics:
  - CIFAR-10: `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`
  - CIFAR-100: `mean=(0.5071, 0.4867, 0.4408)`, `std=(0.2675, 0.2565, 0.2761)`
- Guaranteed reproducibility via `torch.Generator` and `worker_init_fn` with fixed seed.
- 100% compatibility with `TickNet-L` (`cifar=True`).

## Review Focus

1. **Automated Downloader Reliability**: Data downloads cleanly to `data_root` without crashes when launched in fresh or Kaggle environments.
2. **Tensor Dimensions & Types**: Batches emitted by dataloaders must possess shape `[B, 3, 32, 32]`, dtype `torch.float32`, and labels `[B]`, dtype `torch.long`.
3. **Post-Normalization Value Distributions**: Normalized tensors possess mean $\approx 0$ and std $\approx 1$.
4. **Stratified Train/Val Partitioning**: When `val_fraction > 0`, relative class proportions are strictly preserved.
5. **No Data Augmentation on Test Set**: Test split receives tensor conversion and normalization only (no crops or flips).

---

### Task 1: Design and Implement `models/cifar_data.py`

**Files:**
- Create: `models/cifar_data.py`

- [x] **Step 1.1**: Define standard constants `CIFAR10_MEAN`, `CIFAR10_STD`, `CIFAR100_MEAN`, `CIFAR100_STD` and `get_cifar_transforms(dataset_name, augment=True)`.
- [x] **Step 1.2**: Author `build_cifar_datasets(data_root, dataset_name, val_fraction=0.0, seed=42, download=True)` managing downloads and stratified partitioning.
- [x] **Step 1.3**: Author main function `build_cifar_loaders(data_root, dataset_name, batch_size=128, val_fraction=0.0, seed=42, num_workers=2, pin_memory=True, download=True)` returning `(train_loader, val_loader, test_loader)`.

---

### Task 2: Author Comprehensive Unit Tests in `tests/test_cifar_data.py`

**Files:**
- Create: `tests/test_cifar_data.py`

- [x] **Step 2.1**: Test transform functions (shape, type, augment vs eval transforms).
- [x] **Step 2.2**: Author mock dataset tests verifying stratified split logic without requiring 150MB file downloads during unit testing.
- [x] **Step 2.3**: Author integration tests with `build_ticknet_l(num_classes, cifar=True)` verifying end-to-end forward passes.

---

### Task 3: Execute Test Suite and Confirm Integration

- [x] **Step 3.1**: Run `python3 -m pytest tests/test_cifar_data.py -v` confirming 100% pass rate.
- [x] **Step 3.2**: Run full test suite `python3 -m pytest tests -q` to guarantee zero regressions.
- [x] **Step 3.3**: Commit changes to git branch `feature/final-exam-model-l`.
