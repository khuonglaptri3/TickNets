# 6. DataLoader: CIFAR-10 & CIFAR-100 (Batch / Shuffle / Workers / Pin_Memory / Cutout)

This document details the architectural design, technical mechanics, and source code evidence for the **DataLoader** module supporting model training on the standard benchmark datasets **CIFAR-10** and **CIFAR-100**.

---

## 1. Final Examination DataLoader Configuration Summary

| Setting | Training Set | Validation Set | Test Set | Design Rationale & Code Reference |
| :--- | :--- | :--- | :--- | :--- |
| **Sample Count** | **45,000 images** (90% of training split) | **5,000 images** (10% stratified) | **10,000 images** (Official test set) | Zero data leakage; official test set remains completely pristine |
| **Batch Size** | 128 (configurable via CLI) | 128 (identical size) | 128 (identical size) | Balances GPU memory throughput on NVIDIA T4/P100 hardware |
| **Drop Last** | `drop_last=False` | `drop_last=False` | `drop_last=False` | No dropped data; trailing remainder batches are sample-weighted in `run_epoch` |
| **Shuffle** | `shuffle=True` (with `Generator(seed)`) | `shuffle=False` | `shuffle=False` | Shuffling prevents order overfitting; deterministic order preserves evaluation stability |
| **Data Augmentation** | **RandomCrop + Flip + Cutout** | Not applied (Normalize only) | Not applied (Normalize only) | Regularizes training to combat overfitting and improve generalization |
| **Num Workers** | 2 processes (with `seed_worker`) | 2 processes (with `seed_worker`) | 2 processes | Multi-process data loading; `seed_worker` guarantees 100% reproducibility |
| **Pin Memory** | Enabled automatically on CUDA | Enabled automatically on CUDA | Enabled automatically on CUDA | Accelerates direct DMA page-locked transfer from host RAM to GPU VRAM |

---

## 2. Technical Mechanics & Source Code Evidence

All DataLoader creation logic is centralized in [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) and integrated into the training pipeline via [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py).

### 2.1. Stratified Validation Split
Implemented in [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) within `stratified_split_indices`:
- The 50,000 training images of CIFAR are split into a $9 : 1$ ratio ($45,000$ train and $5,000$ validation).
- Each class in CIFAR-10 contains exactly $4,500$ training images and $500$ validation images.
- Each class in CIFAR-100 contains exactly $450$ training images and $50$ validation images.
- The validation set serves as the evaluation signal for saving `best_val.pt`. The 10,000-image test set is strictly evaluated only once upon experiment completion.

### 2.2. Transformation Sequence & Cutout Regularization (DeVries & Taylor, 2017)
Implemented in [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) within `get_cifar_transforms`:
```python
class Cutout(object):
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        mask_tensor = torch.from_numpy(mask).to(dtype=img.dtype, device=img.device).expand_as(img)
        return img * mask_tensor
```

Pipeline for **Training**:
1. `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`
2. `transforms.RandomHorizontalFlip(p=0.5)`
3. `transforms.ToTensor()`
4. `transforms.Normalize(mean=mean, std=std)` (Standard dataset constants)
5. `Cutout(n_holes=1, length=16)` (Randomly masks one $16 \times 16$ pixel square)

Pipeline for **Validation & Testing**:
1. `transforms.ToTensor()`
2. `transforms.Normalize(mean=mean, std=std)`

### 2.3. Per-Channel Dataset Statistics
- **CIFAR-10:**
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Std: `(0.2470, 0.2435, 0.2616)`
- **CIFAR-100:**
  - Mean: `(0.5071, 0.4867, 0.4408)`
  - Std: `(0.2675, 0.2565, 0.2761)`

### 2.4. Sample-Weighted Metric Aggregation
Because `drop_last=False`, the final batch may contain fewer samples than `batch_size`. The `run_epoch` function in [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py) correctly computes sample-weighted loss and accuracy:
```python
loss_sum += loss.item() * labels.numel()
correct += (preds == labels).sum().item()
total += labels.numel()
...
loss_avg = loss_sum / total
top1_acc = 100.0 * correct / total
```
This entirely eliminates statistical distortion that occurs when computing unweighted arithmetic averages across non-uniform batches.

---

## 3. Verification Evidence
The DataLoader system is tested by the automated test suite in [`tests/test_cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/tests/test_cifar_data.py):
- `test_normalize_dataset_name`: Verifies dataset name normalization.
- `test_get_cifar_transforms_shape_and_type`: Confirms output is always a `(3, 32, 32)` float32 tensor.
- `test_cutout_transform`: Validates accurate $16 \times 16$ mask zeroing.
- `test_stratified_split_indices_proportions_and_disjoint`: Ensures zero index overlap between train and validation splits.
- `test_cifar_batches_forward_pass_ticknet_l`: Verifies end-to-end forward pass compatibility with `TickNet-L`.
