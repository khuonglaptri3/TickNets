# 5. Preprocessing: CIFAR-10 & CIFAR-100 (Resize / Normalize / Data Augmentation & Cutout)

This document details the architectural design, technical mechanics, and source code evidence for **Preprocessing** and **Data Augmentation** on the **CIFAR-10** and **CIFAR-100** benchmark datasets.

---

## 1. Final Examination Data Preprocessing Summary

| Category | Training Set | Validation & Test Sets | Design Rationale & Code Reference |
| :--- | :--- | :--- | :--- |
| **Image Resolution** | Native $32 \times 32$ | Native $32 \times 32$ | Native CIFAR resolution is $32 \times 32$; preserves 100% original pixel data without distortion or artificial upsampling |
| **Data Augmentation 1** | **RandomCrop(32, pad=4, reflect)** | Not applied | 4-pixel reflection padding followed by random $32 \times 32$ crop, providing spatial translation invariance |
| **Data Augmentation 2** | **RandomHorizontalFlip(p=0.5)** | Not applied | Randomly flips 50% of images horizontally, doubling symmetric pose diversity |
| **Data Augmentation 3** | **Cutout(1 hole, length=16)** | Not applied | Randomly masks a $16 \times 16$ pixel patch (DeVries & Taylor, 2017), forcing the network to attend to global contextual cues |
| **Tensor Conversion** | `transforms.ToTensor()` | `transforms.ToTensor()` | Converts pixel values from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32 tensors |
| **Normalization** | `transforms.Normalize(mean, std)` | `transforms.Normalize(mean, std)` | Normalizes using official per-channel dataset statistics for CIFAR-10 vs CIFAR-100 |
| **Reproducibility** | Fixed Generator + `seed_worker` | Independent, deterministic | Ensures bit-for-bit reproducible training and evaluation runs |

---

## 2. Technical Details & Source Code Evidence ([`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py))

### 2.1. Standard Per-Channel Dataset Statistics
Each dataset features distinctive illumination and color distributions:
```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
```
Standard normalization centers the input distribution around mean $\approx 0$ and variance $\approx 1$, preventing gradient vanishing or explosion during early backpropagation passes.

### 2.2. Cutout Regularization (DeVries & Taylor, 2017)
Implemented in the `Cutout` class within [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py):
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
* **Operational Mechanism:** For each normalized training image, selects a random center coordinate $(x, y)$ and masks a square patch of $16 \times 16$ pixels to zero (corresponding to the mean normalized pixel value).
* **Empirical Benefit:** Prevents the network from overfitting to isolated minor visual cues (e.g., memorizing only a bird's beak instead of learning holistic wing and body morphology).

### 2.3. Unified Transformation Pipeline (`get_cifar_transforms`)
```python
def get_cifar_transforms(dataset_name: str, *, augment: bool = True, cutout: bool = True, cutout_length: int = 16) -> transforms.Compose:
    canon_name = normalize_dataset_name(dataset_name)
    mean, std = CIFAR_STATS[canon_name]

    if augment:
        tf_list = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if cutout and cutout_length > 0:
            tf_list.append(Cutout(n_holes=1, length=cutout_length))
        return transforms.Compose(tf_list)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
```

---

## 3. Verification Evidence
The preprocessing pipeline is verified via comprehensive unit tests:
1. `tests/test_cifar_data.py::test_get_cifar_transforms_shape_and_type`: Confirms transformed outputs maintain tensor dimensions `(3, 32, 32)`, data type `torch.float32`, and appropriate normalized values outside $[0, 1]$.
2. `tests/test_cifar_data.py::test_cutout_transform`: Confirms the $16 \times 16$ pixel region is masked to 0.0 while preserving unmasked regions intact.
