# 4. Dataset Splitting: Stratification Mechanics and Zero Data Leakage Guarantees

This document synthesizes the methodology, mathematical foundations, and source code evidence for the stratified splitting algorithm deployed across **CIFAR-10** and **CIFAR-100**.

---

## 1. Per-Class Stratification Methodology

### 1.1. Why Stratification is Essential
If the 50,000 training images are partitioned randomly without constraint, significant class frequency skew is virtually guaranteed. On CIFAR-100 (where each class only has 500 images in the original training split), unconstrained random sampling could leave some classes with fewer than 40 validation samples, introducing severe statistical variance into checkpoint evaluation.

### 1.2. Implementation Algorithm in Source Code
Implemented via `stratified_split_indices` in [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py):

```python
def stratified_split_indices(
    targets: List[int],
    num_classes: int,
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_to_indices[int(label)].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c in sorted(class_to_indices):
        indices = class_to_indices[c]
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices
```

### 1.3. Algorithmic Properties & Guarantees:
1. **Isolated OS-Independent Random State:** Uses a localized `random.Random(42)` instance, ensuring partitioning order is deterministic across platforms (Linux, Windows, macOS, Kaggle).
2. **Strict Class Distribution Balance:**
   - On CIFAR-10: Exactly 4,500 Train images and 500 Validation images per class.
   - On CIFAR-100: Exactly 450 Train images and 50 Validation images per class.
3. **Disjoint Index Sets (Zero Overlap):**
   $$\text{Train}_{\text{indices}} \cap \text{Val}_{\text{indices}} = \emptyset$$
   Verified by unit test `test_stratified_split_indices_proportions_and_disjoint`.

---

## 2. Core Protocol: Preventing Data Leakage (Zero Data Leakage)

1. **Strict Isolation of the Held-Out Test Set:**
   - The official 10,000-image test set is isolated from training loops, statistical parameter estimations, and validation-based model selection.
   - Normalization constants (mean and standard deviation) are standard benchmarks computed globally without target leakage.
2. **Decoupled Transformations Between Train and Val:**
   - The `TransformedSubset` wrapper in [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) holds raw underlying arrays and dynamically applies targeted transforms during batch loading:
     - Training samples receive `train_transform` (RandomCrop, Flip, Cutout).
     - Validation samples receive `eval_transform` (ToTensor and Normalize only).
   - This ensures validation accuracy reflects unbiased performance on clean, unaugmented inputs.
