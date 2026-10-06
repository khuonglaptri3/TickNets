# CIFAR-10 & CIFAR-100: Dataset Specification and Train / Val / Test Partitioning

This document records the dataset structure, official University of Toronto distribution sources, MD5 verification checksums, and the partitioning methodology for **Train (45,000) / Validation (5,000) / Test (10,000)** splits in the final examination project.

---

## 1. Data Sources & Integrity Standards

Both benchmark datasets are downloaded and verified automatically via [`scripts/download_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/scripts/download_cifar.py):
1. **CIFAR-10 (`cifar-10-python.tar.gz`):**
   - Primary URL: `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`
   - Fallback URL: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
   - MD5 Checksum: `c58f30108f718f92721af3b95e74349a`
   - Spatial Resolution: $32 \times 32$ pixels, 3 RGB color channels.
   - 10 classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.
2. **CIFAR-100 (`cifar-100-python.tar.gz`):**
   - Primary URL: `https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz`
   - Fallback URL: `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
   - MD5 Checksum: `eb9058c3a382ffc7106e4002c42a8d85`
   - Spatial Resolution: $32 \times 32$ pixels, 3 RGB color channels.
   - 100 fine-grained classes grouped into 20 coarse superclasses.

---

## 2. Dataset Partitioning Methodology (Stratified Random Hold-Out)

The partitioning protocol adheres strictly to empirical machine learning standards:
- **Test Set (10,000 images):** Retains 100% of the official CIFAR test set published by Alex Krizhevsky. This set is strictly isolated—**never involved in training updates, loss computation, or hyperparameter selection**.
- **Original Training Set (50,000 images):** Stratified split into a $9 : 1$ ratio using fixed random seed `seed=42`:
  - **Experimental Training Split:** $45,000$ images ($90\%$).
  - **Validation Split:** $5,000$ images ($10\%$).

---

## 3. Sample Allocation Statistics

| Dataset | Class Count | Training Split (90%) | Validation Split (10%) | Held-Out Test Set | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | 10 classes | **45,000 images** (4,500 / class) | **5,000 images** (500 / class) | **10,000 images** (1,000 / class) | 60,000 images |
| **CIFAR-100** | 100 classes | **45,000 images** (450 / class) | **5,000 images** (50 / class) | **10,000 images** (100 / class) | 60,000 images |

### Functional Roles of Each Partition:
1. **Train (45,000 images):** Drives model parameter updates via gradient descent optimization (SGD + Nesterov momentum or Adam) augmented with `RandomCrop`, `RandomHorizontalFlip`, and `Cutout (16x16)`.
2. **Validation (5,000 images):** Evaluates generalization performance at the close of every epoch to checkpoint the best-performing model weights (`best_val.pt`).
3. **Test (10,000 images):** Provides unbiased black-box evaluation; assessed exactly once using `best_val.pt` to compute official final metrics for the written technical report and oral defense.
