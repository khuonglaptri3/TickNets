# Command-Line Reference Guide: TickNets Final Examination

This handbook consolidates all standardized and tested CLI commands supporting the final examination workflow: from automated CIFAR-10/100 dataset download and verification, FLOPs/parameter profiling, training TickNet-L & author baselines, running hyperparameter grid searches, to evaluation and report aggregation.

---

## 1. Environment Verification & Automated Test Suite (PyTest)

Execute the comprehensive test suite verifying architectures, optimizers (SGD/Adam), datasets, evaluation metrics, checkpoint resuming, and Kaggle notebook integrity:
```bash
python -m pytest tests -v
```
*(Prerequisite: 100% tests PASSED prior to launching long-running training).*

---

## 2. Automated Download & Verification of CIFAR Datasets

Downloads datasets directly from the University of Toronto servers (`cave.cs.toronto.edu`) with automatic fallback mirror (`www.cs.toronto.edu`) and Linux SSL certificate handling:

```bash
# Download both CIFAR-10 and CIFAR-100 into data/
python download_cifar.py --data-root data --dataset all

# Download individually:
python download_cifar.py --data-root data --dataset cifar10
python download_cifar.py --data-root data --dataset cifar100

# Force re-download (bypass local cache) and verify MD5 checksums:
python download_cifar.py --data-root data --force
```

---

## 3. Model Complexity Profiling (Params & FLOPs)

Verifies compliance with final examination constraints (Learnable Params $\le 6,000,000$, FLOPs $< 1,000,000,000$):

```bash
# Profile TickNet-L complexity on CIFAR-10 and CIFAR-100
python -c '
from models.ticknet_l import build_ticknet_l
from models.model_profile import profile_model
print("TickNet-L CIFAR-10:", profile_model(build_ticknet_l(10, cifar=True), 32))
print("TickNet-L CIFAR-100:", profile_model(build_ticknet_l(100, cifar=True), 32))
'

# Profile author baseline TickNet-Basic on CIFAR-10 and CIFAR-100
python -c '
from models.TickNet import build_TickNet
from models.model_profile import profile_model
print("TickNet-Basic CIFAR-10:", profile_model(build_TickNet(10, typesize="basic", cifar=True), 32))
print("TickNet-Basic CIFAR-100:", profile_model(build_TickNet(100, typesize="basic", cifar=True), 32))
'
```

*Verified Profile Statistics:*
- **TickNet-L (CIFAR-10):** 1,100,105 parameters ($\le 6M$) | 0.1578 GFLOPs ($< 1G$).
- **TickNet-L (CIFAR-100):** 1,169,315 parameters ($\le 6M$) | 0.1580 GFLOPs ($< 1G$).
- **TickNet-Basic (CIFAR-10):** 1,067,348 parameters ($\le 6M$) | 0.1584 GFLOPs ($< 1G$).
- **TickNet-Basic (CIFAR-100):** 1,159,598 parameters ($\le 6M$) | 0.1586 GFLOPs ($< 1G$).

---

## 4. Final Examination Model Training (`train_cifar.py`)

Trains for 200 epochs with Cosine Annealing LR decay to 0, supporting both `sgd` (with Nesterov momentum) and `adam`, along with $16 \times 16$ `Cutout` data augmentation:

### 4.1. Training via JSON Configurations (`configs/final/`)
```bash
# CIFAR-10 with SGD lr=0.10 (Nesterov + Cutout)
python train_cifar.py --config configs/final/cifar10_sgd_lr010.json --data-root data --output-dir runs/cifar10_sgd_lr010

# CIFAR-10 with Adam lr=0.001 (Cutout)
python train_cifar.py --config configs/final/cifar10_adam_lr0001.json --data-root data --output-dir runs/cifar10_adam_lr0001

# CIFAR-100 with SGD lr=0.10 (Nesterov + Cutout)
python train_cifar.py --config configs/final/cifar100_sgd_lr010.json --data-root data --output-dir runs/cifar100_sgd_lr010

# Author TickNet-Basic baseline on CIFAR-10
python train_cifar.py --config configs/final/baseline_cifar10_sgd_lr010.json --data-root data --output-dir runs/baseline_cifar10_sgd_lr010
```

### 4.2. Training via Direct CLI Arguments
```bash
# Run TickNet-L on CIFAR-100 with SGD lr=0.15
python train_cifar.py --model l --dataset cifar100 --optimizer sgd --learning-rate 0.15 --momentum 0.9 --nesterov --cutout --cutout-length 16 --epochs 200 --batch-size 128 --output-dir runs/cifar100_sgd_lr015

# Run author TickNet-Basic on CIFAR-100
python train_cifar.py --model basic --dataset cifar100 --optimizer sgd --learning-rate 0.10 --momentum 0.9 --nesterov --cutout --cutout-length 16 --epochs 200 --batch-size 128 --output-dir runs/baseline_cifar100_sgd_lr010
```

### 4.3. Resuming Interrupted Training Runs (`--resume`)
```bash
python train_cifar.py --config configs/final/cifar10_sgd_lr010.json --resume runs/cifar10_sgd_lr010/last.pt --output-dir runs/cifar10_sgd_lr010
```

### 4.4. Standalone Test Set Checkpoint Evaluation (`--evaluate`)
```bash
python train_cifar.py --dataset cifar10 --evaluate runs/cifar10_sgd_lr010/best_val.pt --output-dir runs/cifar10_sgd_lr010/eval_test
```

---

## 5. Kaggle GPU Execution (Tesla T4 / P100)

Modularized into 5 standalone, self-contained notebooks located in [`docs/kaggle/`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle):

1. **Author Baseline:** Upload [`docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb) $\to$ Runs 2 baseline experiments on CIFAR-10 & CIFAR-100.
2. **Phase 1 (CIFAR-10 SGD):** Upload [`docs/kaggle/Phase1_CIFAR10_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase1_CIFAR10_SGD.ipynb) $\to$ Runs SGD lr=0.10 & lr=0.15.
3. **Phase 2 (CIFAR-10 Adam):** Upload [`docs/kaggle/Phase2_CIFAR10_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase2_CIFAR10_Adam.ipynb) $\to$ Runs Adam lr=0.001 & lr=0.0003.
4. **Phase 3 (CIFAR-100 SGD):** Upload [`docs/kaggle/Phase3_CIFAR100_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase3_CIFAR100_SGD.ipynb) $\to$ Runs SGD lr=0.10 & lr=0.15.
5. **Phase 4 (CIFAR-100 Adam):** Upload [`docs/kaggle/Phase4_CIFAR100_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase4_CIFAR100_Adam.ipynb) $\to$ Runs Adam lr=0.001 & lr=0.0003.

Phases 1–4 embed an audited source snapshot, preflight tests, recovery mechanisms, and artifact validation.
Refer to the current [Kaggle Guide](../kaggle/README.md). A `results.zip` package is exported only after completing all 200 epochs; partial runs export a `recovery.zip` bundle.

---

## 6. Master Report & Result Aggregation

After downloading and extracting experiment runs into the `runs/` directory, execute the aggregation script:
```bash
python scripts/aggregate_grid_search.py --runs-dir runs --output-csv docs/results/grid_search_summary.csv --output-md docs/results/grid_search_summary.md
```
The script automatically:
- Ingests all 8 grid search run directories; flags missing runs or invalid artifacts. Baselines are aggregated separately.
- Identifies `Best Val Epoch`, `Best Val Acc`, `Test Top-1 Accuracy`, `Test Loss`, and `Macro F1`.
- Exports Markdown and CSV tables formatted directly for inclusion in the final examination technical report.
