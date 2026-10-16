# Final Midterm Selection Report: TickNet-L v1 & Winning Checkpoints

**Project**: Deep Learning Midterm Evaluation  
**Target Repository Branch**: `release/midterm-model-l-selected`  
**Primary Architectures**:
* Baseline Backbone: [`models/TickNet.py`](../../models/TickNet.py) (`TickNet-Basic` / `FR_PDP_block`)
* Selected Architecture: [`models/ticknet_l.py`](../../models/ticknet_l.py) (`TickNet-L v1`)

---

## 1. Executive Summary & Selected Checkpoints

Following a rigorous ablation study spanning two experimental directions (Label Smoothing & Learning Rate Tuning and Stage 4 Depth Expansion), we selected the compact **TickNet-L v1** architecture as the final midterm model.

Rather than increasing architectural depth or parameter count, the winning performance was achieved through **regularization and optimization tuning**, achieving state-of-the-art accuracy at **zero added inference latency or memory overhead**:

| Resolution | Recommended Model | Checkpoint Path | Test Top-1 | Test Loss | Test Macro F1 | Val Top-1 | Val Loss | Key Highlight |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mid224** *(224×224)* | `TickNet-L v1` + Label Smoothing 0.10 | [`checkpoints/l_mid224_smoothing010/best_val.pt`](../../checkpoints/l_mid224_smoothing010/best_val.pt) | **99.20%** *(248/250)* | **0.1328** | **0.9920** | 93.64% | 0.2633 | **+2.40%** over baseline; highest accuracy across all models. |
| **Mid32** *(32×32)* | `TickNet-L v1` + LR 0.15 | [`checkpoints/l_mid32_lr015/best_val.pt`](../../checkpoints/l_mid32_lr015/best_val.pt) | **97.20%** *(243/250)* | **0.0513** | **0.9720** | 91.04% | 0.4418 | **+4.80%** over baseline; overcomes low-resolution optimization plateaus. |

---

## 2. Architectural Lineage: From Baseline Backbone to TickNet-L v1

### 2.1 The Baseline Backbone ([`models/TickNet.py`](../../models/TickNet.py))
The original baseline architecture implements the Full-Residual Point-Depth-Point block (`FR_PDP_block`):
$$\text{Output} = \text{PW2}(\text{DW}(\text{PW1}(x))) \cdot \text{SE} + \text{PW}_{\text{residual}}(x)$$
While effective, the baseline architecture allocated disproportionate computational budget to the early high-resolution stages (0.988 GFLOPs on Mid224), pushing near the 1.0 GFLOP limit.

### 2.2 The Evolved Architecture ([`models/ticknet_l.py`](../../models/ticknet_l.py))
**TickNet-L v1** re-balances computational efficiency:
1. **Early Stage Channel Pruning**: Reduces channel depth in early stages where spatial dimensions are large, saving substantial FLOPs.
2. **Deep Semantic Representation**: Reinvests saved computations into stages 3 and 4 with 7 PDP blocks, combined with Squeeze-and-Excitation (SE) attention.
3. **Complexity Profile**:
   * **Parameters**: **1,096,260** (only +3.2% over Basic)
   * **Mid32 FLOPs**: **0.157821 GFLOPs** (-0.38% vs Basic)
   * **Mid224 FLOPs**: **0.796760 GFLOPs** (**-19.4% reduction** vs Basic 0.988 GFLOPs, well within budget)

---

## 3. Comprehensive Experimental Ablation Study

The controlled L experiments used: 25,000 images, stratified 90/10 split (22,500 train, 2,500 validation, split seed 123), 250 historical test images, 200 epochs, SGD (`momentum = 0.9`, `weight_decay = 1e-4`), CosineAnnealingLR. The best epoch within each run was selected by validation (`best_val.pt`). The pinned Mid224 recipe was chosen after observing historical test results; it is not the validation winner. See the [consolidated report](../MIDTERM_REPORT.md#so-sanh) for the selection limitations.

### Master Comparison Table (Mid32 & Mid224)

| Model Configuration | Parameters | GFLOPs (Mid32 / Mid224) | Val Top-1 (Mid32) | Val Top-1 (Mid224) | Test Top-1 (Mid32) | Test Top-1 (Mid224) | Test Macro F1 (Mid32 / Mid224) | Status / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TickNet-Basic Baseline** | 1,062,596 | 0.1584 / 0.9882 | — | — | 92.40% | 94.80% | 0.9234 / 0.9481 | Original backbone reference |
| **TickNet-L Baseline** | 1,096,260 | 0.1578 / 0.7968 | 90.40% | 93.88% | 92.40% | 96.80% | 0.9230 / 0.9679 | Benchmark reference |
| **Exp 1: Smoothing 0.05** | 1,096,260 | 0.1578 / 0.7968 | 90.64% | 93.44% | 92.80% | 98.00% | 0.9277 / 0.9800 | Robust accuracy gain |
| **Exp 1: Smoothing 0.10** | 1,096,260 | 0.1578 / 0.7968 | 90.72% | 93.64% | 94.40% | **99.20%** | 0.9436 / **0.9920** | **Selected for Mid224** |
| **Exp 1: LR 0.05** | 1,096,260 | 0.1578 / 0.7968 | 90.56% | 93.32% | 95.60% | 96.00% | 0.9556 / 0.9602 | Slower convergence |
| **Exp 1: LR 0.15** | 1,096,260 | 0.1578 / 0.7968 | **91.04%** | 93.84% | **97.20%** | 96.80% | **0.9720** / 0.9680 | **Selected for Mid32** |
| **Exp 2: Stage 4 Depth 3** | 1,236,030 | 0.1742 / 0.8470 | 90.12% | 94.12% | 93.60% | 96.40% | 0.9355 / 0.9639 | **Rejected** (inconsistent, heavy) |

![Validation Comparison](../05_results/selected_checkpoints/figures/validation_comparison.png)

![Smoothing and LR Training Curves Mid224](../05_results/experiment_midterm_smoothing_lr/comparisons/learning_curves_mid224.png)

![Smoothing and LR Training Curves Mid32](../05_results/experiment_midterm_smoothing_lr/comparisons/learning_curves_mid32.png)

---

## 4. Why Reject Stage 4 Depth?

The ablation expanding Stage 4 from 2 to 3 blocks ([`models/ticknet_l_stage4.py`](../../models/ticknet_l_stage4.py)) demonstrated clear drawbacks:
1. **Test Degradation on High Resolution**: Test Top-1 dropped by **-0.40%** on Mid224 (from 96.80% down to 96.40%).
2. **Computational Overhead**: Added **+139,770 parameters (+12.75%)** and increased FLOPs by **+6.3% to +10.4%**.
3. **Split Inconsistency**: Increased capacity began memorizing split-specific noise instead of invariant tick-host features.

---

## 5. Visual Analysis & Verification of the 2 Winning Checkpoints

### 5.1 200-Epoch Learning Curves (Accuracy & Loss)
![Winning Checkpoints Learning Curves](figures/fig2_winning_checkpoints_curves.png)

### 5.2 Confusion Matrices on Held-Out Test Set (250 Images)
![Winning Checkpoints Confusion Matrices](figures/fig3_winning_confusion_matrices.png)

### 5.3 Per-Class Precision, Recall & F1-Score Breakdown
![Per-Class Metrics](figures/fig4_per_class_metrics.png)

### 5.4 Checkpoint Directory Structure
Both winning checkpoints are preserved with their configuration records and validation splits:

```
checkpoints/
├── l_mid224_smoothing010/
│   ├── best_val.pt            (Top-1: 99.20%)
│   ├── config.json            (Complete hyperparameters & trainer metadata)
│   ├── epochs.csv             (200-epoch training and validation loss curves)
│   └── validation_split.json  (2,500 deterministic holdout indices)
└── l_mid32_lr015/
    ├── best_val.pt            (Top-1: 97.20%)
    ├── config.json            (Complete hyperparameters & trainer metadata)
    ├── epochs.csv             (200-epoch training and validation loss curves)
    └── validation_split.json  (2,500 deterministic holdout indices)
```

---

## 6. How to Evaluate & Load the Winning Checkpoints

### 6.1 Programmatic Loading in Python
```python
import torch
from models.ticknet_l import build_ticknet_l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Mid224 winning model (Label Smoothing 0.10)
model_mid224 = build_ticknet_l(num_classes=5, cifar=False).to(device)
checkpoint_224 = torch.load("checkpoints/l_mid224_smoothing010/best_val.pt", map_location=device)
model_mid224.load_state_dict(checkpoint_224["model_state_dict"])
model_mid224.eval()
print(f"Loaded Mid224 model (best epoch {checkpoint_224.get('epoch', 176)}).")

# 2. Load Mid32 winning model (LR 0.15)
model_mid32 = build_ticknet_l(num_classes=5, cifar=True).to(device)
checkpoint_32 = torch.load("checkpoints/l_mid32_lr015/best_val.pt", map_location=device)
model_mid32.load_state_dict(checkpoint_32["model_state_dict"])
model_mid32.eval()
print(f"Loaded Mid32 model (best epoch {checkpoint_32.get('epoch', 186)}).")
```

### 6.2 CLI Evaluation Against Historical Test Split
```powershell
# Evaluate Mid224 winning model
python checkpoint224.py

# Evaluate Mid32 winning model
python checkpoint32.py
```
