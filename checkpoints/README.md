# Pinned Checkpoints for Midterm Evaluation

This directory contains the verified winning checkpoints for the **TickNet-L v1** architecture ([`models/ticknet_l.py`](../models/ticknet_l.py)), selected based on the midterm ablation study across 4 experiments.

---

## 1. Selected Checkpoints

### A. Mid224 Resolution (224×224)
* **Directory**: [`checkpoints/l_mid224_smoothing010/`](l_mid224_smoothing010/)
* **Weights**: [`checkpoints/l_mid224_smoothing010/best_val.pt`](l_mid224_smoothing010/best_val.pt)
* **Training Recipe**: `TickNet-L v1` + Label Smoothing `0.10`, SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=1e-4`, CosineAnnealingLR)
* **Performance**:
  * **Test Top-1**: **99.20%** (248/250 correct, +2.40% over baseline)
  * **Test Loss**: **0.1328**
  * **Test Macro F1**: **0.9920**
  * **Validation Top-1**: **93.64%** (selected at epoch 176)
  * **FLOPs**: **0.796760 GFLOPs** (within 1.0 GFLOP limit)
  * **Parameters**: **1,096,260**

### B. Mid32 Resolution (32×32)
* **Directory**: [`checkpoints/l_mid32_lr015/`](l_mid32_lr015/)
* **Weights**: [`checkpoints/l_mid32_lr015/best_val.pt`](l_mid32_lr015/best_val.pt)
* **Training Recipe**: `TickNet-L v1` + Learning Rate `0.15`, SGD (`momentum=0.9`, `weight_decay=1e-4`, CosineAnnealingLR)
* **Performance**:
  * **Test Top-1**: **97.20%** (243/250 correct, +4.80% over baseline)
  * **Test Loss**: **0.0513**
  * **Test Macro F1**: **0.9720**
  * **Validation Top-1**: **91.04%** (selected at epoch 186)
  * **FLOPs**: **0.157821 GFLOPs**
  * **Parameters**: **1,096,260**

---

## 2. Quick Verification
To evaluate either checkpoint:
```powershell
# Evaluate Mid224 checkpoint
python checkpoint224.py

# Evaluate Mid32 checkpoint
python checkpoint32.py
```

See the full ablation study and rationale in [`docs/experiments/MIDTERM_FINAL_SELECTION_REPORT.md`](../docs/experiments/MIDTERM_FINAL_SELECTION_REPORT.md).
