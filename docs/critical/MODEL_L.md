# Proposed TickNet-L v1 Architecture for CIFAR-10 & CIFAR-100

This document details the architectural design of **TickNet-L v1** ([`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py)), its technical improvements compared to the author's original **TickNet-Basic** baseline ([`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py)), and mathematical proofs verifying compliance with the final examination budget constraints.

---

## 1. Design Objectives & Final Examination Constraints

The examination specification establishes two strict prerequisites for Model L:
1. **Learnable Parameters:** Strictly capped at **$\le 6,000,000$ parameters (6M)**.
2. **Computational Complexity (Forward FLOPs):** Strictly capped at **$< 1,000,000,000$ FLOPs (1G)**.
3. **Experimental Target:** Train and evaluate on two standard benchmark datasets: **CIFAR-10** (10 classes) and **CIFAR-100** (100 classes) at native $32 \times 32$ image resolution.

---

## 2. Comparative Matrix: TickNet-L v1 vs. Author's TickNet-Basic

| Architectural Component | TickNet-Basic (Author Baseline) | TickNet-L v1 (Proposed) | Technical Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Stem Conv (Initial)** | 32 channels, Stride 1 | **24 channels, Stride 1** | Conserves computation at early layers where spatial resolution is largest ($32 \times 32$). |
| **5-Stage Elasticity Channels** | [128, 64, 128, 256, 512] | **[112, 64, 144, 288, 512]** | Budget reallocation: Decreases channels in Stage 1, boosts representation capacity in Stages 3 & 4. |
| **Stage Block Depths** | [1, 1, 1, 1, 1] (5 blocks) | **[1, 1, 2, 2, 1] (7 blocks)** | Increases network depth at compact spatial resolutions ($16 \times 16$ and $8 \times 8$), improving non-linear feature extraction. |
| **Block-Entry Pointwise Conv** | $C_{in} \rightarrow C_{in}$ (No compression) | **$C_{in} \rightarrow \text{hidden}$ (0.75 Ratio)** | Reduces matrix multiplication operations by $25\%$ at the computationally expensive pointwise stage. |
| **Depthwise Convolution** | Pure $3 \times 3$ across all channels | **Mixed DW ($3 \times 3$ and $5 \times 5$)** | Stages 1–2 use $3 \times 3$; Stages 3–5 split channels into parallel $3 \times 3$ and $5 \times 5$ branches, capturing multi-scale receptive fields. |
| **Squeeze-and-Excitation** | Included in all blocks ($r=16$) | **Retained SE Attention** | Recalibrates channel-wise feature dependencies. |
| **Pre-Pooling Conv Layer** | 1024 channels | **768 channels** | Saves parameters and FLOPs prior to the final classification head. |
| **Classifier Head** | 10 or 100 classes | **10 or 100 classes** | Fully compatible with both CIFAR-10 and CIFAR-100. |

---

## 3. Improved FR-PDP Block Structure in TickNet-L

The enhanced FR-PDP v1 block integrates pointwise bottlenecking with multi-scale depthwise convolutions:

```text
                     +---------- Shortcut Identity / Linear PW 1x1 ----------+
                     |                                                        |
Input (Cin) -> Linear PW 1x1 (hidden = 0.75*Cin) 
            -> Split channels (50% DW 3x3, 50% DW 5x5) 
            -> Concat -> BN + ReLU 
            -> PW 1x1 (Cout) + BN + ReLU 
            -> SE Attention (ChannelGate, r=16) 
            -> Element-wise Addition (+) -> Output (Cout)
```

Hidden channel rounding formula:
$$\text{hidden} = \max\left(16, \left\lfloor \frac{0.75 \times C_{in} + 4}{8} \right\rfloor \times 8\right)$$
Ensures hidden channel dimensions are multiples of 8, optimizing memory alignment and vectorized execution on GPU Tensor Cores.

---

## 4. Quantitative Complexity Proof & Budget Verification

Measured via [`models/model_profile.py`](../../models/model_profile.py) across Conv2d/Linear layers (excluding BN, activations, pooling, and element-wise ops), following the convention $1\text{ MAC} = 2\text{ FLOPs}$, batch size = 1, input tensor $(1, 3, 32, 32)$:

### 4.1. Results on CIFAR-10 (10 Classes)
- **Learnable Parameters:** **1,100,105 parameters** ($\approx 1.10\text{M} \le 6,000,000$ $\to$ **PASSED, utilizes 18.3% of allowed ceiling**).
- **Computational Cost (Forward FLOPs):** **157,828,544 FLOPs** ($\approx 0.1578\text{ GFLOPs} < 1,000,000,000$ $\to$ **PASSED, utilizes 15.8% of allowed ceiling**).

### 4.2. Results on CIFAR-100 (100 Classes)
- **Learnable Parameters:** **1,169,315 parameters** ($\approx 1.17\text{M} \le 6,000,000$ $\to$ **PASSED, utilizes 19.5% of allowed ceiling**).
- **Computational Cost (Forward FLOPs):** **157,966,784 FLOPs** ($\approx 0.1580\text{ GFLOPs} < 1,000,000,000$ $\to$ **PASSED, utilizes 15.8% of allowed ceiling**).

### 4.3. Complexity Comparison Summary

| Model Architecture | Dataset | Learnable Params | Param Budget ($\le 6\text{M}$) | Forward FLOPs | FLOPs Budget ($< 1\text{G}$) | Compliance Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TickNet-Basic** (Author Baseline) | CIFAR-10 | 1,067,348 | 17.8% | 0.1584 GFLOPs | 15.8% | **PASSED** |
| **TickNet-Basic** (Author Baseline) | CIFAR-100 | 1,159,598 | 19.3% | 0.1586 GFLOPs | 15.9% | **PASSED** |
| **TickNet-L v1** (Proposed) | CIFAR-10 | **1,100,105** | **18.3%** | **0.1578 GFLOPs** | **15.8%** | **PASSED** |
| **TickNet-L v1** (Proposed) | CIFAR-100 | **1,169,315** | **19.5%** | **0.1580 GFLOPs** | **15.8%** | **PASSED** |

*(Both models are verified and confirmed with 100% pass rate in unit test suite `tests/test_ticknet_l.py`).*
