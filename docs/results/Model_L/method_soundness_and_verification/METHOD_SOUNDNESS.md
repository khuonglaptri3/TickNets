# Methodological and Algorithmic Soundness of TickNet-L v1

This document consolidates the theoretical foundations, mathematical analyses, computational architecture, and empirical evidence proving the **methodological soundness and correctness** of the **TickNet-L v1** architecture in the context of Midterm and Final Examination projects.

---

## 1. Theoretical Foundations & Architectural Motivation

### 1.1. Computational Bottleneck Analysis in Baseline TickNet-Basic
In the baseline TickNet-Basic architecture at $224 \times 224$ input resolution:
- Stage 2 consumes **0.52103 GFLOPs**, representing **52.7% of total Conv/Linear compute across the network** (0.98834 GFLOPs).
- Root Cause: The entry 1×1 Pointwise convolution (PW1) processes high-resolution spatial feature maps ($112 \times 112$) with $C_{in} = 128$ channels before depthwise downsampling (stride 2).
- Impact: The network experiences an early computational bottleneck, exhausting the FLOP budget (< 1G) and preventing depth expansion in deeper semantic stages.

### 1.2. Controlled Linear Bottlenecking
Model L restructures the PDP block by inserting a controlled channel compression ratio:
$$\text{hidden} = \max\left(16, \left\lfloor \frac{0.75 \times C_{in} + 4}{8} \right\rfloor \times 8\right)$$
- In Stage 2: Input channels are adjusted from 128 to 112, with hidden channel dimension $\text{hidden} = 88$.
- Stage 2 computational cost decreases sharply from **0.521 GFLOPs to 0.333 GFLOPs** (~36% compute reduction in this stage).
- PW1 maintains linearity (omitting ReLU activation following the bottleneck projection) to preserve non-linear manifold capacity in low-dimensional space (adhering to MobileNetV2 principles).

### 1.3. Strategic FLOP Reallocation
Compute conserved at early stages is reinvested into deeper stages:
- **Increased Block Depth**: Stage 3 increases from 1 to 2 blocks; Stage 4 increases from 1 to 2 blocks (operating on spatial feature maps of $28 \times 28$ and $14 \times 14$).
- **Expanded Representational Capacity**: The network extracts richer semantic features without inflating overall FLOPs.
- Net Outcome: Total FLOPs at $224 \times 224$ drops from **0.988 GFLOPs to 0.797 GFLOPs** (a 19.4% reduction), while learnable parameters grow modestly from 1.06M to 1.10M (+3.2%).

### 1.4. Mixed Depthwise Convolution (MixConv)
Rather than applying a uniform $3 \times 3$ kernel across all channels:
- Hidden channels are partitioned into two parallel groups:
  - Group 1: $3 \times 3$ Depthwise convolution (capturing local high-frequency textures and edges).
  - Group 2: $5 \times 5$ Depthwise convolution (expanding effective receptive fields to capture macro-level object structure).
- Each branch enforces `groups = channels`, maintaining depthwise computational efficiency. Channel outputs are concatenated and processed by Pointwise PW2 ($1 \times 1$) to fuse cross-group information.

---

## 2. Algorithmic & Implementation Correctness Proofs

### 2.1. Base-Case Mathematical Reduction (Block Equivalence)
To verify `FR_PDP_block_L` contains zero structural distortion or mathematical drift compared to the author's original `FR_PDP_block`:
- Under the degenerate configuration $\text{hidden} = C_{in}$ and $\text{kernels} = (3,)$, weights from the author's original block are mapped into block L.
- Evaluated across four distinct $(C_{in}, C_{out}, stride)$ configurations:
  - $(32, 32, 1)$: Maximum absolute error $\mathbf{= 0.0}$
  - $(32, 64, 1)$: Maximum absolute error $\mathbf{= 0.0}$
  - $(32, 64, 2)$: Maximum absolute error $\mathbf{= 0.0}$
  - $(32, 32, 2)$: Maximum absolute error $\mathbf{= 0.0}$
- Verified in artifact: [`block_equivalence.json`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/block_equivalence.json).

### 2.2. Gradient Flow & Backpropagation Verification
- 32 unit tests covering model internals (`tests/test_ticknet_l.py`, `tests/test_mid_pipeline.py`, etc.) passed with 100% success rate.
- Verified forward and backward passes across both input resolutions ($32 \times 32$ and $224 \times 224$).
- Gradients across all learnable parameters (`requires_grad=True`) are verified non-zero and finite (no NaN or Inf values).
- Identity and linear projection shortcuts maintain uncorrupted gradient highways, and SE attention gates execute properly prior to residual addition.

### 2.3. Independent Parameter & FLOP Verification
- Protocol: Single forward pass in `eval` mode, batch size 1, float32, CPU, $1 \text{ MAC} = 2 \text{ FLOPs}$ (Conv2d and Linear only).
- Cross-verified using two independent profiling tools:
  1. Internal forward hook profiler.
  2. PyTorch native `torch.utils.flop_counter.FlopCounterMode`.
- Verification Outcome: **100% exact numerical agreement**:
  - Mid32: 1,096,260 parameters, 0.07891 GMACs $\to$ **0.157821 GFLOPs**.
  - Mid224: 1,096,260 parameters, 0.39838 GMACs $\to$ **0.796760 GFLOPs**.
- Artifact record: [`verification.json`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/verification.json).

---

## 3. Empirical Verification & Reproducibility

### 3.1. Dataset Integrity Verification
- SHA-256 digests across all **50,500 local image files** match the reference dataset manifest exactly.
- Rebuilding the CSV schema from standard membership reproduced the exact reference SHA-256 hash:
  `35bcc76fde54e97d54f7825cdb34376a3732fd79aa829624ea4fc982b4590bdf`.
- Conclusion: Basic, C, and L models were evaluated on the **identical train/test split (25,000 train / 250 test)**, guaranteeing fair benchmark comparisons.

### 3.2. Independent Test Set Re-inference
Independent verification on 2026-10-01 reloaded both `last.pt` checkpoints (epoch 200) for Model L and executed standalone test inference:
- **Mid32**:
  - Top-1 Accuracy: **91.6%** (229/250 correct).
  - Macro F1: **0.91575**.
  - Replicated Confusion Matrix: 100% identical to the reference $5 \times 5$ matrix.
- **Mid224**:
  - Top-1 Accuracy: **95.6%** (239/250 correct).
  - Macro F1: **0.95588**.
  - Replicated Confusion Matrix: 100% identical to the reference $5 \times 5$ matrix.
- Detailed per-sample test predictions are recorded in:
  - [`l_mid32_predictions.csv`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/l_mid32_predictions.csv)
  - [`l_mid224_predictions.csv`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/l_mid224_predictions.csv)

### 3.3. Statistical Analysis & McNemar Significance Testing
- Two-tailed McNemar test on per-sample predictions (250 test samples):
  - Model L vs Basic Mid32: L correct on 11 images where Basic failed; Basic correct on 6 where L failed ($p = 0.3323$).
  - Model L vs Basic Mid224: L correct on 14 images where Basic failed; Basic correct on 6 where L failed ($p = 0.1153$).
  - Model L vs Model C Mid32: L correct on 11 images where C failed; C correct on 10 where L failed ($p = 1.0000$).
  - Model L vs Model C Mid224: L correct on 8 images where C failed; C correct on 5 where L failed ($p = 0.5811$).
- Wilson 95% Confidence Intervals for Model L Top-1 Accuracy:
  - Mid32: $[87.50\% - 94.44\%]$
  - Mid224: $[92.29\% - 97.53\%]$
- Per-Class Analysis:
  - On Mid224: L performs exceptionally across all 5 classes: Horse 100% (50/50), Frog 98% (49/50), Bird 96% (48/50), Cat 92% (46/50), Dog 92% (46/50).
  - On Mid32: L exhibits mild confusion between `dog` and `cat` (9 dog images misclassified as cat), attributable to resolution downsampling from 224 to 32 attenuating fine whiskers and facial features.

---

## 4. Instructions for Re-running Verification Scripts

To re-run the verification pipeline locally:

```bash
python /home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/verify_evidence.py
```

The script automatically executes:
1. Re-reading all six checkpoints and computing SHA-256 digests.
2. Verifying dataset manifest membership equivalence.
3. Measuring FLOPs and parameter counts.
4. Running inference on the local test split and cross-referencing published reports.
