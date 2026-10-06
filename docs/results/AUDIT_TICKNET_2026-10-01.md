# Technical Audit and Performance Summary: TickNet Basic / C / L

Audit Date: 2026-10-01. Source code inspected at commit
`9cc8126942ba1a59899c348d20a34365eec7f137`.

## 1. Executive Summary

**Basic retains 100% of the original TickNet-Basic architecture. C inherits directly from the TickNet class and original FR-PDP block. L is an evolved architectural variant derived from TickNet-Basic with internal block structural modifications; L should be described as a proposed variant rather than an unmodified FR-PDP block.**

No computational flow errors, residual disconnects, tensor shape mismatches, or gradient anomalies were found in the current L configuration. The architectural strategy of reducing early stage compute to fund deeper stages is theoretically well-founded. Empirical results for L constitute compelling preliminary evidence, though they do not conclusively isolate the contribution of each individual component or establish unconditional superiority over Basic/C.

**All six checkpoints were re-evaluated on the local test split: Top-1 Accuracy, Macro F1, and all confusion matrices match published reports exactly.** A two-sided McNemar paired test (L vs. Basic) yields $p = 0.3323$ on Mid32 and $p = 0.1153$ on Mid224 (above the conventional 0.05 threshold). This does not demonstrate statistical equivalence; rather, statistical power is naturally constrained on this sample size.

A previous discrepancy was resolved: **the train/test membership of L matches Basic/C**, verified by reconstructing the manifest in the notebook's exact schema and matching SHA-256 digests. Differing raw CSV hashes resulted from schema formatting, not differing data splits.

A notable reproducibility gap was identified for Basic/C Mid224 notebooks: they require `--resume` and `--stop-after-epoch`, options absent from the current checked-in `train_mid.py`. Replicating the 100+100 epoch schedule requires bundling those trainer flags. This represents a packaging omission rather than an error in model weights or checkpoints.

## 2. Experimental Results Summary

Each training run used 25,000 training images, 250 test images across 5 balanced classes, 200 epochs, seed 42, SGD with learning rate 0.1, momentum 0.9, weight decay 1e-4, and CosineAnnealingLR. Primary logs reside in `training_logs/` and per-model reports.

| Model | Dataset | Top-1 | Correct / 250 | Macro F1 | Parameters | GFLOPs |
|---|---|---:|---:|---:|---:|---:|
| Basic | Mid32 | 89.6% | 224 | 0.89560 | 1,062,223 | 0.158428 |
| C | Mid32 | 91.2% | 228 | 0.91171 | 5,155,467 | 0.256830 |
| L | Mid32 | 91.6% | 229 | 0.91575 | 1,096,260 | 0.157821 |
| Basic | Mid224 | 92.4% | 231 | 0.92396 | 1,062,223 | 0.988343 |
| C | Mid224 | 94.4% | 236 | 0.94395 | 5,155,467 | 0.821054 |
| L | Mid224 | 95.6% | 239 | 0.95588 | 1,096,260 | 0.796760 |

L outperforms Basic by 5 correct images on Mid32 and 8 images on Mid224; outperforming C by 1 and 3 images respectively (each image represents 0.4 percentage points). L increases parameter count by ~3.2% relative to Basic, reduces FLOPs by 19.4% on Mid224, and reduces FLOPs by 0.38% on Mid32.

C increases parameter count by ~4.85× relative to Basic. That C consumes fewer FLOPs than Basic on Mid224 is consistent: C downsamples spatial resolution earlier, shifting the bulk of capacity into smaller feature maps. On Mid32, C's FLOPs are higher.

FLOP calculations cover Conv2d and Linear layers following **1 MAC = 2 FLOPs**, single image, eval mode, excluding BN, activations, pooling, residual additions, and SE scaling. This measures arithmetic operations, not hardware latency or VRAM footprint.

L achieves highest Top-1 and F1 across these six runs, but does not dominate every metric: test loss on Mid224 for C is **0.16260**, lower than L's **0.16667**. Top-1 and cross-entropy evaluate distinct objectives; this reflects expected trade-offs.

## 3. Preservation of Original Architecture in Kaggle Scripts

### 3.1. Basic: Architectural Fidelity with Task-Specific Recipe
Git diff confirms `models/TickNet.py`, `models/common.py`, and `models/SE_Attention.py` match `upstream/main` commit `e71679978f861a599192ece7613dc7ad3768e55b` from the [author's TickNets repository](https://github.com/nttbdrk25/TickNets).

`models/mid_models.py` calls `build_TickNet(..., typesize='basic', cifar=...)`. The original block flow at `models/TickNet.py:33` is:

```text
x → PW1 linear → DW 3×3 + BN + ReLU → PW2 + BN + ReLU → SE
  → add shortcut identity or PW projection
```

Stem 32 channels, stage schedule 128 → 64 → 128 → 256 → 512, head 1024, global average pooling, and classifier head are preserved. The output layer is adapted to 5 classes; Mid32 selects the CIFAR stride schedule from the author's factory.

This represents **applying the author's architecture to the midterm task**, rather than replicating the exact paper training recipe (e.g., `TickNet_Dogs.py` used MultiStepLR with milestones [100, 150, 180], while `train_mid.py` uses Cosine Annealing; dataset, augmentation, and evaluation protocols differ).

### 3.2. Model C: Direct Inheritance with Backbone Reconfiguration
`models/ticknet_c.py:16` directly constructs `TickNet`. C contains 9 original `FR_PDP_block` instances, retaining SE, stem 32, head 1024, and operational ordering. Reconfigurations include:
- Channel schedule: 80 / 48 / (96,128) / (192,224) / (640,768,896).
- Block counts: 1 / 1 / 2 / 2 / 3.
- Mid224 uses stride schedule (2,1,2,2,2) adapted from TickNet-small/large.

### 3.3. Model L: PDP Variant with Bottlenecking and Mixed Depthwise Convolutions
`models/ticknet_l.py:47` implements the redesigned block, preserving the PW–DW–PW sequence, SE attention, and residual shortcuts. Output channel schedule:
24 → 112 → 64 → 144 → 144 → 288 → 288 → 512 → 768.

| Component | Basic | L |
|---|---|---|
| Stem | 32 | 24 |
| Block count | 5 | 7 |
| PW1 | Cin → Cin, linear | Cin → hidden, linear |
| hidden | Cin | ~0.75 Cin from stage 2, multiple of 8 |
| DW | 3×3 | Stage 1–2: 3×3; stage 3–5: split channels 3×3 and 5×5 |
| SE | Built-in | Retained |
| Shortcut | Identity / projection | Identity / projection |
| Head | 1024 | 768 |

L's parallel DW branches process **disjoint channel subsets** before concatenation. Convolutions enforce `groups = channels`, maintaining depthwise efficiency. PW2 recombines cross-branch features. Strides and padding ensure dimensional alignment with residual shortcuts.

Note: L is not the author's TickNet-**large**: `--model l` and `--model large` are separate options. The recommended designation is **TickNet-L v1, proposed variant of TickNet-Basic**.

## 4. Implementation and Methodological Soundness of TickNet-L

### 4.1. Verification of Implementation Correctness
32 unit tests across model, pipeline, and dataset modules passed with 100% success rate. Verifications cover forward/backward passes at 32 and 224, finite gradients across all L parameters, independent FLOP profiling, and checkpoint loading.

Degenerate equivalence test: setting `hidden=Cin`, `kernels=(3,)` and copying weights from the original block into block L yielded maximum absolute error **0.0** across all four $(C_{in}, C_{out}, stride)$ configurations: (32,32,1), (32,64,1), (32,64,2), (32,32,2), recorded in [`block_equivalence.json`](audit_20261001/block_equivalence.json).

L's shortcuts receive full uncompressed tensors. Equal-shape blocks use identity mappings. SE precedes residual addition as in the baseline. Softmax is omitted before CrossEntropyLoss. Evaluation mode is strictly enforced during testing without augmentation.

### 4.2. Methodological Rationale and Scope of Claims
In Basic Mid224, stage 2 consumes 0.52103 GFLOPs (~52.7% of the network). PW1 precedes downsampling, resulting in high compute on $112 \times 112$ feature maps. L reduces input channels from 128 to 112 and hidden channels to 88, decreasing stage 2 cost to 0.33252 GFLOPs. This provides a clear computational budget to add blocks in stages 3–4 and expand kernel sizes on channel subsets.

Mixed depthwise convolutions draw from [MixConv, Tan & Le (2019)](https://arxiv.org/abs/1907.09595). SE attention is inherited from TickNet. The defensible contribution is **the resource allocation and configuration of the L variant**, rather than the invention of SE, residuals, or mixed depthwise convolutions.

Channel bottlenecking compresses representations; deeper blocks and larger kernels increase capacity. The 0.75 ratio and specific channel schedules are justified by computational budget constraints.

### 4.3. Empirical Scope and Limitations
- Evaluated on a single seed; cross-run variance not yet quantified.
- L Mid32 used batch size 128; Basic/C used 64. With 25,000 images and `drop_last=False`, L runs 196 steps/epoch vs 391 for Basic/C (39,200 vs 78,200 updates total), affecting batch normalization dynamics.
- L used 2 workers; Basic/C used 0, which can alter augmentation sequences.
- Validation curves were not tracked separately; training curves approached 100% across all models.
- Individual component contributions (MixConv vs. depth vs. channels) have not been isolated via ablation.
- Test set contains 250 images; the 1-image delta between L and C on Mid32 should not be over-interpreted.

### 4.4. Per-Sample Prediction Verification & McNemar Analysis
All checkpoints were re-evaluated on the identical test split, saving six `*_predictions.csv` files. Two-sided exact McNemar tests evaluate disagreement pairs:

| Dataset | Pair A → B | A Correct, B Wrong | A Wrong, B Correct | Two-sided p |
|---|---|---:|---:|---:|
| Mid32 | Basic → L | 6 | 11 | 0.3323 |
| Mid224 | Basic → L | 6 | 14 | 0.1153 |
| Mid32 | C → L | 10 | 11 | 1.0000 |
| Mid224 | C → L | 5 | 8 | 0.5811 |
| Mid32 | Basic → C | 12 | 16 | 0.5716 |
| Mid224 | Basic → C | 8 | 13 | 0.3833 |

None of the pairs achieve $p < 0.05$. Therefore, claims of "statistically significant superiority" should be avoided. The small sample size naturally limits statistical power without precluding practical performance benefits.

Wilson 95% Confidence Intervals for L: **87.50–94.44%** on Mid32 and **92.29–97.53%** on Mid224.

On Mid32, L misclassifies 9/50 dog images as cat (dog recall 80%, cat recall 98%), highlighting resolution-dependent confusion. On Mid224, recall reaches 92% dog, 92% cat, 96% bird, 98% frog, and 100% horse (50 samples per class).

## 5. Kaggle Reproducibility: Gaps and Action Items

1. **Trainer packaging for Basic/C Mid224**: Notebooks pass `--stop-after-epoch 100` and `--resume`. TheChecked-in `train_mid.py` needs explicit support for these arguments and RNG/DataLoader state serialization.
2. **Branch pinning**: `Kaggle_TickNet_L_Midterm_Final.ipynb` used `--branch feature/giua-ki-model-l`; committing specific commit SHAs into artifacts ensures immutability.
3. **Manifest schema alignment**: `train_mid.py:103` verifies raw CSV hashes. Standardizing manifest schemas prevents CLI `--evaluate` rejection when membership is equivalent.
4. **Packaging manifests in artifacts**: Future runs should bundle manifest and dataset checksums directly in release archives.

## 6. Recommended Reporting Guidelines

Recommended framing for academic presentation:

> TickNet-L v1 evolves from TickNet-Basic via intermediate channel bottlenecking, reallocation of depth and channels to deeper stages, and mixed 3×3–5×5 depthwise convolutions. With seed 42 on the documented split, L achieves 91.6% on Mid32 and 95.6% on Mid224, reducing Conv/Linear FLOPs on Mid224 by 19.4% relative to Basic. These results demonstrate the feasibility and promise of the architecture; individual component contributions and multi-seed stability remain subjects for extended study.

Recommended roadmap for extended experiments:
1. Package trainer supporting multi-session resume with pinned commits and checksums.
2. Maintain separate validation splits to prevent test-set overfitting.
3. Benchmark across 3–5 seeds with identical batch sizes, worker counts, and optimizers.
4. Conduct controlled ablations (e.g. MixConv vs. single kernel, bottleneck vs. uncompressed).
5. Measure wall-clock inference latency and memory on identical hardware.

TickNet-L satisfies all parameter and FLOP constraints mandated by the examination specifications, introducing concrete architectural enhancements with validated performance across both benchmark resolutions.

## 7. Audit Evidence and Artifact Manifest

- [`verification.json`](audit_20261001/verification.json): Six checkpoints verified via SHA-256, strict `state_dict` loading, 200 epochs completed, cosine LR schedule matching theory (max error ~4.86e-17). Recomputed Top-1 and confusion matrices match published values exactly.
- SHA-256 hashes of all **50,500 local images** match standard manifests.
- Six Conv/Linear layer counts match published figures and PyTorch's `FlopCounterMode`.
- [`source_checks.json`](audit_20261001/source_checks.json): Provenance of files, CLI options, and unit test logs.
- [`verify_evidence.py`](audit_20261001/verify_evidence.py): Verification script for membership hashes, checkpoints, and test inference.
- [`block_equivalence.json`](audit_20261001/block_equivalence.json): Verifies degenerate equivalence between block L and original block.
