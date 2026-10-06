# Results Analysis and Final Examination Development Strategy — 2026-10-01

**Executive Conclusion:** Continue developing the TickNet-L family, using L v1 as the baseline benchmark on CIFAR, prioritizing rigorous training protocols and generalization, followed by controlled single-factor investigations (increasing depth or relaxing channel compression). Premature adoption of Coordinate Attention, unsupported CIFAR accuracy forecasting, or unverified claims that smaller models are unconditionally superior to larger architectures must be avoided.

---

## 1. Audit Scope & Verification Rigor

This analysis evaluated technical reports, configurations, 200-epoch training logs, comparative summary tables, per-layer profiles, per-sample test predictions, forensic audit records, model/trainer/dataloader codebases, and both exam specifications in `.doc/`.

This review recomputed Top-1 Accuracy, Macro F1, confusion matrices, and McNemar test statistics (Basic vs. L) directly from raw CSV files; verifying SHA-256 digests across all four Basic/L checkpoints and their directory replicas. Full details are compiled in [FINAL_DEVELOPMENT_EVIDENCE_2026-10-01.json](FINAL_DEVELOPMENT_EVIDENCE_2026-10-01.json).

This review focuses on artifact verification. The Python runtime used here lacked PyTorch/TorchVision. The prior audit executed inference on all six checkpoints; currently, the repository retains prediction CSVs and checkpoints for Basic/L. Model C figures are cited from comparative tables and audit documents, and were not re-inferred in this pass. No CIFAR experimental results were found in `docs/results` at the time of this audit.

---

## 2. What Do Current Empirical Results Support?

| Architecture | Mid32 Top-1 | Mid224 Top-1 | Parameters (5 classes) | GFLOPs Mid32 | GFLOPs Mid224 |
|---|---:|---:|---:|---:|---:|
| Basic | 89.6% | 92.4% | 1,062,223 | 0.158428 | 0.988343 |
| C | 91.2% | 94.4% | 5,155,467 | 0.256830 | 0.821054 |
| L v1 | 91.6% | 95.6% | 1,096,260 | 0.157821 | 0.796760 |

Sources: [reported_metrics.csv](model_comparison/reported_metrics.csv), [audit report](AUDIT_TICKNET_2026-10-01.md), [verification.json](audit_20261001/verification.json).

L outperforms Basic by 2.0 percentage points on Mid32 and 3.2 percentage points on Mid224 with a modest +3.2% parameter increase. L reduces FLOPs by 19.4% on Mid224, but by only 0.38% on Mid32. Because the final examination operates on native $32 \times 32$ CIFAR images, large FLOP savings observed on Mid224 do not transfer directly to CIFAR.

L serves as a sound development baseline: solid observed performance, compact model budget, ample resource headroom, and architectural inheritance from the midterm. However:
- The test split contains only 250 samples; one sample equates to 0.4 percentage points. L outperforms C on Mid32 by exactly one image.
- A single seed (42) does not measure variance across runs. McNemar tests between Basic and L yield $p = 0.3323$ on Mid32 and $p = 0.1153$ on Mid224 (neither reaching $p < 0.05$). This does not demonstrate equivalence, but confirms statistical power is constrained.
- L Mid32 used batch size 128 while Basic/C used 64; worker counts also differed. Equal epochs do not equal equal SGD updates.
- Multiple architectural elements changed simultaneously. Accuracy gains cannot be attributed exclusively to mixed kernels, bottlenecking, or added depth without ablation.

---

## 3. Critical Weaknesses Requiring Priority

| Model Metric (L v1) | Mid32 | Mid224 |
|---|---:|---:|
| Train Top-1 Epoch 200 | 99.984% | 99.992% |
| Test Top-1 | 91.6% | 95.6% |
| Train–Test Gap | 8.384% | 4.392% |
| Total Test Errors | 21 | 11 |
| Cat ↔ Dog Confusion | 10/21 errors | 7/11 errors |
| Bird ↔ Frog Confusion | 8/21 errors | 2/11 errors |

On Mid32, dog recall drops to 80%; 9 out of 50 dog images are misclassified as cat. Cat recall is 98%, but precision is 84.48% due to false positives from dog images. This represents a classification asymmetry despite balanced class splits. Naively increasing class loss weights is discouraged as it risks inverting confusion patterns (shifting dog→cat into cat→dog).

Near-100% training accuracy alongside lower test performance motivates careful investigation of generalization. Training performance is measured with augmentations and active gradient updates, while testing uses eval mode; this gap is not a pure measure of overfitting. The absence of per-epoch validation curves leaves optimal early-stopping checkpoints uncharacterized.

SE attention utilizes spatial pooling to compute channel recalibration weights before re-multiplying the original spatial tensor; `models/SE_Attention.py` does not collapse feature maps into 1×1 tensors. Claims that "SE eliminates ear/whisker details and causes cat-dog confusion" are unfounded. Coordinate Attention (CA) is an interesting exploratory avenue, not a proven remedy. Mechanisms are detailed in the [SE paper](https://arxiv.org/abs/1709.01507) and [Coordinate Attention paper](https://arxiv.org/abs/2103.02907).

---

## 4. Final Examination Requirements & Priority Shifts

Exam specification [.doc/Final exam.docx](../../.doc/Final%20exam.docx): parameters $\le 6\text{M}$, FLOPs $< 1\text{G}$; train/test on CIFAR-10 and CIFAR-100 under multiple learning rates and both SGD/Adam. Grading breakdown: CIFAR-10 30%, CIFAR-100 30%, Oral Defense 30%, Technical Report 10%; passing oral defense is mandatory. The prompt allows inheriting midterm proposed architectures.

Both CIFAR benchmarks consist of $32 \times 32$ RGB images, with 50,000 training and 10,000 test samples. CIFAR-100 contains 100 fine classes (500 training images per class), requiring fine labels for classification. Source: [Alex Krizhevsky's CIFAR repository](https://cave.cs.toronto.edu/kriz/cifar.html).

Mid224 performance on 5 classes does not extrapolate to 100-class CIFAR-100. Speculative accuracy projections in earlier documents (e.g., 93.5–95.2% on CIFAR-10, 72–76.5% on CIFAR-100) lack empirical support and should be treated as aspiration targets.

Because 60% of the grade depends on empirical performance, the lightest model is not automatically optimal. Expanding capacity within constraints may benefit CIFAR-100, though risk of overfitting must be managed. A unified L backbone is maintained across both datasets, swapping 10/100 class heads and tailoring training recipes based on validation metrics.

---

## 5. Priority 1: Rigorous CIFAR Benchmarking & Training Pipeline

The midterm trainer lacked multi-dataset validation, resumption, and fine-grained logging. Transitioning to CIFAR requires production-grade engineering:
- **Stratified Partitioning:** Fixed 45,000 train / 5,000 validation split from official training data; official 10,000 test images remain held-out. For CIFAR-100, this yields 450 train + 50 validation per class.
- **Reproducibility Isolation:** Separate data split seeds from training RNG seeds. Record split indices, dataset checksums, git commit SHAs, configurations, and environment dependencies.
- **Dual Metric Tracking:** Log train/validation loss and Top-1; checkpoint `best_val.pt` and `last.pt`. Finalize checkpoint selection criteria before test inference.
- **Robust Resumption:** Serialize model, optimizer, scheduler, epoch index, RNG states, and scaler states. Anchor cosine annealing cycles to avoid resetting schedules mid-run.
- **Standardized Execution:** Enforce consistent batch sizes, worker counts, augmentations, and epoch counts across comparative architectures. Batch size 128 is standard for CIFAR on modern GPUs.
- **Train from Scratch Baseline:** Train from scratch on CIFAR to establish clean baselines. Midterm pretraining, if explored, must be documented separately.

Initial Learning Rate Grid Search Matrix:

| Optimizer | Initial LR | Configuration Details |
|---|---|---|
| SGD | 0.05, 0.10, 0.15 | Momentum 0.9, Weight Decay 1e-4, Nesterov enabled |
| Adam | 3e-4, 1e-3, 3e-3 | Betas (0.9, 0.999), Epsilon 1e-8, Weight Decay 1e-4 |

The exam cites 0.10 and 0.15 as examples without requiring identical learning rates across optimizers. Adam defaults to LR=0.001 in [PyTorch documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html); evaluating a localized range around 1e-3 is empirically sound.

Initial benchmark length: 200 epochs with Cosine Annealing. Brief 1–5 epoch sanity runs should verify gradient stability and data pipelines prior to full runs.

Standard augmentation baseline: RandomCrop (padding 4, reflection) + RandomHorizontalFlip + Cutout (16×16). Regularization options (Mixup, CutMix, Label Smoothing) should be evaluated through single-factor ablation against validation metrics.

---

## 6. Priority 2: Controlled Capacity Scaling for Model L

L v1 for CIFAR uses `cifar=True`, stem stride 1, and stage strides (1,1,2,2,2), producing feature map dimensions 32→32→32→16→8→4.

The classification head has 768 input channels and bias. When switching to $K$ classes, parameters scale by $769 \times (K - 5)$, and FLOPs scale by $1536 \times (K - 5)$:

| Candidate ($32 \times 32$ input) | CIFAR-10 Params | CIFAR-100 Params | CIFAR-100 GFLOPs |
|---|---:|---:|---:|
| L v1 (classifier swap only) | 1,100,105 | 1,169,315 | 0.157967 |
| L + 1 repeated Stage 4 block | 1,239,875 | 1,309,085 | 0.174383 |

Candidate Exploration Sequence:
1. **Candidate A: Stage 4 Depth 2 $\rightarrow$ 3**: Keeps all other factors constant. Compact modification (+139,770 params, +0.0164 GFLOPs), easy to defend theoretically.
2. **Candidate B: Relax Hidden Ratio in Stages 3–5 (0.75 $\rightarrow$ 1.0)**: Tests whether aggressive compression limits representational richness on 100 classes.
3. **Candidate C: Channel Width Scaling in Stages 3–4 (144/288 $\rightarrow$ 192/384)**: Supported by [Wide Residual Networks](https://arxiv.org/abs/1605.07146) principles.
4. **Candidate D: Coordinate Attention (CA) Ablation**: Evaluated only after establishing baseline performance under controlled comparisons.

A/B/C/D remain experimental hypotheses. Modifications should be isolated and verified against validation sets before composite deployment.

---

## 7. Execution Roadmap & Decision Criteria

| Phase | Deliverables | Decision Rule |
|---|---|---|
| Standardization | CIFAR dataloaders, validation split, SGD/Adam, resume, profiling | Shape/label integrity; held-out test isolation |
| CIFAR Benchmark | L v1 evaluated across SGD/Adam and target learning rates | Checkpoints selected on validation metrics |
| Recipe Optimization | Data augmentation, weight decay, warmup tuning | Retain only reproducible validation gains |
| Architecture Ablation | Candidates A/B/C/D tested under identical recipes | Evaluate accuracy relative to computational overhead |
| Final Verification | Selected configuration evaluated across seeds 42, 43, 44 | Report mean ± std; never select seeds using test data |
| Final Submission | Technical PDF report, reproducible checkpoints/code, oral defense preparation | Defend architectural rationale and document limitations |

Initial grid search: 2 datasets × 2 optimizers × 2–3 learning rates. Pilot runs estimate compute requirements before expanding searches. When constrained by compute, prioritize L v1 under solid protocols and Candidate A.

Final model selection is governed by Validation Top-1 across both datasets. Report Macro F1 and minority-class metrics for diagnostic depth.

---

## 8. Clarifications to Historical Documentation

- In historical drafts, certain reported per-class metrics exhibited minor discrepancies with raw CSV tables. For official PDF reporting, verified CSV numbers take precedence over prose approximations.
- Speculative assertions that "L is guaranteed to generalize better on CIFAR-100 solely due to fewer parameters" or "trains substantially faster due to lower FLOPs" should be framed as hypotheses pending empirical validation.
- Model L should be consistently designated as `TickNet-L v1`, a proposed architectural variant of TickNet-Basic, acknowledging prior works on SE attention, residual learning, and mixed depthwise convolutions.
