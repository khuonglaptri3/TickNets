# TickNet Results & Benchmarks Summary

This document summarizes comparative experimental results between the baseline **TickNet-Basic** (author's original model) and the proposed **TickNet-L v1** (Model L selected for the Midterm and Final Examination projects).

| Model Architecture | Top-1 Mid32 | Top-1 Mid224 | Parameters | GFLOPs Mid32 | GFLOPs Mid224 | Project Status |
|---|---:|---:|---:|---:|---:|:---:|
| **TickNet-Basic** (Author Baseline) | 89.6% | 92.4% | 1,062,223 | 0.158428 | 0.988343 *(near 1G budget)* | Benchmark Baseline |
| **TickNet-L v1** (Proposed) | **91.6%** | **95.6%** | 1,096,260 | **0.157821** | **0.796760** *(lightest)* | **Selected for Development** |

FLOPs convention: 1 MAC = 2 FLOPs, single image, eval mode, Conv2d and Linear layers only.

- **[Master Evidence & Dossier for Model L (Selected for Development)](Model_L/README.md)**
- **[Master Evidence & Dossier for Baseline Model Basic](Model_Basic/README.md)**
- [Independent Forensic Audit Report (2026-10-01)](AUDIT_TICKNET_2026-10-01.md)
- [Comprehensive Comparative Benchmarks](model_comparison/README.md)
- [Checkpoints and SHA-256 Checksums](checkpoints/README.md)
- [Independent Verification Scripts](audit_20261001/)
