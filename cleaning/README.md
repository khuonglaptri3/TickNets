# Data Cleaning & Quality Assurance Pipeline

The `cleaning/` module provides auditing, deduplication, and quality assurance utilities for the `Mid224` and `Mid32` datasets based on three core technical pillars:

1. **Deduplication**: Perceptual Hashing (pHash, dHash)
2. **Quality Filtering**: Laplacian blur variance, exposure levels, and contrast metrics
3. **Object & Label Verification**: Lightweight MobileNetV3-Small (pretrained on ImageNet-1K) supporting both CPU and GPU execution

---

## 1. Methodology & Technical Criteria

### 1.1 Deduplication (pHash & dHash)

* **pHash (Perceptual Hash)**:
  - Resizes images to 32×32 grayscale and applies a 2D DCT (`scipy.fftpack.dct`).
  - Extracts the top-left 8×8 low-frequency matrix, comparing against the median coefficient to produce a 64-bit hash (16 hexadecimal characters).
  - Robust against slight scaling, minor luminance shifts, and JPEG quantization noise.
* **Hamming Distance**:
  - $\text{dist} = 0$: Exact perceptual duplicate.
  - $0 < \text{dist} \le 4$: Near-duplicate pair (subtle camera rotation, crop).

### 1.2 Quality Filtering

* **Blur Variance**:
  - Computes the variance of the Laplacian operator on grayscale images: $\sigma^2(\nabla^2 I)$.
  - Images with $\sigma^2 < 80$ (at 224×224 resolution) or $< 15$ (at 32×32 resolution) are flagged as `BLURRY`.
* **Exposure & Contrast**:
  - Mean luminance $\mu < 35$: Flagged as `UNDEREXPOSED` (overly dark).
  - Mean luminance $\mu > 220$: Flagged as `OVEREXPOSED` (blown-out highlights).
  - Luminance standard deviation $\sigma < 18$: Flagged as `LOW_CONTRAST` (flat tone / near-monochrome).

### 1.3 Object & Label Filtering

* Deploys **MobileNetV3-Small** (2.54M parameters, pretrained on ImageNet-1K):
  - Maps 1,000 ImageNet categories into the 5 target classes:
    - `bird`: 59 avian species
    - `dog`: 118 canine breeds
    - `cat`: 13 feline species
    - `frog`: 3 anuran species (`bullfrog`, `tree frog`, `tailed frog`)
    - `horse`: equine categories (`sorrel`, `zebra`)
  - Computes target class cumulative probability $P(\text{class} = c_{\text{target}})$.
  - If target probability $< 0.05$ while the model strongly predicts an alternative class ($P_{\text{top1}} > 0.20$), the sample is flagged as `SUSPICIOUS_LABEL`.

---

## 2. Audit Execution Guidelines

### 2.1 Quick Pilot Audit (250 Samples)

```bash
python -m cleaning.clean_dataset --variant Mid224 --split test --max-samples 250
```

### 2.2 Complete Dataset Audit (Train or Test)

```bash
# Audit test split (250 images per resolution)
python -m cleaning.clean_dataset --variant Mid224 --split test --output-dir cleaning/reports

# Audit complete training split (25,000 images)
python -m cleaning.clean_dataset --variant Mid224 --split train --batch-size 64 --device auto --output-dir cleaning/reports
```

### 2.3 Fast Mode (Skip Object Filter, pHash & Blur Only)

For rapid CPU execution (thousands of images in seconds):

```bash
python -m cleaning.clean_dataset --variant Mid224 --split train --skip-object-filter
```

---

## 3. Emitted Audit Artifacts

Report files are saved to `--output-dir` (defaults to `cleaning/reports/`):

- `cleaning_summary_<variant>_<split>.json`: Summary statistics including total scanned images, clean pass rate (%), flag distribution breakdowns, and duplicate clusters.
- `flagged_samples_<variant>_<split>.csv`: Detailed per-sample records including specific flag triggers (`BLURRY`, `UNDEREXPOSED`, `DUPLICATE_PHASH`, `SUSPICIOUS_LABEL`), metric scores, and file paths for inclusion in technical reports.
