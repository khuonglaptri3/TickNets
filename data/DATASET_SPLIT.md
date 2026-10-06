# Mid32 / Mid224 — Dataset Specification and Splitting Methodology

## 1. Objectives & Data Sources

5-class image classification dataset: `bird`, `cat`, `dog`, `frog`, `horse`. Both dataset versions share identical 25,250 sample identifiers; each sample identifier contains one 32×32 image and one 224×224 image.
Source path during creation: `C:\Users\lanph\Downloads\Final_Dataset_`.
Output destination: `data/`.

Images are preserved with raw JPEG byte streams without resizing or secondary re-compression. All images were decoded to verify JPEG headers, RGB color channels, dimensions, and post-copy SHA-256 checksums.

## 2. Partitioning Methodology: Per-Class Stratified Random Hold-Out

- **Seed: 42**; re-partitions all 5 classes, consolidating legacy train/test folders prior to sampling. Legacy split assignments are retained solely in the manifest for provenance tracking.
- Random sampling without replacement selects exactly **50 test images per class**; remaining images comprise **5,000 training images per class**. Yields 99.009901% train and 0.990099% test splits, derived from requirements specified in the midterm prompt.
- Iterates over classes in alphabetical order (`bird`, `cat`, `dog`, `frog`, `horse`), sorting sample identifiers within each class lexicographically in Python prior to sampling.
- Instantiates **a single** `random.Random(42)`; calls `rng.sample(sorted_ids, 50)` sequentially across classes without re-seeding between classes.
- Sample identifiers follow `class_name/filename`; identical filenames across resolutions constitute matched pairs. Splits are decided once and mirrored across both resolutions.
- This represents per-class stratified partitioning with fixed test quotas, rather than an assertion of a mandated PyTorch split ratio.

## 3. Sample Counts in EACH Dataset Version

| Class | Class Index | Train Samples | Test Samples | Total |
|---|---:|---:|---:|---:|
| bird | 0 | 5,000 | 50 | 5,050 |
| cat | 1 | 5,000 | 50 | 5,050 |
| dog | 2 | 5,000 | 50 | 5,050 |
| frog | 3 | 5,000 | 50 | 5,050 |
| horse | 4 | 5,000 | 50 | 5,050 |
| **Total** | | **25,000** | **250** | **25,250** |

No validation split is baked into the archive bundles. For hyperparameter tuning, separate validation subsets from training data while keeping the test set isolated; never select seeds, hyperparameters, or checkpoints using the test set.

## 4. Directory Structure Compatible with `torchvision.datasets.ImageFolder`

```text
data/
  Mid32/
    train/bird/*.jpeg    (likewise for cat, dog, frog, horse)
    test/bird/*.jpeg
  Mid224/
    train/bird/*.jpeg
    test/bird/*.jpeg
  split_manifest.csv
  split_config.json
  DATASET_SPLIT.md
  Mid32.tar.xz
  Mid224.tar.xz
```

Pass `data/Mid32/train` or `data/Mid224/train` to `ImageFolder` during training; pass respective test directories during evaluation. Class label mappings remain uniform across all splits.

## 5. Manifest, Verification, and Reproducibility

`split_manifest.csv` contains a single row per image pair: sample identifier, class label, class index, new split, legacy split, source/destination paths, file SHA-256, pixel SHA-256, and byte size per resolution. **The manifest is the authoritative split record** for downstream replication.

- Mid32/Mid224 image pairs share identical splits and class labels: verified.
- Train and test splits are completely disjoint by sample ID: verified.
- Identical pixel collisions across train and test: **0 instances** in both versions.
- Exact pixel hashing does not detect near-duplicates, crops, or identical subjects; raw source metadata lacked group IDs for group-aware splitting.
- Re-executing with identical sources, filenames, algorithm, and seed reproduces identical manifests.
- Python: 3.11.16; Pillow: 12.3.0.
- Manifest SHA-256: `9939a6ee404c6fbbdbe1b07a50763dc6d606497709385708512e1e98d71f2780`.

Recreating into an empty destination directory:

```powershell
python prepare_mid_dataset.py --source-root "C:\Users\lanph\Downloads\Final_Dataset_" --output-root data_recreated --seed 42 --archive-format tar.xz
```

## 6. Archive Sizing & Compression

MB values in this table = 1,000,000 bytes (decimal standard), rather than disk cluster allocation sizes.

| Dataset Version | Resolution | Total Raw JPEG (MB) | Compressed Archive (MB) |
|---|---|---:|---:|
| Mid32 | 32 x 32 | 19.0571 | 12.3062 |
| Mid224 | 224 x 224 | 31.8035 | 24.6096 |

The TAR.XZ archives contain only image directory hierarchies; metadata files reside externally.
Prompt Constraints: Mid224 under 25M, Mid32 under 20M. TAR.XZ solid stream compression eliminates redundant headers across small files, restoring raw JPEG byte streams upon extraction. ZIP packaging exceeds size limits due to per-file headers; TAR.XZ is mandated.
Extract using 7-Zip, WinRAR, or `tar -xf Mid32.tar.xz` / `tar -xf Mid224.tar.xz`.

## 7. PyTorch Dataloaders & Training Pipelines

```powershell
conda activate fresher
python train_mid.py --data-root data --variant Mid32 --seed 42 --output-dir runs/mid32_seed42
python train_mid.py --data-root data --variant Mid224 --seed 42 --output-dir runs/mid224_seed42
```

Defaults to TickNet-Basic with a 5-class head; Mid32 uses 32×32 stride schedules. Pipeline logs training configs, per-epoch metrics, final checkpoints, and end-of-run test metrics; checkpoint selection never uses test data.

Training applies random crops with padding and horizontal flips; evaluation applies no augmentations. `ToTensor()` scales RGB values to $[0, 1]$; manual normalization is omitted because TickNet incorporates internal `data_bn`.

## 8. Methodology References

- [ImageFolder — PyTorch](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- [Stratification — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)
- [Reproducibility — PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html)
