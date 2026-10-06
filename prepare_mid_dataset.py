"""Create paired, seeded ImageFolder datasets without re-encoding source JPEGs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import tarfile
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, __version__ as pillow_version

CLASSES = ("bird", "cat", "dog", "frog", "horse")
VARIANTS = {"Mid32": 32, "Mid224": 224}


def discover_samples(source_root: Path, expected_per_class: int) -> list[dict]:
    """Pair by class/filename, independent of the original train/test directory."""
    inventories = {}
    directories = {p.name.casefold(): p for p in source_root.iterdir() if p.is_dir()}
    for variant in VARIANTS:
        base = directories.get(variant.casefold())
        if base is None:
            raise ValueError(f"Missing source variant: {variant}")
        classes = sorted(p.name for p in base.iterdir() if p.is_dir())
        if classes != list(CLASSES):
            raise ValueError(f"Unexpected classes in {base}: {classes}")
        inventory = {}
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            parts = path.relative_to(base).parts
            if path.suffix.lower() not in (".jpeg", ".jpg"):
                raise ValueError(f"Unexpected non-JPEG source file: {path}")
            if len(parts) == 2:
                old_split = "unsplit"
            elif len(parts) == 3 and parts[1] in ("train", "test"):
                old_split = parts[1]
            else:
                raise ValueError(f"Unsupported source layout: {path}")
            sample_id = f"{parts[0]}/{path.name}"
            if sample_id in inventory:
                raise ValueError(f"Duplicate class/filename identifier: {sample_id}")
            inventory[sample_id] = (path, old_split)
        inventories[variant] = inventory
    if set(inventories["Mid32"]) != set(inventories["Mid224"]):
        missing = sorted(set(inventories["Mid32"]) ^ set(inventories["Mid224"]))
        raise ValueError(f"Missing matching counterpart image(s): {missing[:5]}")
    counts = Counter(sample_id.split("/", 1)[0] for sample_id in inventories["Mid32"])
    if counts != Counter({label: expected_per_class for label in CLASSES}):
        raise ValueError(f"Wrong image counts per class: {dict(counts)}; expected {expected_per_class}")
    samples = []
    for sample_id in sorted(inventories["Mid32"]):
        label = sample_id.split("/", 1)[0]
        row = {"sample_id": sample_id, "class_name": label, "class_index": CLASSES.index(label),
               "original_split": inventories["Mid32"][sample_id][1]}
        for variant in VARIANTS:
            path, old_split = inventories[variant][sample_id]
            if old_split != row["original_split"]:
                raise ValueError(f"Original split mismatch between paired images: {sample_id}")
            row[f"{variant.lower()}_source"] = path.relative_to(source_root).as_posix()
        samples.append(row)
    return samples


def assign_splits(samples: list[dict], seed: int, test_per_class: int) -> list[dict]:
    """Stratified hold-out: sample an exact number uniformly within each class."""
    rng = random.Random(seed)
    selected = set()
    for label in CLASSES:
        identifiers = sorted(row["sample_id"] for row in samples if row["class_name"] == label)
        selected.update(rng.sample(identifiers, test_per_class))
    rows = []
    for sample in samples:
        row = dict(sample, split="test" if sample["sample_id"] in selected else "train")
        for variant in VARIANTS:
            row[f"{variant.lower()}_path"] = f"{variant}/{row['split']}/{row['sample_id']}"
        rows.append(row)
    return rows


def _copy_image(job):
    source, target, variant, sample_id, split = job
    data = source.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        required_size = (VARIANTS[variant], VARIANTS[variant])
        if image.format != "JPEG" or image.mode != "RGB" or image.size != required_size:
            raise ValueError(f"Invalid JPEG/RGB/image size at {source}; expected {required_size}")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    with target.open("xb") as handle:
        handle.write(data)
    digest = hashlib.sha256(data).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise IOError(f"Copied image checksum mismatch: {target}")
    return variant, sample_id, split, digest, pixel_hash, data


def write_archive_member(archive, name: str, data: bytes, archive_format: str) -> None:
    if archive_format == "zip":
        info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
    else:
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        archive.addfile(info, io.BytesIO(data))


def verify_archive(path: Path, variant: str, rows: list[dict], archive_format: str) -> None:
    expected = {row[f"{variant.lower()}_path"]: row[f"{variant.lower()}_sha256"] for row in rows}
    seen = set()
    with ExitStack() as stack:
        if archive_format == "zip":
            archive = stack.enter_context(ZipFile(path))
            members = ((info.filename, archive.read(info)) for info in archive.infolist())
        else:
            archive = stack.enter_context(tarfile.open(path, "r:xz"))
            members = ((info.name, archive.extractfile(info).read()) for info in archive if info.isfile())
        for name, data in members:
            if name in seen or name not in expected or hashlib.sha256(data).hexdigest() != expected[name]:
                raise IOError(f"Archive member verification failed: {path}: {name}")
            seen.add(name)
    if seen != set(expected):
        raise IOError(f"Archive has missing images: {path}")


def render_description(config: dict) -> str:
    train_count, test_count = config["train_per_class"], config["test_per_class"]
    total = train_count + test_count
    counts = "\n".join(f"| {name} | {index} | {train_count:,} | {test_count:,} | {total:,} |"
                       for name, index in config["class_to_idx"].items())
    sizes = "\n".join(
        f"| {variant} | {size} x {size} | {config['image_bytes'][variant] / 1e6:.4f} | "
        f"{config.get('archive_bytes', {}).get(variant, 0) / 1e6:.4f} |"
        for variant, size in VARIANTS.items())
    archive_format = config.get("archive_format") or "tar.xz"
    archives_note = (f"The {archive_format.upper()} archive contains only the image directory tree for each variant; metadata resides outside the archive."
                     if config["archive_bytes"] else "No compressed archives were generated during this run.")
    return f"""# Mid32 / Mid224 — Dataset Specification and Splitting Methodology

## 1. Objectives & Data Sources

5-class image classification dataset: bird, cat, dog, frog, horse. Both dataset versions share identical
{len(CLASSES) * total:,} sample identifiers; each sample identifier contains one 32x32 image and one 224x224 image.
Source path during creation: `{config['source_root']}`.
Output destination: `{config['output_root']}`.

Images are preserved with raw JPEG byte streams without resizing or secondary re-compression. All images were
decoded to verify JPEG headers, RGB color channels, dimensions, and post-copy SHA-256 checksums.

## 2. Partitioning Methodology: Per-Class Stratified Random Hold-Out

- **Seed: {config['seed']}**; re-partitions all 5 classes, consolidating legacy train/test folders
  prior to sampling. Legacy split assignments are retained solely in the manifest for provenance tracking.
- Random sampling without replacement selects exactly **{test_count} test images per class**; remaining images
  comprise **{train_count} training images per class**. Yields {100 * train_count / total:.6f}% train
  and {100 * test_count / total:.6f}% test splits, derived from requirements specified in the midterm prompt.
- Iterates over classes in alphabetical order (`bird, cat, dog, frog, horse`), sorting sample identifiers within
  each class lexicographically in Python prior to sampling.
- Instantiates **a single** `random.Random({config['seed']})`; calls `rng.sample(sorted_ids,
  {test_count})` sequentially across classes without re-seeding between classes.
- Sample identifiers follow `class_name/filename`; identical filenames across resolutions constitute matched pairs.
  Splits are decided once and mirrored across both resolutions.
- This represents per-class stratified partitioning with fixed test quotas, rather than an
  assertion of a mandated PyTorch split ratio.

## 3. Sample Counts in EACH Dataset Version

| Class | Class Index | Train | Test | Total |
|---|---:|---:|---:|---:|
{counts}
| **Total** | | **{train_count * len(CLASSES):,}** | **{test_count * len(CLASSES):,}** | **{total * len(CLASSES):,}** |

No validation split is baked into the archive bundles. For hyperparameter tuning, separate validation subsets from
training data while keeping the test set isolated; never select seeds, hyperparameters, or checkpoints using the test set.

## 4. Directory Structure Compatible with torchvision.datasets.ImageFolder

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
  Mid32.{archive_format}
  Mid224.{archive_format}
```

Pass `data/Mid32/train` or `data/Mid224/train` to ImageFolder during training; pass respective test directories
during evaluation. Class label mappings remain uniform across all splits.

## 5. Manifest, Verification, and Reproducibility

`split_manifest.csv` contains a single row per image pair: sample identifier, class label, class index,
new split, legacy split, source/destination paths, file SHA-256, pixel SHA-256, and byte size
per resolution. **The manifest is the authoritative split record** for downstream replication.

- Mid32/Mid224 image pairs share identical splits and class labels: verified.
- Train and test splits are completely disjoint by sample ID: verified.
- Identical pixel collisions across train and test: **0 instances** in both versions.
- Exact pixel hashing does not detect near-duplicates, crops, or identical subjects;
  raw source metadata lacked group IDs for group-aware splitting.
- Re-executing with identical sources, filenames, algorithm, and seed reproduces identical manifests.
  Retain the manifest to avoid future library version discrepancies.
- Python: {config['python_version']}; Pillow: {config['pillow_version']}.
- Manifest SHA-256: `{config['manifest_sha256']}`.

Recreating into an empty destination directory:

```powershell
python prepare_mid_dataset.py --source-root "{config['source_root']}" --output-root data_recreated --seed {config['seed']} --archive-format {archive_format}
```

## 6. Archive Sizing & Compression

MB values in this table = 1,000,000 bytes (decimal standard), rather than disk cluster allocation sizes.

| Dataset Version | Resolution | Total Raw JPEG (MB) | Compressed Archive (MB) |
|---|---|---:|---:|
{sizes}

{archives_note}
Prompt Constraints: Mid224 under 25M, Mid32 under 20M. Cross-reference archive sizes in the table
when submitting compressed packages; prompt does not specify M/MB vs MiB.
TAR.XZ solid stream compression eliminates redundant headers across small files, restoring raw JPEG byte streams
upon extraction. ZIP packaging exceeds size limits due to per-file headers; TAR.XZ is mandated.

Extract using 7-Zip, WinRAR, or `tar -xf Mid32.tar.xz` / `tar -xf Mid224.tar.xz`.

## 7. PyTorch Dataloaders & Training Pipelines

```powershell
conda activate fresher
python train_mid.py --data-root data --variant Mid32 --seed {config['seed']} --output-dir runs/mid32_seed{config['seed']}
python train_mid.py --data-root data --variant Mid224 --seed {config['seed']} --output-dir runs/mid224_seed{config['seed']}
```

These training commands should be run after dataset preparation, not pre-existing training artifacts.
Defaults to TickNet-Basic with a 5-class head; Mid32 uses 32x32 stride schedules.
Pipeline logs training configs, per-epoch metrics, final checkpoints, and end-of-run test metrics; checkpoint selection never uses test data.

Training applies random crops with padding and horizontal flips; evaluation applies no augmentations.
ToTensor() scales RGB values to [0, 1]; manual normalization is omitted because TickNet incorporates internal data_bn.
Seed governs Python, NumPy, Torch, sampler, and worker processes. Training dynamics remain subject to
hardware/runtime differences; a fixed seed does not guarantee bitwise determinism across different machines.

## 8. Methodology References

- [ImageFolder — PyTorch](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- [Stratification — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)
- [Reproducibility — PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html)
"""


def prepare_dataset(source_root, output_root, *, seed=42, train_per_class=5000,
                    test_per_class=50, workers=16, make_archives=False, archive_format="tar.xz") -> dict:
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if min(train_per_class, test_per_class, workers) < 1:
        raise ValueError("Train/test counts and workers must be positive")
    if make_archives and archive_format not in ("zip", "tar.xz"):
        raise ValueError("archive_format must be zip or tar.xz")
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Source and output paths must not overlap")
    if output_root.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {output_root}")
    rows = assign_splits(discover_samples(source_root, train_per_class + test_per_class), seed, test_per_class)
    # A failed run leaves a named staging directory for inspection, never a valid-looking output.
    staging = output_root.parent / f"{output_root.name}.building-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    for variant in VARIANTS:
        for split in ("train", "test"):
            for label in CLASSES:
                (staging / variant / split / label).mkdir(parents=True)
    by_id = {row["sample_id"]: row for row in rows}
    jobs = [(source_root / row[f"{variant.lower()}_source"], staging / row[f"{variant.lower()}_path"],
             variant, row["sample_id"], row["split"]) for row in rows for variant in VARIANTS]
    pixel_groups = {variant: defaultdict(set) for variant in VARIANTS}
    image_bytes = Counter()
    try:
        with ExitStack() as stack:
            archives = {}
            if make_archives:
                for variant in VARIANTS:
                    target = staging / f"{variant}.{archive_format}"
                    archive = (ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=9)
                               if archive_format == "zip" else tarfile.open(target, "w:xz", preset=6))
                    archives[variant] = stack.enter_context(archive)
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
            for index, (variant, sample_id, split, digest, pixel_hash, data) in enumerate(pool.map(_copy_image, jobs), 1):
                row = by_id[sample_id]
                prefix = variant.lower()
                row.update({f"{prefix}_sha256": digest, f"{prefix}_pixel_sha256": pixel_hash,
                            f"{prefix}_bytes": len(data)})
                image_bytes[variant] += len(data)
                pixel_groups[variant][pixel_hash].add(split)
                if make_archives:
                    write_archive_member(archives[variant], row[f"{prefix}_path"], data, archive_format)
                if index % 2500 == 0:
                    print(f"Copied, decoded and verified {index}/{len(jobs)} images", flush=True)
        leaks = {v: sum(len(splits) > 1 for splits in groups.values()) for v, groups in pixel_groups.items()}
        if any(leaks.values()):
            raise ValueError(f"Identical-pixel train/test leakage detected: {leaks}")
        archive_bytes = {}
        for variant in archives:
            archive_path = staging / f"{variant}.{archive_format}"
            verify_archive(archive_path, variant, rows, archive_format)
            archive_bytes[variant] = archive_path.stat().st_size
        manifest_path = staging / "split_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config = {"schema_version": 1, "status": "complete", "seed": seed,
                  "split_method": "stratified_random_holdout_all_classes",
                  "rng": "Python random.Random; one instance; sorted classes and sample IDs",
                  "train_per_class": train_per_class, "test_per_class": test_per_class,
                  "class_to_idx": {name: index for index, name in enumerate(CLASSES)},
                  "source_root": str(source_root), "output_root": str(output_root),
                  "python_version": platform.python_version(), "pillow_version": pillow_version,
                  "paired_samples": len(rows), "image_bytes": dict(image_bytes),
                  "archive_bytes": archive_bytes, "archive_format": archive_format if make_archives else None,
                  "cross_split_identical_pixel_groups": leaks,
                  "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        (staging / "split_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "DATASET_SPLIT.md").write_text(render_description(config), encoding="utf-8")
        staging.rename(output_root)
        return config
    except Exception:
        print(f"Preparation failed. Original data is intact; partial output: {staging}", flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--archive-format", choices=("tar.xz", "zip", "none"), default="tar.xz")
    args = parser.parse_args(argv)
    result = prepare_dataset(**vars(args), make_archives=args.archive_format != "none")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
