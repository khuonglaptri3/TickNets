"""Main orchestrator for dataset cleaning, quality assessment, and object filtering."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from cleaning.deduplication import compute_phash, find_duplicates
from cleaning.quality_filter import assess_image_quality
from cleaning.object_filter import ObjectVerifier


CLASSES = ("bird", "cat", "dog", "frog", "horse")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing dataset")
    parser.add_argument("--variant", choices=("Mid224", "Mid32"), default="Mid224", help="Dataset resolution variant")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all", help="Split to scan")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for fast audit")
    parser.add_argument("--skip-object-filter", action="store_true", help="Skip MobileNetV3 object verification")
    parser.add_argument("--output-dir", type=Path, default=Path("cleaning/reports"), help="Output directory for reports")
    return parser.parse_args(argv)


def discover_files(variant_dir: Path, split: str) -> List[Tuple[Path, str, str]]:
    """Discover all images in split/class hierarchy. Returns (path, split, class_name)."""
    splits = ["train", "test"] if split == "all" else [split]
    entries = []
    for s in splits:
        split_dir = variant_dir / s
        if not split_dir.is_dir():
            continue
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.jpeg")):
                entries.append((img_path, s, class_name))
    return entries


def run_cleaning_pipeline(
    data_root: Path,
    variant: str = "Mid224",
    split: str = "all",
    device: str = "auto",
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    skip_object_filter: bool = False,
    output_dir: Path = Path("cleaning/reports"),
) -> dict:
    variant_dir = data_root / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"Variant directory does not exist: {variant_dir}")

    files = discover_files(variant_dir, split)
    if max_samples:
        files = files[:max_samples]

    if not files:
        raise ValueError(f"No image files found in {variant_dir} for split={split}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting Data Cleaning Pipeline on {variant} ({split})")
    print(f"[*] Found {len(files)} image files to inspect")
    print(f"[*] Device setting: {device}")

    verifier = None
    if not skip_object_filter:
        print("[*] Loading MobileNetV3-Small ObjectVerifier...")
        verifier = ObjectVerifier(device=device)

    start_time = time.time()
    records = []
    
    # Process in batches
    for i in range(0, len(files), batch_size):
        chunk = files[i : i + batch_size]
        batch_images = []
        batch_targets = []
        batch_meta = []

        for path, s, class_name in chunk:
            sample_id = f"{s}/{class_name}/{path.name}"
            try:
                img = Image.open(path)
                img.load()
            except Exception as e:
                # Corrupt image
                records.append({
                    "sample_id": sample_id,
                    "path": str(path),
                    "split": s,
                    "class_name": class_name,
                    "phash": "",
                    "blur_score": 0.0,
                    "mean_luminance": 0.0,
                    "std_luminance": 0.0,
                    "flags": ["CORRUPT"],
                    "is_clean": False,
                    "object_check": {"target_prob": 0.0, "top1_label": "error", "is_suspicious": True},
                })
                continue

            q_report = assess_image_quality(img, sample_id=sample_id)
            phash_str = compute_phash(img)

            batch_images.append(img)
            batch_targets.append(class_name)
            batch_meta.append((path, s, class_name, sample_id, phash_str, q_report))

        # Run object verification on loaded batch
        obj_results = [None] * len(batch_images)
        if verifier and batch_images:
            try:
                obj_results = verifier.verify_batch(batch_images, batch_targets)
            except Exception as e:
                print(f"[!] Warning: Object verification failed for batch {i}: {e}")

        # Assemble records
        for (path, s, class_name, sample_id, phash_str, q_report), obj_res in zip(batch_meta, obj_results):
            flags = list(q_report.flags)
            if obj_res and obj_res.get("is_suspicious"):
                flags.append("SUSPICIOUS_LABEL")

            records.append({
                "sample_id": sample_id,
                "path": str(path),
                "split": s,
                "class_name": class_name,
                "phash": phash_str,
                "blur_score": q_report.blur_score,
                "mean_luminance": q_report.mean_luminance,
                "std_luminance": q_report.std_luminance,
                "flags": flags,
                "is_clean": len(flags) == 0,
                "object_check": obj_res or {},
            })

        if (i + len(chunk)) % 1000 == 0 or (i + len(chunk)) == len(files):
            elapsed = time.time() - start_time
            print(f"    Inspected {i + len(chunk)}/{len(files)} images ({elapsed:.1f}s)")

    # Run Deduplication Analysis
    valid_hash_samples = [r for r in records if r["phash"]]
    dup_report = find_duplicates(valid_hash_samples, near_dup_threshold=4, hash_key="phash")

    # Mark exact duplicate samples with flag
    exact_dup_ids = set()
    for group in dup_report["exact_duplicate_groups"].values():
        # Keep first as canonical, mark remaining as duplicates
        for dup_id in group[1:]:
            exact_dup_ids.add(dup_id)

    flag_counts = Counter()
    clean_count = 0
    flagged_records = []

    for r in records:
        if r["sample_id"] in exact_dup_ids and "DUPLICATE_PHASH" not in r["flags"]:
            r["flags"].append("DUPLICATE_PHASH")
            r["is_clean"] = False

        if r["is_clean"]:
            clean_count += 1
        else:
            flagged_records.append(r)

        for f in r["flags"]:
            flag_counts[f] += 1

    summary = {
        "variant": variant,
        "split": split,
        "total_scanned": len(records),
        "clean_samples": clean_count,
        "clean_ratio_percent": round(100.0 * clean_count / max(1, len(records)), 2),
        "flagged_samples": len(flagged_records),
        "flag_breakdown": dict(flag_counts),
        "deduplication": {
            "exact_duplicate_groups": len(dup_report["exact_duplicate_groups"]),
            "exact_duplicate_samples_flagged": len(exact_dup_ids),
            "near_duplicate_pairs": dup_report["near_duplicate_pair_count"],
        },
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    # Save summary report JSON
    report_json_path = output_dir / f"cleaning_summary_{variant.lower()}_{split}.json"
    report_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save flagged samples CSV
    flagged_csv_path = output_dir / f"flagged_samples_{variant.lower()}_{split}.csv"
    with flagged_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "class_name", "split", "flags", "blur_score",
                      "mean_luminance", "std_luminance", "target_prob", "top1_prediction", "top1_prob", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flagged_records:
            obj = r.get("object_check", {})
            writer.writerow({
                "sample_id": r["sample_id"],
                "class_name": r["class_name"],
                "split": r["split"],
                "flags": ";".join(r["flags"]),
                "blur_score": r["blur_score"],
                "mean_luminance": r["mean_luminance"],
                "std_luminance": r["std_luminance"],
                "target_prob": obj.get("target_prob", ""),
                "top1_prediction": obj.get("top1_label", ""),
                "top1_prob": obj.get("top1_prob", ""),
                "path": r["path"],
            })

    print("\n" + "=" * 60)
    print(f"DATA CLEANING AUDIT COMPLETE for {variant} ({split})")
    print("=" * 60)
    print(f"Total Images Scanned : {summary['total_scanned']:,}")
    print(f"Clean Images         : {summary['clean_samples']:,} ({summary['clean_ratio_percent']}%)")
    print(f"Flagged Images       : {summary['flagged_samples']:,}")
    print("Flags Breakdown:")
    for flag, cnt in flag_counts.most_common():
        print(f"  - {flag:<20}: {cnt:,}")
    print(f"Deduplication (pHash):")
    print(f"  - Exact Dup Groups : {summary['deduplication']['exact_duplicate_groups']:,}")
    print(f"  - Near Dup Pairs   : {summary['deduplication']['near_duplicate_pairs']:,}")
    print(f"Reports saved to     : {output_dir}")
    print("=" * 60)

    return summary


def main():
    args = parse_args()
    run_cleaning_pipeline(
        data_root=args.data_root,
        variant=args.variant,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        skip_object_filter=args.skip_object_filter,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
