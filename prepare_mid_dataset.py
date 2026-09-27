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
    archives_note = (f"Gói {archive_format.upper()} chỉ chứa cây thư mục ảnh của từng phiên bản; metadata nằm ngoài gói nén."
                     if config["archive_bytes"] else "Lần tạo này không tạo gói nén.")
    return f"""# Mid32 / Mid224 — mô tả dữ liệu và cách chia

## 1. Mục tiêu và nguồn dữ liệu

Dataset phân loại 5 lớp: bird, cat, dog, frog, horse. Hai phiên bản có cùng
{len(CLASSES) * total:,} mã mẫu; mỗi mã mẫu có một ảnh 32x32 và một ảnh 224x224.
Nguồn khi tạo: `{config['source_root']}`.
Đầu ra: `{config['output_root']}`.

Ảnh được sao chép nguyên byte JPEG, không resize hoặc nén lại. Tất cả ảnh đã
được giải mã để kiểm tra JPEG, RGB, kích thước và kiểm tra SHA-256 sau sao chép.

## 2. Phương pháp chia: stratified random hold-out theo lớp

- **Seed: {config['seed']}**; chia lại toàn bộ 5 lớp, gộp các thư mục train/test cũ
  trước khi chọn mẫu. Split cũ chỉ được giữ trong manifest để truy vết.
- Mỗi lớp chọn ngẫu nhiên không hoàn lại đúng **{test_count} ảnh test**; các ảnh
  còn lại là **{train_count} ảnh train**. Tỷ lệ là {100 * train_count / total:.6f}% train
  và {100 * test_count / total:.6f}% test, xuất phát từ số lượng đề bài quy định.
- Duyệt lớp theo thứ tự `bird, cat, dog, frog, horse` và sắp xếp mã mẫu trong
  từng lớp theo thứ tự chuỗi Python trước khi lấy mẫu.
- Khởi tạo **một** `random.Random({config['seed']})`; gọi `rng.sample(sorted_ids,
  {test_count})` lần lượt cho từng lớp. Không khởi tạo lại RNG giữa các lớp.
- Mã mẫu là `class_name/filename`; tên giống nhau ở hai độ phân giải là một cặp.
  Chọn split một lần rồi áp dụng cho cả hai phiên bản. Không chia độc lập từng bản.
- Đây là cách chia phân tầng theo nhãn với số mẫu test cố định mỗi lớp, không
  phải lời khẳng định PyTorch quy định một tỷ lệ chia train/test bắt buộc.

## 3. Số lượng trong MỖI phiên bản

| Lớp | Class index | Train | Test | Tổng |
|---|---:|---:|---:|---:|
{counts}
| **Tổng** | | **{train_count * len(CLASSES):,}** | **{test_count * len(CLASSES):,}** | **{total * len(CLASSES):,}** |

Không tạo validation trong bản đóng gói. Nếu cần tuning, tách validation từ
train và giữ test độc lập; không dùng test để chọn seed, siêu tham số hay checkpoint.

## 4. Cấu trúc tương thích torchvision.datasets.ImageFolder

```text
data/
  Mid32/
    train/bird/*.jpeg    (tương tự cat, dog, frog, horse)
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

Truyền `data/Mid32/train` hoặc `data/Mid224/train` vào ImageFolder khi huấn luyện;
truyền thư mục test tương ứng khi đánh giá. Ánh xạ lớp giống nhau cho mọi split.

## 5. Manifest, kiểm tra và tái lập

`split_manifest.csv` có một dòng cho mỗi cặp ảnh: mã mẫu, nhãn, chỉ số nhãn,
split mới, split cũ, đường dẫn nguồn/đích, SHA-256 tệp, SHA-256 pixel và số byte
của từng phiên bản. **Manifest là danh sách phân chia chính thức** để dùng lại.

- Cặp ảnh Mid32/Mid224 có cùng split và tên lớp: đã kiểm tra.
- Train/test không giao nhau theo mã mẫu: đã kiểm tra.
- Ảnh trùng pixel chính xác nằm ở cả train và test: **0 nhóm** ở từng phiên bản.
- Kiểm tra trùng pixel không phát hiện được mọi ảnh gần trùng, crop hay cùng chủ thể;
  dữ liệu nguồn không cung cấp group ID để thực hiện group-aware split.
- Cùng tập nguồn, tên tệp, thuật toán và seed sẽ tạo lại cùng danh sách. Lưu
  manifest để tránh phụ thuộc vào thay đổi phiên bản thư viện trong tương lai.
- Python: {config['python_version']}; Pillow: {config['pillow_version']}.
- SHA-256 của manifest: `{config['manifest_sha256']}`.

Tái tạo vào một thư mục đích chưa tồn tại:

```powershell
python prepare_mid_dataset.py --source-root "{config['source_root']}" --output-root data_recreated --seed {config['seed']} --archive-format {archive_format}
```

## 6. Dung lượng

MB trong bảng = 1.000.000 byte, không phải dung lượng cấp phát trên ổ đĩa.

| Bản | Kích thước | Tổng JPEG (MB) | Gói nén (MB) |
|---|---|---:|---:|
{sizes}

{archives_note}
Giới hạn đề bài: Mid224 không quá 25M, Mid32 không quá 20M. Đối chiếu dung lượng
gói nén trong bảng khi nộp theo dạng nén; đề không định nghĩa rõ M/MB hay MiB.
TAR.XZ nén toàn bộ luồng TAR chung (solid), giúp loại bỏ phần lặp giữa nhiều
tệp nhỏ. Giải nén khôi phục nguyên byte JPEG. ZIP nén từng tệp riêng và có thêm
header cho từng ảnh, nên bản ZIP của bộ này lớn hơn giới hạn; dùng TAR.XZ để đóng gói.

Giải nén bằng 7-Zip/WinRAR hoặc lệnh `tar -xf Mid32.tar.xz` / `tar -xf Mid224.tar.xz`.

## 7. Bộ đọc và pipeline PyTorch

```powershell
conda activate fresher
python train_mid.py --data-root data --variant Mid32 --seed {config['seed']} --output-dir runs/mid32_seed{config['seed']}
python train_mid.py --data-root data --variant Mid224 --seed {config['seed']} --output-dir runs/mid224_seed{config['seed']}
```

Đây là lệnh huấn luyện để chạy sau khi chuẩn bị dữ liệu, không phải kết quả đã
huấn luyện. Mặc định dùng TickNet-basic với đầu ra 5 lớp; Mid32 sử dụng cấu hình
stride dành cho 32x32. Pipeline lưu cấu hình, log train từng epoch, checkpoint
cuối và kết quả test cuối quá trình; không chọn checkpoint theo test.

Train áp dụng random crop có padding và horizontal flip; test không augment.
ToTensor chuyển RGB về [0, 1]; không thêm Normalize vì TickNet hiện có data_bn.
Seed điều khiển Python, NumPy, Torch, sampler và worker. Kết quả huấn luyện vẫn
phụ thuộc môi trường/phần cứng; seed cố định không bảo đảm giống bit giữa mọi máy.

## 8. Tài liệu phương pháp

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
