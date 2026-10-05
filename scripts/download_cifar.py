"""Automated downloader and integrity checker for CIFAR-10 and CIFAR-100 datasets."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import ssl
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "cifar10": {
        "filename": "cifar-10-python.tar.gz",
        "primary_url": "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
        "fallback_url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "extracted_dir": "cifar-10-batches-py",
    },
    "cifar100": {
        "filename": "cifar-100-python.tar.gz",
        "primary_url": "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz",
        "fallback_url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "extracted_dir": "cifar-100-python",
    },
}

# Official extracted-file checksums, matching torchvision.datasets.CIFAR10/100.
EXTRACTED_CHECKSUMS = {
    "cifar10": {
        "data_batch_1": "c99cafc152244af753f735de768cd75f",
        "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
        "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
        "data_batch_4": "634d18415352ddfa80567beed471001a",
        "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
        "test_batch": "40351d587109b95175f43aff81a1287e",
        "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    },
    "cifar100": {"train": "16019d7e3df5f24257cddd939b257f8d",
                 "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
                 "meta": "7973b15100ade9c7d40fb424638fde48"},
}


def extracted_dataset_valid(name: str, data_root: Path) -> bool:
    folder = data_root / DATASET_METADATA[name]["extracted_dir"]
    return all((folder / filename).is_file() and compute_md5(folder / filename) == expected
               for filename, expected in EXTRACTED_CHECKSUMS[name].items())


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_ssl_context() -> Tuple[ssl.SSLContext, bool]:
    """Create SSL context, falling back to unverified if local cert store lacks Toronto intermediate cert."""
    try:
        ctx = ssl.create_default_context()
        return ctx, True
    except Exception:
        ctx = ssl._create_unverified_context()
        return ctx, False


def download_file_with_fallback(urls: List[str], target_path: Path, expected_md5: Optional[str] = None) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_path.with_suffix(".tmp")

    last_error: Optional[Exception] = None

    for url in urls:
        print(f"[*] Attempting download from: {url}")
        for verify_ssl in (True, False):
            try:
                ctx = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (TickNets-CI-Downloader)"})
                
                with opener.open(req, timeout=60) as response, temp_target.open("wb") as out_file:
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    chunk_size = 1024 * 1024  # 1MB chunks

                    while chunk := response.read(chunk_size):
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded / total_size * 100.0
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r    -> Downloading: {mb_down:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
                        else:
                            mb_down = downloaded / (1024 * 1024)
                            print(f"\r    -> Downloading: {mb_down:.1f} MB", end="", flush=True)

                    print()  # newline after completion

                if expected_md5:
                    actual_md5 = compute_md5(temp_target)
                    if actual_md5.lower() != expected_md5.lower():
                        raise ValueError(f"MD5 mismatch for {url}: expected {expected_md5}, got {actual_md5}")

                temp_target.replace(target_path)
                print(f"[+] Successfully downloaded and verified: {target_path.name}")
                return target_path

            except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError, ValueError) as err:
                last_error = err
                if temp_target.exists():
                    temp_target.unlink()
                print(f"    [!] Failed with SSL verify={verify_ssl}: {err}")
                continue

    raise RuntimeError(f"All download attempts failed for {urls}. Last error: {last_error}")


def extract_tar_gz(archive_path: Path, extract_dir: Path) -> Path:
    print(f"[*] Extracting {archive_path.name} to {extract_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        destination = extract_dir.resolve()
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
        tar.extractall(path=destination, members=members, filter="data")
    print(f"[+] Extracted: {archive_path.name}")
    return extract_dir


def ensure_cifar_dataset(name: str, data_root: Path, force: bool = False) -> Dict[str, str]:
    if name not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset {name!r}. Choices: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[name]
    archive_file = data_root / meta["filename"]
    extracted_path = data_root / meta["extracted_dir"]

    # Check if already present and valid
    if not force and extracted_dataset_valid(name, data_root):
        print(f"[✓] {name.upper()} already extracted and present at: {extracted_path}")
        return {"dataset": name, "status": "already_present", "path": str(extracted_path)}

    urls = [meta["primary_url"], meta["fallback_url"]]

    # Download archive if not present or corrupt
    if force or not archive_file.is_file() or compute_md5(archive_file) != meta["md5"]:
        download_file_with_fallback(urls, archive_file, expected_md5=meta["md5"])

    # Extract archive
    extract_tar_gz(archive_file, data_root)
    if not extracted_dataset_valid(name, data_root):
        raise ValueError(f"Extracted {name} files failed official MD5 verification")

    return {"dataset": name, "status": "downloaded_and_extracted", "path": str(extracted_path)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify CIFAR datasets for CI/CD and training.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory to save datasets (default: data)")
    parser.add_argument("--dataset", choices=("all", "cifar10", "cifar100"), default="all", help="Dataset to download")
    parser.add_argument("--force", action="store_true", help="Force redownload even if present")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    targets = ["cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]
    print(f"=== CIFAR Automated Downloader ===")
    print(f"Destination: {data_root}")
    print(f"Target datasets: {targets}\n")

    for target in targets:
        ensure_cifar_dataset(target, data_root, force=args.force)

    print("\n[✓] All target CIFAR datasets are verified and ready for training!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
