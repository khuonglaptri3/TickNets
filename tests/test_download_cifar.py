"""Unit tests for scripts.download_cifar module."""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from scripts.download_cifar import (
    DATASET_METADATA,
    compute_md5,
    ensure_cifar_dataset,
    extract_tar_gz,
)


def test_dataset_metadata_urls_and_keys():
    assert "cifar10" in DATASET_METADATA
    assert "cifar100" in DATASET_METADATA

    c10 = DATASET_METADATA["cifar10"]
    assert c10["primary_url"] == "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
    assert c10["fallback_url"] == "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    assert c10["filename"] == "cifar-10-python.tar.gz"

    c100 = DATASET_METADATA["cifar100"]
    assert c100["primary_url"] == "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz".replace("10", "100")
    assert c100["fallback_url"] == "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    assert c100["filename"] == "cifar-100-python.tar.gz"


def test_compute_md5(tmp_path: Path):
    dummy_file = tmp_path / "sample.txt"
    dummy_file.write_text("TickNet CI Test\n", encoding="utf-8")
    md5_hash = compute_md5(dummy_file)
    assert isinstance(md5_hash, str)
    assert len(md5_hash) == 32


def test_ensure_cifar_dataset_already_present(tmp_path: Path):
    fake_extracted = tmp_path / "cifar-10-batches-py"
    fake_extracted.mkdir(parents=True)
    (fake_extracted / "batches.meta").write_text("meta", encoding="utf-8")

    result = ensure_cifar_dataset("cifar10", data_root=tmp_path, force=False)
    assert result["status"] == "already_present"
    assert result["dataset"] == "cifar10"


def test_extract_tar_gz(tmp_path: Path):
    content_dir = tmp_path / "payload"
    content_dir.mkdir()
    (content_dir / "sample.bin").write_bytes(b"12345")

    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(content_dir, arcname="payload")

    dest_dir = tmp_path / "unpacked"
    dest_dir.mkdir()
    extract_tar_gz(archive_path, dest_dir)

    assert (dest_dir / "payload" / "sample.bin").is_file()
    assert (dest_dir / "payload" / "sample.bin").read_bytes() == b"12345"
