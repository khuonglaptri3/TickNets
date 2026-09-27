"""Contracts: paired splits, repeatability, source preservation, and safe failure."""
import csv
import hashlib
import importlib
import json
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image


CLASSES = ("bird", "cat", "dog", "frog", "horse")


@pytest.fixture
def source_dataset(tmp_path):
    root = tmp_path / "source"
    for variant, size in (("mid224", 224), ("Mid32", 32)):
        for label_index, label in enumerate(CLASSES):
            for index in range(6):
                split = "" if label in ("bird", "frog") else ("test" if index < 2 else "train")
                path = root / variant / label / split / f"{label}_{index}.jpeg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (size, size), (label_index * 45, index * 35, 91)).save(path, quality=90)
    return root


def prepare(source, output, **kwargs):
    module = importlib.import_module("prepare_mid_dataset")
    return module.prepare_dataset(source, output, train_per_class=4, test_per_class=2,
                                  workers=2, **kwargs)


def manifest(root):
    with (root / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepared_images_are_paired_balanced_and_byte_identical(source_dataset, tmp_path):
    output = tmp_path / "prepared"
    summary = prepare(source_dataset, output, seed=42, make_archives=True, archive_format="zip")
    rows = manifest(output)
    assert len(rows) == 30
    assert summary["class_to_idx"] == {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}
    for variant in ("Mid32", "Mid224"):
        for label in CLASSES:
            assert len(list((output / variant / "train" / label).glob("*.jpeg"))) == 4
            assert len(list((output / variant / "test" / label).glob("*.jpeg"))) == 2
        with ZipFile(output / f"{variant}.zip") as archive:
            assert len(archive.namelist()) == 30
            assert archive.testzip() is None
    for row in rows:
        for variant in ("mid32", "mid224"):
            original = source_dataset / row[f"{variant}_source"]
            copied = output / row[f"{variant}_path"]
            assert original.read_bytes() == copied.read_bytes()
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == row[f"{variant}_sha256"]
        assert Path(row["mid32_path"]).parts[1:] == Path(row["mid224_path"]).parts[1:]
    assert (output / "DATASET_SPLIT.md").is_file()
    assert json.loads((output / "split_config.json").read_text(encoding="utf-8"))["seed"] == 42


def test_same_seed_reproduces_manifest_and_another_seed_changes_membership(source_dataset, tmp_path):
    for name, seed in (("first", 42), ("second", 42), ("third", 43)):
        prepare(source_dataset, tmp_path / name, seed=seed)
    assert (tmp_path / "first" / "split_manifest.csv").read_bytes() == (tmp_path / "second" / "split_manifest.csv").read_bytes()
    choices = lambda root: {r["sample_id"] for r in manifest(root) if r["split"] == "test"}
    assert choices(tmp_path / "first") != choices(tmp_path / "third")


def test_reshuffle_all_does_not_lock_original_test_membership(source_dataset, tmp_path):
    output = tmp_path / "reshuffled"
    prepare(source_dataset, output, seed=42)
    rows = manifest(output)
    assert any(r["class_name"] == "cat" and r["split"] != r["original_split"] for r in rows)
    assert sum(r["split"] == "test" for r in rows) == 10


def test_missing_counterpart_fails_before_creating_output(source_dataset, tmp_path):
    (source_dataset / "mid224" / "bird" / "bird_0.jpeg").unlink()
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="pair|counterpart|matching"):
        prepare(source_dataset, output)
    assert not output.exists()


def test_existing_destination_is_never_overwritten(source_dataset, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare(source_dataset, output)
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_wrong_resolution_is_rejected(source_dataset, tmp_path):
    image = source_dataset / "Mid32" / "bird" / "bird_0.jpeg"
    Image.new("RGB", (31, 32), "red").save(image)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="32|size|dimension"):
        prepare(source_dataset, output)
    assert not output.exists()


def test_identical_pixels_across_train_test_are_rejected(source_dataset, tmp_path):
    original = source_dataset / "Mid32" / "cat" / "test" / "cat_0.jpeg"
    data = original.read_bytes()
    for duplicate in (source_dataset / "Mid32" / "cat").rglob("*.jpeg"):
        duplicate.write_bytes(data)
    with pytest.raises(ValueError, match="leak|duplicate|overlap"):
        prepare(source_dataset, tmp_path / "prepared")


def test_destination_inside_source_is_rejected(source_dataset):
    with pytest.raises(ValueError, match="overlap|inside|source"):
        prepare(source_dataset, source_dataset / "prepared")


def test_solid_tar_xz_contains_all_original_bytes(source_dataset, tmp_path):
    output = tmp_path / "solid"
    summary = prepare(source_dataset, output, seed=42, make_archives=True, archive_format="tar.xz")
    rows = manifest(output)
    assert summary["archive_format"] == "tar.xz"
    for variant in ("Mid32", "Mid224"):
        with tarfile.open(output / f"{variant}.tar.xz", "r:xz") as archive:
            expected = {row[f"{variant.lower()}_path"] for row in rows}
            assert set(archive.getnames()) == expected
            for member in archive:
                assert member.isfile()
                assert archive.extractfile(member).read() == (output / member.name).read_bytes()
