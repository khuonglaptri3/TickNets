"""Real ImageFolder contracts for paired, leakage-free experiment splits."""
import hashlib
import importlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


CLASSES = ("bird", "cat", "dog", "frog", "horse")


@pytest.fixture
def paired_data(tmp_path):
    root = tmp_path / "data"
    for variant, size in (("Mid32", 32), ("Mid224", 224)):
        for split, count in (("train", 8), ("test", 2)):
            for label_index, label in enumerate(CLASSES):
                for index in range(count):
                    path = root / variant / split / label / f"{split}_{index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Asymmetric pixels expose accidental crop/flip augmentation.
                    pixels = np.zeros((size, size, 3), dtype=np.uint8)
                    pixels[:, :, 0] = np.arange(size, dtype=np.uint8)
                    pixels[:, :, 1] = label_index * 40
                    pixels[:, :, 2] = index * 25
                    Image.fromarray(pixels).save(path)
    return root


def build(root, variant="Mid32", **kwargs):
    return importlib.import_module("models.mid_experiment_data").build_experiment_loaders(
        root, variant, **kwargs
    )


def sample_keys(dataset):
    return [f"{dataset.classes[label]}/{Path(path).name}" for path, label in dataset.samples]


def test_paired_split_is_stratified_disjoint_exhaustive_and_root_independent(paired_data, tmp_path):
    relocated = tmp_path / "relocated"
    shutil.copytree(paired_data / "Mid224", relocated / "Mid224")
    first = build(paired_data, "Mid32", val_fraction=0.25, seed=42)
    second = build(relocated, "Mid224", val_fraction=0.25, seed=99)
    for train, validation, test, info in (first, second):
        assert info["class_to_idx"] == {"bird": 0, "cat": 1, "dog": 2, "frog": 3, "horse": 4}
        assert (info["train_samples"], info["val_samples"], info["test_samples"]) == (30, 10, 10)
        assert info["split_seed"] == 123
        assert info["val_fraction"] == 0.25
        assert sample_keys(train.dataset) == info["train_keys"]
        assert sample_keys(validation.dataset) == info["validation_keys"]
        assert sample_keys(test.dataset) == info["test_keys"]
        train_keys, val_keys, test_keys = map(set, (info["train_keys"], info["validation_keys"], info["test_keys"]))
        assert not train_keys & val_keys
        assert not (train_keys | val_keys) & test_keys
        assert train_keys | val_keys == {f"{label}/train_{i}.png" for label in CLASSES for i in range(8)}
        assert Counter(key.split("/")[0] for key in val_keys) == {label: 2 for label in CLASSES}
        assert Counter(key.split("/")[0] for key in train_keys) == {label: 6 for label in CLASSES}
        assert json.loads(json.dumps(info)) == info
    for field in ("train_keys", "validation_keys", "test_keys", "membership_sha256", "validation_sha256"):
        assert first[3][field] == second[3][field]
    assert next(iter(first[0]))[0].shape == (30, 3, 32, 32)
    assert next(iter(second[0]))[0].shape == (30, 3, 224, 224)


def test_split_seed_changes_validation_but_training_seed_only_changes_shuffle(paired_data):
    runs = [build(paired_data, seed=seed, split_seed=split, augment=False)
            for seed, split in ((42, 123), (42, 123), (43, 123), (42, 124))]
    orders = [list(iter(train.sampler)) for train, _, _, _ in runs]
    assert orders[0] == orders[1]
    assert orders[0] != orders[2]
    assert runs[0][3] == runs[1][3] == runs[2][3]
    assert runs[0][3]["validation_keys"] != runs[3][3]["validation_keys"]
    assert runs[0][3]["validation_sha256"] != runs[3][3]["validation_sha256"]
    assert runs[0][3]["membership_sha256"] == runs[3][3]["membership_sha256"]


@pytest.mark.parametrize("variant", ("Mid32", "Mid224"))
def test_validation_and_test_use_unaugmented_pixels_and_sequential_order(paired_data, variant):
    _, validation, test, _ = build(paired_data, variant, augment=True, batch_size=3)
    for loader in (validation, test):
        assert list(iter(loader.sampler)) == list(range(len(loader.dataset)))
        for index, (path, label) in enumerate(loader.dataset.samples):
            with Image.open(path) as image:
                expected = to_tensor(image.convert("RGB"))
            for rng_seed in (1, 52):
                torch.manual_seed(rng_seed)
                random.seed(rng_seed)
                np.random.seed(rng_seed)
                pixels, actual_label = loader.dataset[index]
                assert torch.equal(pixels, expected)
                assert actual_label == label
        batches = list(loader)
        assert sum(len(labels) for _, labels in batches) == len(loader.dataset)


def test_evaluation_iteration_cannot_advance_training_or_other_evaluation_generator(paired_data):
    train, validation, test, _ = build(paired_data, batch_size=3, augment=False)
    reference, _, _, _ = build(paired_data, batch_size=3, augment=False)
    assert train.generator.initial_seed() == 42
    assert len({id(loader.generator) for loader in (train, validation, test)}) == 3
    train_state = train.generator.get_state().clone()
    test_state = test.generator.get_state().clone()
    list(validation)
    assert torch.equal(train.generator.get_state(), train_state)
    assert torch.equal(test.generator.get_state(), test_state)
    list(test)
    assert torch.equal(train.generator.get_state(), train_state)
    assert list(iter(train.sampler)) == list(iter(reference.sampler))


@pytest.mark.parametrize("fraction", (0.0001, 0.9999))
def test_fraction_extremes_leave_at_least_one_sample_per_class_in_both_splits(paired_data, fraction):
    _, _, _, info = build(paired_data, val_fraction=fraction)
    expected = 1 if fraction < 0.5 else 7
    assert Counter(key.split("/")[0] for key in info["validation_keys"]) == {label: expected for label in CLASSES}
    assert Counter(key.split("/")[0] for key in info["train_keys"]) == {label: 8 - expected for label in CLASSES}


@pytest.mark.parametrize("fraction", (0, 1, -0.1, 1.1, float("nan"), float("inf")))
@pytest.mark.parametrize("full_train", (False, True))
def test_invalid_validation_fraction_is_rejected(paired_data, fraction, full_train):
    with pytest.raises(ValueError, match="fraction"):
        build(paired_data, val_fraction=fraction, full_train=full_train)


def test_full_train_has_no_validation_and_retains_original_membership(paired_data):
    train, validation, test, info = build(paired_data, full_train=True)
    partial = build(paired_data)[3]
    assert validation is None
    assert len(train.dataset) == info["train_samples"] == 40
    assert len(test.dataset) == info["test_samples"] == 10
    assert info["val_samples"] == 0
    assert info["validation_keys"] == []
    assert set(info["train_keys"]) == set(partial["train_keys"] + partial["validation_keys"])
    assert info["membership_sha256"] == partial["membership_sha256"]
    assert info["full_train"] is True
    assert partial["full_train"] is False


def test_singleton_class_requires_full_train(paired_data):
    for path in (paired_data / "Mid32" / "train" / "bird").glob("train_[1-7].png"):
        path.unlink()
    with pytest.raises(ValueError, match="bird|at least two"):
        build(paired_data)
    assert build(paired_data, full_train=True)[3]["train_samples"] == 33


def test_rejects_official_train_test_filename_leakage(paired_data):
    bird = paired_data / "Mid32" / "test" / "bird"
    (bird / "test_0.png").rename(bird / "train_0.png")
    with pytest.raises(ValueError, match="train and test|overlap|leak"):
        build(paired_data)


@pytest.mark.parametrize("split", ("train", "test"))
def test_rejects_class_mapping_drift(paired_data, split):
    folder = paired_data / "Mid32" / split
    (folder / "bird").rename(folder / "unexpected")
    with pytest.raises(ValueError, match="class"):
        build(paired_data)


def test_rejects_ambiguous_class_filename_keys_in_nested_folders(paired_data):
    bird = paired_data / "Mid32" / "train" / "bird"
    (bird / "nested").mkdir()
    shutil.copyfile(bird / "train_0.png", bird / "nested" / "train_0.png")
    with pytest.raises(ValueError, match="duplicate|ambiguous|unique"):
        build(paired_data)


@pytest.mark.parametrize("split", ("train", "test"))
def test_membership_fingerprint_detects_filename_changes_in_each_official_split(paired_data, split):
    before = build(paired_data)[3]
    folder = paired_data / "Mid32" / split / "bird"
    (folder / f"{split}_0.png").rename(folder / "renamed.png")
    after = build(paired_data)[3]
    assert before["membership_sha256"] != after["membership_sha256"]


def test_validation_fingerprint_is_sha256_of_canonical_sorted_keys(paired_data):
    info = build(paired_data)[3]
    payload = json.dumps(sorted(info["validation_keys"]), ensure_ascii=False, separators=(",", ":"))
    assert info["validation_sha256"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert len(info["membership_sha256"]) == 64
    assert info["variant"] == "Mid32"
    assert info["image_size"] == 32


@pytest.mark.parametrize("loader_index", (0, 1, 2))
def test_native_size_check_is_kept_for_every_loader(paired_data, loader_index):
    loader = build(paired_data)[loader_index]
    path = Path(loader.dataset.samples[0][0])
    Image.new("RGB", (31, 32)).save(path)
    with pytest.raises(ValueError, match="32x32"):
        loader.dataset[0]


@pytest.mark.parametrize("kwargs", ({"variant": "Wrong"}, {"batch_size": 0}, {"num_workers": -1}))
def test_existing_loader_argument_validation_is_preserved(paired_data, kwargs):
    with pytest.raises(ValueError):
        build(paired_data, **kwargs)
