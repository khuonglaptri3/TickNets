"""Tests for the dataset cleaning module."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
import pytest

from cleaning.deduplication import compute_phash, compute_dhash, hamming_distance, find_duplicates
from cleaning.quality_filter import assess_image_quality, compute_blur_score
from cleaning.object_filter import ObjectVerifier, TARGET_IMAGENET_MAP


def test_phash_identical_and_near_duplicates():
    # Create a smooth continuous gradient pattern
    x = np.linspace(0, 10, 224)
    y = np.linspace(0, 10, 224)
    xx, yy = np.meshgrid(x, y)
    arr = (np.sin(xx) * np.cos(yy) * 127 + 128).astype(np.uint8)
    arr_rgb = np.stack([arr, arr, arr], axis=-1)
    img1 = Image.fromarray(arr_rgb)

    # Identical image
    img2 = Image.fromarray(arr_rgb.copy())
    h1 = compute_phash(img1)
    h2 = compute_phash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Slight brightness shift
    img_slight = Image.fromarray(np.clip(arr_rgb.astype(int) + 10, 0, 255).astype(np.uint8))
    h_slight = compute_phash(img_slight)
    assert hamming_distance(h1, h_slight) <= 4

    # Completely different image (solid white)
    img_diff = Image.new("RGB", (224, 224), color=(255, 255, 255))
    h_diff = compute_phash(img_diff)
    assert hamming_distance(h1, h_diff) > 4


def test_find_duplicates():
    samples = [
        {"sample_id": "img1", "phash": "1111111111111111"},
        {"sample_id": "img2", "phash": "1111111111111111"}, # Exact duplicate of img1
        {"sample_id": "img3", "phash": "1111111111111113"}, # Near duplicate (1 bit diff)
        {"sample_id": "img4", "phash": "ffffffffffffffff"}, # Far away
    ]
    report = find_duplicates(samples, near_dup_threshold=4)
    assert "1111111111111111" in report["exact_duplicate_groups"]
    assert len(report["exact_duplicate_groups"]["1111111111111111"]) == 2
    assert report["near_duplicate_pair_count"] >= 1


def test_quality_filter_blur_detection():
    # Sharp image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sharp_img = Image.fromarray(arr)
    sharp_report = assess_image_quality(sharp_img, sample_id="sharp")
    assert not sharp_report.is_blurry

    # Heavily blurred image
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=15))
    blurry_report = assess_image_quality(blurry_img, sample_id="blurry", blur_threshold=80.0)
    assert blurry_report.is_blurry
    assert "BLURRY" in blurry_report.flags


def test_quality_filter_exposure_and_contrast():
    # Very dark image
    dark_img = Image.new("RGB", (224, 224), color=(10, 10, 10))
    dark_report = assess_image_quality(dark_img, sample_id="dark")
    assert dark_report.is_underexposed
    assert "UNDEREXPOSED" in dark_report.flags

    # Very bright image
    bright_img = Image.new("RGB", (224, 224), color=(245, 245, 245))
    bright_report = assess_image_quality(bright_img, sample_id="bright")
    assert bright_report.is_overexposed
    assert "OVEREXPOSED" in bright_report.flags

    # Flat image (zero variance)
    flat_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    flat_report = assess_image_quality(flat_img, sample_id="flat")
    assert flat_report.is_low_contrast
    assert "LOW_CONTRAST" in flat_report.flags


def test_object_verifier(monkeypatch):
    import torch
    import cleaning.object_filter as module
    class FixedClassifier(torch.nn.Module):
        def forward(self, images):
            logits = torch.full((images.shape[0], 1000), -10.0, device=images.device)
            logits[:, 281] = 10.0  # known ImageNet cat class; no pretrained download
            return logits
    monkeypatch.setattr(module.models, "mobilenet_v3_small", lambda **kwargs: FixedClassifier())
    verifier = ObjectVerifier(device="cpu")
    assert verifier is not None

    # Check target classes exist in map
    for c in ("bird", "cat", "dog", "frog", "horse"):
        assert c in TARGET_IMAGENET_MAP
        assert len(TARGET_IMAGENET_MAP[c]) > 0

    # Run verification on a synthetic batch
    test_imgs = [Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40)) for i in range(2)]
    res = verifier.verify_batch(test_imgs, ["cat", "dog"])
    assert len(res) == 2
    assert res[0]["target_prob"] == 1.0 and not res[0]["is_suspicious"]
    assert res[1]["target_prob"] == 0.0 and res[1]["is_suspicious"]
    for r in res:
        assert "target_class" in r
        assert "target_prob" in r
        assert "top1_label" in r
        assert "is_suspicious" in r
