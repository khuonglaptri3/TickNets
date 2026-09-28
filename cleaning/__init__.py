"""Data Cleaning and Quality Assurance module for TickNets datasets (Mid224 & Mid32).

Provides:
- Deduplication via Perceptual Hashing (pHash, dHash)
- Quality filtering (Laplacian blur variance, exposure, contrast)
- Object/label verification using lightweight MobileNetV3-Small
"""
from cleaning.deduplication import compute_phash, compute_dhash, hamming_distance, find_duplicates
from cleaning.quality_filter import assess_image_quality, QualityReport
from cleaning.object_filter import ObjectVerifier

__all__ = [
    "compute_phash",
    "compute_dhash",
    "hamming_distance",
    "find_duplicates",
    "assess_image_quality",
    "QualityReport",
    "ObjectVerifier",
]
