"""Perceptual hashing and near-duplicate detection."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from PIL import Image
from scipy.fftpack import dct


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute 64-bit Perceptual Hash (pHash) using Discrete Cosine Transform (DCT).
    
    Robust to minor resizing, compression artifacts, and slight brightness shifts.
    """
    img_size = hash_size * highfreq_factor
    resized = image.convert("L").resize((img_size, img_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    
    # 2D DCT
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_2d = dct(dct_rows, axis=1, norm="ortho")
    
    # Extract low-frequency components (top-left hash_size x hash_size)
    low_freq = dct_2d[:hash_size, :hash_size]
    # Median excluding DC term at (0, 0)
    flat = low_freq.flatten()
    med = float(np.median(flat[1:])) if len(flat) > 1 else float(flat[0])
    
    diff = low_freq > med
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    # Format as 16-hex digit string (64 bits)
    hex_hash = f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"
    return hex_hash


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute Difference Hash (dHash) based on horizontal pixel gradients."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate the Hamming distance (number of differing bits) between two hex hashes."""
    if len(hash1) != len(hash2):
        raise ValueError(f"Hash lengths do not match: {len(hash1)} vs {len(hash2)}")
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    return bin(val1 ^ val2).count("1")


def find_duplicates(
    samples: List[dict],
    near_dup_threshold: int = 4,
    hash_key: str = "phash",
) -> Dict[str, any]:
    """Find exact duplicates and near duplicates within samples.
    
    Args:
        samples: List of dicts, each containing at least 'sample_id' and hash_key.
        near_dup_threshold: Max Hamming distance to consider two images as near-duplicates (0 = exact).
        hash_key: Key in sample dict containing the hex hash string.
        
    Returns:
        Dict with 'exact_duplicate_groups' and 'near_duplicate_pairs'.
    """
    exact_groups = defaultdict(list)
    for s in samples:
        exact_groups[s[hash_key]].append(s["sample_id"])
    
    # Filter to groups with > 1 element
    exact_duplicates = {h: ids for h, ids in exact_groups.items() if len(ids) > 1}
    
    # Near duplicate search across unique hashes
    unique_hashes = list(exact_groups.keys())
    near_duplicate_pairs = []
    
    # For small-to-medium clusters, compare pairwise
    # In large sets, grouping by coarse prefixes or bucket indexing accelerates search
    for i in range(len(unique_hashes)):
        h1 = unique_hashes[i]
        for j in range(i + 1, len(unique_hashes)):
            h2 = unique_hashes[j]
            dist = hamming_distance(h1, h2)
            if 0 < dist <= near_dup_threshold:
                near_duplicate_pairs.append({
                    "sample_a_examples": exact_groups[h1][:3],
                    "sample_b_examples": exact_groups[h2][:3],
                    "hamming_distance": dist,
                    "hash_a": h1,
                    "hash_b": h2,
                })
                
    return {
        "exact_duplicate_groups": exact_duplicates,
        "exact_duplicate_sample_count": sum(len(ids) for ids in exact_duplicates.values()),
        "near_duplicate_pairs": near_duplicate_pairs,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
    }
