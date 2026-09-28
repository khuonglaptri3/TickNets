"""Quality filtering: Blur detection, exposure and contrast checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import laplace


@dataclass
class QualityReport:
    sample_id: str
    blur_score: float
    mean_luminance: float
    std_luminance: float
    is_blurry: bool
    is_underexposed: bool
    is_overexposed: bool
    is_low_contrast: bool
    flags: List[str]
    is_clean: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compute_blur_score(image: Image.Image) -> float:
    """Compute blurriness score using the variance of the Laplacian.
    
    A higher score indicates sharper edges; very low scores indicate blurriness.
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float64)
    lap = laplace(arr)
    return float(lap.var())


def compute_luminance_stats(image: Image.Image) -> tuple[float, float]:
    """Compute mean and standard deviation of luminance (grayscale values)."""
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def assess_image_quality(
    image: Image.Image,
    sample_id: str = "",
    blur_threshold: Optional[float] = None,
    dark_threshold: float = 35.0,
    bright_threshold: float = 220.0,
    low_contrast_threshold: float = 18.0,
) -> QualityReport:
    """Assess visual quality of an image across blur, exposure, and contrast.
    
    Args:
        image: PIL Image instance.
        sample_id: Identifier of the sample.
        blur_threshold: Threshold below which an image is flagged as blurry.
                        Default: 80.0 for >=128px, 15.0 for 32px.
        dark_threshold: Mean luminance below which an image is underexposed.
        bright_threshold: Mean luminance above which an image is overexposed.
        low_contrast_threshold: Std dev below which an image has flat contrast.
    """
    width, height = image.size
    if blur_threshold is None:
        blur_threshold = 80.0 if max(width, height) >= 128 else 15.0
        
    blur_score = compute_blur_score(image)
    mean_lum, std_lum = compute_luminance_stats(image)
    
    flags = []
    is_blurry = blur_score < blur_threshold
    if is_blurry:
        flags.append("BLURRY")
        
    is_underexposed = mean_lum < dark_threshold
    if is_underexposed:
        flags.append("UNDEREXPOSED")
        
    is_overexposed = mean_lum > bright_threshold
    if is_overexposed:
        flags.append("OVEREXPOSED")
        
    is_low_contrast = std_lum < low_contrast_threshold
    if is_low_contrast:
        flags.append("LOW_CONTRAST")
        
    is_clean = len(flags) == 0
    
    return QualityReport(
        sample_id=sample_id,
        blur_score=round(blur_score, 2),
        mean_luminance=round(mean_lum, 2),
        std_luminance=round(std_lum, 2),
        is_blurry=is_blurry,
        is_underexposed=is_underexposed,
        is_overexposed=is_overexposed,
        is_low_contrast=is_low_contrast,
        flags=flags,
        is_clean=is_clean,
    )
