"""Object and label verification using lightweight MobileNetV3-Small."""
from __future__ import annotations

import ssl
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
import torchvision.models as models

# Safeguard for corporate / proxy environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# ImageNet-1K index groupings for the 5 target classes
TARGET_IMAGENET_MAP: Dict[str, List[int]] = {
    "bird": list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
    "dog": list(range(151, 269)),
    "cat": list(range(281, 294)),
    "frog": [30, 31, 32],
    "horse": [339, 340],
}


class ObjectVerifier:
    """Verifies image content against expected class labels using pretrained MobileNetV3-Small."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.transforms = weights.transforms()
        
        self.model = models.mobilenet_v3_small(weights=weights).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def verify_batch(
        self,
        images: List[Image.Image],
        target_labels: List[str],
        confidence_threshold: float = 0.05,
    ) -> List[dict]:
        """Perform batched verification of images against expected class labels."""
        if len(images) != len(target_labels):
            raise ValueError("Number of images and labels must match")
        if not images:
            return []

        tensors = torch.stack([self.transforms(img.convert("RGB")) for img in images]).to(self.device)
        logits = self.model(tensors)
        probs = torch.softmax(logits, dim=1)

        results = []
        for i, target_label in enumerate(target_labels):
            norm_label = target_label.lower().strip()
            synsets = TARGET_IMAGENET_MAP.get(norm_label, [])
            
            target_prob = float(probs[i, synsets].sum().item()) if synsets else 0.0
            
            top5 = torch.topk(probs[i], 5)
            top1_idx = int(top5.indices[0].item())
            top1_prob = float(top5.values[0].item())
            top1_label = self.categories[top1_idx]
            
            top5_indices = set(top5.indices.tolist())
            is_target_in_top5 = bool(top5_indices.intersection(synsets))
            
            # Suspicious if target class probability is extremely low
            # AND model confidently predicts a conflicting category
            is_suspicious = (target_prob < confidence_threshold) and (top1_prob > 0.20) and not is_target_in_top5

            results.append({
                "target_class": norm_label,
                "target_prob": round(target_prob, 4),
                "top1_label": top1_label,
                "top1_prob": round(top1_prob, 4),
                "is_target_in_top5": is_target_in_top5,
                "is_suspicious": is_suspicious,
            })

        return results
