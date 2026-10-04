"""Batch CutMix with clipped-area target weights for a trainer-owned loss."""

import math

import torch


def mix_batch(images, labels, alpha):
    """Replace one shared random rectangle in floating NCHW images.

    Return a clone with paired patches, original/paired integer labels, and
    lambda corrected by the actual clipped area. All random draws use torch;
    inputs are preserved and gradients flow through source and paired pixels.
    """
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("alpha must be finite and greater than zero") from error
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and greater than zero")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be an NCHW tensor")
    if any(size == 0 for size in images.shape) or not images.is_floating_point():
        raise ValueError("images must have nonempty dimensions and floating dtype")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional integer tensor")
    if labels.shape[0] != images.shape[0]:
        raise ValueError("images and labels must have the same batch size")
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("labels must have integer dtype")
    if labels.device != images.device:
        raise ValueError("images and labels must be on the same device")

    concentration = torch.tensor(alpha, dtype=torch.float64, device=images.device)
    sampled_lam = torch.distributions.Beta(concentration, concentration).sample().item()
    permutation = torch.randperm(images.shape[0], device=images.device)
    height, width = images.shape[-2:]
    ratio = math.sqrt(1 - sampled_lam)
    patch_height, patch_width = int(height * ratio), int(width * ratio)
    center_y = torch.randint(height, (), device=images.device).item()
    center_x = torch.randint(width, (), device=images.device).item()
    top, left = center_y - patch_height // 2, center_x - patch_width // 2
    bottom, right = min(height, top + patch_height), min(width, left + patch_width)
    top, left = max(0, top), max(0, left)

    mixed_images = images.clone()
    mixed_images[:, :, top:bottom, left:right] = images[permutation, :, top:bottom, left:right]
    lam = 1 - ((bottom - top) * (right - left)) / (height * width)
    return mixed_images, labels, labels[permutation], lam
