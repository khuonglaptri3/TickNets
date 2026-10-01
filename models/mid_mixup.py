"""Batch Mixup with integer target pairing for a trainer-owned loss."""

import math

import torch


def mix_batch(images, labels, alpha):
    """Mix floating NCHW images; return images, original/paired labels, lambda.

    Labels must be a one-dimensional integer tensor on the images' device.
    All randomness follows torch's seeded RNG. Inputs are never modified;
    gradients flow through both members of each pair, including batch size one.
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
    lam = torch.distributions.Beta(concentration, concentration).sample().item()
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed_images = lam * images + (1 - lam) * images[permutation]
    return mixed_images, labels, labels[permutation], lam
