"""Candidate C: configure the teacher's original TickNet and FR-PDP blocks.

Only channel/depth schedules and the existing TickNet stride presets change.
TickNet.forward, FR_PDP_block.forward, SE, stem and head implementations come
directly from models/TickNet.py; no layer-order or kernel replacement is used.
"""
from .TickNet import TickNet


ARCHITECTURE_REVISION = "ticknet-c-v1"
STAGE_CHANNELS = ((80,), (48,), (96, 128), (192, 224), (640, 768, 896))
MID224_STRIDES = (2, 1, 2, 2, 2)  # The original small/large TickNet preset.
MID32_STRIDES = (1, 1, 2, 2, 2)   # The original CIFAR-size TickNet preset.


def build_ticknet_c(num_classes=5, *, cifar=False):
    """Return an original TickNet instance with a new nine-block tick schedule."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    model = TickNet(
        num_classes=num_classes,
        init_conv_channels=32,
        init_conv_stride=1 if cifar else 2,
        channels=STAGE_CHANNELS,
        strides=MID32_STRIDES if cifar else MID224_STRIDES,
        in_size=(32, 32) if cifar else (224, 224),
    )
    model.architecture_revision = ARCHITECTURE_REVISION
    return model
