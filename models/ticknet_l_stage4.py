"""TickNet-L v1 ablation with one extra compressed PDP block in Stage 4."""
from torch import nn

from .ticknet_l import CompressedPDPBlock, TickNetL


ARCHITECTURE_REVISION = "ticknet-l-v1-stage4-depth3"


class TickNetLStage4(TickNetL):
    """Eight blocks; preserve L v1 and deepen only Stage 4 from two to three."""

    def __init__(self, num_classes=5, *, cifar=False):
        super().__init__(num_classes, cifar=cifar)
        block = CompressedPDPBlock(
            in_channels=288, out_channels=288, stride=1,
            hidden_channels=216, kernels=(3, 5))
        # Match the parent's Conv2d initialization for the newly appended block.
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.backbone.stage4.add_module("unit3", block)
        self.architecture_revision = ARCHITECTURE_REVISION


def build_ticknet_l_stage4(num_classes=5, *, cifar=False):
    return TickNetLStage4(num_classes, cifar=cifar)
