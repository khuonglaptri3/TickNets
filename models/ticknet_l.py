"""TickNet-L v1: compressed PDP blocks and mixed depthwise kernels.

Derived from this repository's TickNet-Basic, including its existing SE gate.
Mixed depthwise kernels follow the idea in MixConv (arXiv:1907.09595).
This is an untrained architecture candidate, not an accuracy claim.
"""
from collections import OrderedDict

import torch
from torch import nn

from .common import Classifier, conv1x1_block, conv3x3_block
from .SE_Attention import SE


STAGE_CHANNELS = (112, 64, 144, 288, 512)
STAGE_DEPTHS = (1, 1, 2, 2, 1)
STEM_CHANNELS = 24
HEAD_CHANNELS = 768
ARCHITECTURE_REVISION = "ticknet-l-v1"


class MixedDepthwise(nn.Module):
    """Apply different spatial kernels to disjoint channel groups."""

    def __init__(self, channels, stride, kernels):
        super().__init__()
        if not kernels or channels % len(kernels):
            raise ValueError("Depthwise channels must divide evenly among kernels")
        self.split_channels = channels // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=kernel,
                      stride=stride, padding=kernel // 2,
                      groups=self.split_channels, bias=False)
            for kernel in kernels
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        chunks = x.split(self.split_channels, dim=1)
        outputs = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        x = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        return self.activation(self.bn(x))


class CompressedPDPBlock(nn.Module):
    """Linear channel compression -> depthwise -> pointwise -> SE + shortcut."""

    def __init__(self, in_channels, out_channels, stride, hidden_channels, kernels):
        super().__init__()
        # Like Basic's first pointwise layer, the narrow projection is linear.
        self.pw1 = conv1x1_block(in_channels, hidden_channels,
                                use_bn=False, activation=None)
        self.dw = MixedDepthwise(hidden_channels, stride, kernels)
        self.pw2 = conv1x1_block(hidden_channels, out_channels)
        self.se = SE(out_channels, 16)
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else conv1x1_block(in_channels, out_channels, stride=stride)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return self.se(x) + residual


class TickNetL(nn.Module):
    """Seven PDP blocks; identical channel/depth design for both image sizes."""

    def __init__(self, num_classes=5, *, cifar=False):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.in_size = (32, 32) if cifar else (224, 224)
        self.architecture_revision = ARCHITECTURE_REVISION
        strides = (1, 1, 2, 2, 2) if cifar else (1, 2, 2, 2, 2)
        layers = OrderedDict([
            ("data_bn", nn.BatchNorm2d(3)),
            ("init_conv", conv3x3_block(3, STEM_CHANNELS, stride=1 if cifar else 2)),
        ])
        in_channels = STEM_CHANNELS
        
        for stage_index, (out_channels, depth, stride) in enumerate(
                zip(STAGE_CHANNELS, STAGE_DEPTHS, strides)):
            blocks = OrderedDict()
            for block_index in range(depth):
                ratio = 1.0 if stage_index == 0 else 0.75
                # Round to the nearest multiple of 8 so mixed branches split evenly.
                hidden = max(16, int(in_channels * ratio + 4) // 8 * 8)
                kernels = (3,) if stage_index < 2 else (3, 5)
                blocks[f"unit{block_index + 1}"] = CompressedPDPBlock(
                    in_channels, out_channels, stride if block_index == 0 else 1,
                    hidden, kernels)
                in_channels = out_channels
            layers[f"stage{stage_index + 1}"] = nn.Sequential(blocks)
        layers["final_conv"] = conv1x1_block(in_channels, HEAD_CHANNELS)
        """
        # Mixed kernels (3, 5) require an even hidden_channels value.
        layers["stage1"] = nn.Sequential(OrderedDict([
            # Block 1
            ("unit1", CompressedPDPBlock( in_channels=STEM_CHANNELS, out_channels=112, stride=1, hidden_channels=24, kernels=(3,),)),]))
        
        layers["stage2"] = nn.Sequential(OrderedDict([
            # Block 2
            ("unit1", CompressedPDPBlock( in_channels=112, out_channels=64, stride=1 if cifar else 2, hidden_channels=88, kernels=(3,),)),]))
    
        layers["stage3"] = nn.Sequential(OrderedDict([
            # Block 3
            ("unit1", CompressedPDPBlock( in_channels=64, out_channels=144, stride=2, hidden_channels=48, kernels=(3, 5),)),
            # Block 4
            ("unit2", CompressedPDPBlock( in_channels=144, out_channels=144, stride=1, hidden_channels=112, kernels=(3, 5),)),]))
        
        layers["stage4"] = nn.Sequential(OrderedDict([
            # Block 5
            ("unit1", CompressedPDPBlock( in_channels=144, out_channels=288, stride=2, hidden_channels=112, kernels=(3, 5),)),
            # Block 6
            ("unit2", CompressedPDPBlock( in_channels=288, out_channels=288, stride=1, hidden_channels=216, kernels=(3, 5),)),]))

        layers["stage5"] = nn.Sequential(OrderedDict([
            # Block 7
            ("unit1", CompressedPDPBlock( in_channels=288, out_channels=512, stride=2, hidden_channels=216, kernels=(3, 5),)),]))
        
        layers["final_conv"] = conv1x1_block(512, HEAD_CHANNELS)
    """
        layers["global_pool"] = nn.AdaptiveAvgPool2d(1)
        self.backbone = nn.Sequential(layers)
        self.classifier = Classifier(HEAD_CHANNELS, num_classes)
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.classifier.init_params()

    def forward(self, x):
        return self.classifier(self.backbone(x))


def build_ticknet_l(num_classes=5, *, cifar=False):
    return TickNetL(num_classes, cifar=cifar)
