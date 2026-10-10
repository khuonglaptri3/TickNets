"""TickNet-L Large V2: 14 block PDP, viết riêng từng stage để dễ vấn đáp.

Gộp SE, MixedDepthwise và CompressedPDPBlock vào cùng file này.
Vẫn dùng Classifier, conv1x1_block và conv3x3_block từ common.py của dự án.

Cách chỉnh khi vấn đáp:
1. Sửa trực tiếp từng CompressedPDPBlock trong stage tương ứng.
2. out_channels của block trước phải bằng in_channels của block kế tiếp.
3. hidden_channels phải chia hết cho số kernel; (3, 5) cần số chẵn.
4. Nếu đổi số kênh cuối stage5, sửa cả in_channels của final_conv.
5. Nếu thêm/bớt block, đặt tên unit liên tiếp và cập nhật STAGE_DEPTHS.

Các số hidden_channels dưới đây đã được tính theo bản cũ:
max(16, int(in_channels * ratio + 4) // 8 * 8), với ratio=1.0 ở stage1,
ratio=0.75 ở các stage sau. Khi sửa block, bạn chủ động sửa hidden_channels.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .common import Classifier, conv1x1_block, conv3x3_block


STEM_CHANNELS = 32
# Hai tuple dưới đây chỉ tóm tắt kiến trúc; các block được viết riêng bên dưới.
# Chỉnh trực tiếp các block, rồi cập nhật tuple tương ứng nếu cần.
STAGE_CHANNELS = (160, 128, 256, 512, 768)
STAGE_DEPTHS = (1, 2, 5, 5, 1)
HEAD_CHANNELS = 1024
ARCHITECTURE_REVISION = "ticknet-l-v1"


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """Original SE gate, with identical parameter names for saved checkpoints."""

    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x):
        squeeze_avg = F.avg_pool2d(x, (x.size(2), x.size(3)),
                                   stride=(x.size(2), x.size(3)))
        channel_att = self.mlp(squeeze_avg)
        scale = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)

    def forward(self, x):
        return self.ChannelGate(x)


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
    """14 block PDP; các stage được khai báo trực tiếp, không dùng vòng lặp dựng stage."""

    def __init__(self, num_classes=100, *, cifar=True):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.in_size = (32, 32) if cifar else (224, 224)
        self.architecture_revision = ARCHITECTURE_REVISION
        layers = OrderedDict([
            ("data_bn", nn.BatchNorm2d(3)),
            ("init_conv", conv3x3_block(3, STEM_CHANNELS, stride=1 if cifar else 2)),
        ])
        
        
        """
        in_channels = STEM_CHANNELS
                for stage_index, (out_channels, depth, stride) in enumerate(
                        zip(STAGE_CHANNELS, STAGE_DEPTHS, strides)):
                    blocks = OrderedDict()
                    for block_index in range(depth):
                        ratio = 1.0 if stage_index == 0 else 0.75
                        hidden = max(16, int(in_channels * ratio + 4) // 8 * 8)
                        kernels = (3,) if stage_index < 2 else (3, 5)
                        blocks[f"unit{block_index + 1}"] = CompressedPDPBlock(
                            in_channels, out_channels, stride if block_index == 0 else 1,
                            hidden, kernels)
                        in_channels = out_channels
                    layers[f"stage{stage_index + 1}"] = nn.Sequential(blocks)
        layers["final_conv"] = conv1x1_block(in_channels, HEAD_CHANNELS)
        """
        
        # STAGE 1: 1 block, đầu ra 160 kênh.
        # CIFAR: 32x32 -> 32x32; ảnh 224: 112x112 -> 112x112.
        layers["stage1"] = nn.Sequential(OrderedDict([
            # Block 1
            ("unit1", CompressedPDPBlock(
                in_channels=32, out_channels=160, stride=1,
                hidden_channels=32, kernels=(3,),
            )),
        ]))

        # STAGE 2: 2 block, đầu ra 128 kênh.
        # CIFAR giữ 32x32
        layers["stage2"] = nn.Sequential(OrderedDict([
            # Block 2
            ("unit1", CompressedPDPBlock(
                in_channels=160, out_channels=128,
                stride=1 if cifar else 2,
                hidden_channels=120, kernels=(3,),
            )),
            # Block 3
            ("unit2", CompressedPDPBlock(
                in_channels=128, out_channels=128, stride=1,
                hidden_channels=96, kernels=(3,),
            )),
        ]))

        # STAGE 3: 5 block, đầu ra 256 kênh.
        # unit1 giảm kích thước: CIFAR 32x32 -> 16x16
        layers["stage3"] = nn.Sequential(OrderedDict([
            # Block 4
            ("unit1", CompressedPDPBlock(
                in_channels=128, out_channels=256, stride=2,
                hidden_channels=96, kernels=(3, 5),
            )),
            # Block 5
            ("unit2", CompressedPDPBlock(
                in_channels=256, out_channels=256, stride=1,
                hidden_channels=192, kernels=(3, 5),
            )),
            # Block 6
            ("unit3", CompressedPDPBlock(
                in_channels=256, out_channels=256, stride=1,
                hidden_channels=192, kernels=(3, 5),
            )),
            # Block 7
            ("unit4", CompressedPDPBlock(
                in_channels=256, out_channels=256, stride=1,
                hidden_channels=192, kernels=(3, 5),
            )),
            # Block 8
            ("unit5", CompressedPDPBlock(
                in_channels=256, out_channels=256, stride=1,
                hidden_channels=192, kernels=(3, 5),
            )),
        ]))

        # STAGE 4: 5 block, đầu ra 512 kênh.
        # unit1 giảm kích thước: CIFAR 16x16 -> 8x8;
        layers["stage4"] = nn.Sequential(OrderedDict([
            # Block 9
            ("unit1", CompressedPDPBlock(
                in_channels=256, out_channels=512, stride=2,
                hidden_channels=192, kernels=(3, 5),
            )),
            # Block 10
            ("unit2", CompressedPDPBlock(
                in_channels=512, out_channels=512, stride=1,
                hidden_channels=384, kernels=(3, 5),
            )),
            # Block 11
            ("unit3", CompressedPDPBlock(
                in_channels=512, out_channels=512, stride=1,
                hidden_channels=384, kernels=(3, 5),
            )),
            # Block 12
            ("unit4", CompressedPDPBlock(
                in_channels=512, out_channels=512, stride=1,
                hidden_channels=384, kernels=(3, 5),
            )),
            # Block 13
            ("unit5", CompressedPDPBlock(
                in_channels=512, out_channels=512, stride=1,
                hidden_channels=384, kernels=(3, 5),
            )),
        ]))

        # STAGE 5: 1 block, đầu ra 768 kênh.
        # CIFAR 8x8 -> 4x4; 
        layers["stage5"] = nn.Sequential(OrderedDict([
            # Block 14
            ("unit1", CompressedPDPBlock(
                in_channels=512, out_channels=768, stride=2,
                hidden_channels=384, kernels=(3, 5),
            )),
        ]))
        # Head: 768 -> 1024 kênh; giữ nguyên kích thước không gian.
        layers["final_conv"] = conv1x1_block(768, HEAD_CHANNELS)
        
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


def build_ticknet_l(num_classes=100, *, cifar=True):
    return TickNetL(num_classes, cifar=cifar)
