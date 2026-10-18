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

from author.models.common import Classifier, ConvBlock, conv1x1_block, conv3x3_block, get_activation


STEM_CHANNELS = 32
# Hai tuple dưới đây chỉ tóm tắt kiến trúc; các block được viết riêng bên dưới.
# Chỉnh trực tiếp các block, rồi cập nhật tuple tương ứng nếu cần.
STAGE_CHANNELS = (160, 128, 256, 512, 768)
STAGE_DEPTHS = (1, 2, 5, 5, 1)
HEAD_CHANNELS = 1024
ARCHITECTURE_REVISION = "ticknet-l-large-v2"


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

    def __init__(
        self,
        channels,
        stride,
        kernels,
        paddings=None,
        dilation=1,
        groups=None,
        bias=False,
        use_bn=True,
        activation="relu",
    ):
        super().__init__()
        if not kernels or channels % len(kernels):
            raise ValueError("Depthwise channels must divide evenly among kernels")
        self.split_channels = channels // len(kernels)
        if paddings is None:
            branch_paddings = [kernel // 2 for kernel in kernels]
        elif isinstance(paddings, int):
            branch_paddings = [paddings for _ in kernels]
        else:
            branch_paddings = list(paddings)
            if len(branch_paddings) != len(kernels):
                raise ValueError("Length of paddings must match length of kernels")

        if groups is None:
            branch_groups = [self.split_channels for _ in kernels]
        elif isinstance(groups, int):
            branch_groups = [groups for _ in kernels]
        else:
            branch_groups = list(groups)

        self.branches = nn.ModuleList([
            nn.Conv2d(
                self.split_channels,
                self.split_channels,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=grp,
                bias=bias,
            )
            for kernel, pad, grp in zip(kernels, branch_paddings, branch_groups)
        ])
        self.bn = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.activation = get_activation(activation) if activation is not None else nn.Identity()

    def forward(self, x):
        chunks = x.split(self.split_channels, dim=1)
        outputs = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        x = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        return self.activation(self.bn(x))


class CompressedPDPBlock(nn.Module):
    """Linear channel compression -> depthwise -> pointwise -> SE + shortcut.

    Có thể truyền trực tiếp từng sub-block (pw1, dw, pw2, se, shortcut) ở mỗi block
    để tùy ý chỉnh sửa (S, P, K, groups, ...) hoặc bọc nn.Sequential nối thêm lớp
    vào riêng 1 block mà không ảnh hưởng các block khác.
    """

    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        stride=1,
        hidden_channels=None,
        kernels=(3,),
        use_se=True,
        use_shortcut=True,
        *,
        pw1=None,
        dw=None,
        pw2=None,
        se=None,
        shortcut=None,
    ):
        super().__init__()
        self.pw1 = pw1 if pw1 is not None else conv1x1_block(
            in_channels, hidden_channels, use_bn=False, activation=None
        )
        self.dw = dw if dw is not None else MixedDepthwise(
            hidden_channels, stride, kernels
        )
        self.pw2 = pw2 if pw2 is not None else conv1x1_block(
            hidden_channels, out_channels
        )
        if not use_se:
            self.se = nn.Identity()
        elif se is not None:
            self.se = se
        else:
            self.se = SE(out_channels, 16)

        self.use_shortcut = use_shortcut
        if not use_shortcut:
            self.shortcut = None
        elif shortcut is not None:
            self.shortcut = shortcut
        else:
            self.shortcut = (
                nn.Identity() if stride == 1 and in_channels == out_channels
                else conv1x1_block(in_channels, out_channels, stride=stride)
            )

    def forward(self, x):
        residual = self.shortcut(x) if (self.use_shortcut and self.shortcut is not None) else None
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        x = self.se(x)
        return x if residual is None else x + residual


class TickNetL(nn.Module):
    """14 block PDP; mọi sub-block (pw1, dw, pw2, se, shortcut) và tham số (K, S, P, groups) được đưa hết ra ngoài từng block."""

    def __init__(self, num_classes=100, *, cifar=True):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.in_size = (32, 32) if cifar else (224, 224)
        self.architecture_revision = ARCHITECTURE_REVISION
        layers = OrderedDict([
            ("data_bn", nn.BatchNorm2d(3)),
            ("init_conv", ConvBlock(
                in_channels=3, out_channels=STEM_CHANNELS,
                kernel_size=3, stride=1 if cifar else 2, padding=1,
                dilation=1, groups=1, bias=False, use_bn=True, activation="relu",
            )),
        ])

        # =====================================================================
        # STAGE 1: 1 block, đầu ra 160 kênh.
        # CIFAR: 32x32 -> 32x32; ảnh 224: 112x112 -> 112x112.
        # =====================================================================
        layers["stage1"] = nn.Sequential(OrderedDict([
            # Block 1 (in=32, hidden=32, out=160, stride=1)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=32, out_channels=32,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=32, stride=1,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=32, out_channels=160,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=160, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=32, out_channels=160,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 2: 2 block (+ unitthem), đầu ra 128 kênh.
        # CIFAR giữ 32x32
        # =====================================================================
        layers["stage2"] = nn.Sequential(OrderedDict([
            # Block 2 (in=160, hidden=120, out=128, stride=1 if cifar else 2)
            # Ví dụ: muốn nối thêm 1 conv1x1 vào pw1 hoặc shortcut của riêng Block 2,
            # chỉ cần bọc pw1=nn.Sequential(ConvBlock(...), ConvBlock(...)) ngay tại đây.
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=160, out_channels=120,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=120, stride=1 if cifar else 2,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=120, out_channels=128,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=128, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=160, out_channels=128,
                    kernel_size=1, stride=1 if cifar else 2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 3 (in=128, hidden=96, out=128, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=128, out_channels=96,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=96, stride=1,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=96, out_channels=128,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=128, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            
        ]))

        # =====================================================================
        # STAGE 3: 5 block, đầu ra 256 kênh.
        # unit1 giảm kích thước: CIFAR 32x32 -> 16x16
        # =====================================================================
        layers["stage3"] = nn.Sequential(OrderedDict([
            # Block 4 (in=128, hidden=96, out=256, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=128, out_channels=96,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=96, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=96, out_channels=256,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=256, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=128, out_channels=256,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 5 (in=256, hidden=192, out=256, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=256, out_channels=192,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=192, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=192, out_channels=256,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=256, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 6 (in=256, hidden=192, out=256, stride=1)
            ("unit3", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=256, out_channels=192,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=192, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=192, out_channels=256,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=256, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 7 (in=256, hidden=192, out=256, stride=1)
            ("unit4", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=256, out_channels=192,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=192, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=192, out_channels=256,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=256, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 8 (in=256, hidden=192, out=256, stride=1)
            ("unit5", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=256, out_channels=192,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=192, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=192, out_channels=256,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=256, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 4: 5 block, đầu ra 512 kênh.
        # unit1 giảm kích thước: CIFAR 16x16 -> 8x8;
        # =====================================================================
        layers["stage4"] = nn.Sequential(OrderedDict([
            # Block 9 (in=256, hidden=192, out=512, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=256, out_channels=192,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=192, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=192, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=256, out_channels=512,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 10 (in=512, hidden=384, out=512, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=512, out_channels=384,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=384, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=384, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 11 (in=512, hidden=384, out=512, stride=1)
            ("unit3", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=512, out_channels=384,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=384, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=384, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 12 (in=512, hidden=384, out=512, stride=1)
            ("unit4", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=512, out_channels=384,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=384, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=384, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 13 (in=512, hidden=384, out=512, stride=1)
            ("unit5", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=512, out_channels=384,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=384, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=384, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 5: 1 block, đầu ra 768 kênh.
        # CIFAR 8x8 -> 4x4;
        # =====================================================================
        layers["stage5"] = nn.Sequential(OrderedDict([
            # Block 14 (in=512, hidden=384, out=768, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=512, out_channels=384,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=384, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=384, out_channels=768,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=768, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=512, out_channels=768,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
        ]))
        # Head: 768 -> 1024 kênh; giữ nguyên kích thước không gian.
        layers["final_conv"] = ConvBlock(
            in_channels=768, out_channels=HEAD_CHANNELS,
            kernel_size=1, stride=1, padding=0,
            dilation=1, groups=1, bias=False, use_bn=True, activation="relu",
        )
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

