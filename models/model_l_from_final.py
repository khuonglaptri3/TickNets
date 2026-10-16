"""TickNet-L From Final: 10 block PDP viết tường minh từng stage cho bộ dữ liệu Giữa kỳ (Mid32 & Mid224, 5 lớp).

Lấy cấu trúc viết tường minh từng block từ nhánh cuối kỳ (release/Final_report) và điều chỉnh
theo Cách 2 (giữ nguyên lịch stride gốc của Mid32/Mid224, thu gọn từ [1, 2, 5, 5, 1] xuống [1, 2, 3, 3, 1]
và kênh (112, 64, 144, 288, 512)) để đạt chuẩn ngân sách (< 1.0 GFLOPs và <= 6M tham số) trên cả
hai độ phân giải Mid32 (32x32) và Mid224 (224x224) với num_classes=5.

Cách chỉnh khi vấn đáp / thử nghiệm:
1. Sửa trực tiếp từng CompressedPDPBlock trong stage tương ứng.
2. out_channels của block trước phải bằng in_channels của block kế tiếp.
3. hidden_channels phải chia hết cho số kernel; (3, 5) cần số chẵn.
4. Nếu đổi số kênh cuối stage5, sửa cả in_channels của final_conv.
5. Nếu thêm/bớt block, đặt tên unit liên tiếp và cập nhật STAGE_DEPTHS.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .common import Classifier, ConvBlock, conv1x1_block, conv3x3_block, get_activation


STEM_CHANNELS = 24
STAGE_CHANNELS = (112, 64, 144, 288, 512)
STAGE_DEPTHS = (1, 2, 3, 3, 1)
HEAD_CHANNELS = 768
ARCHITECTURE_REVISION = "ticknet-l-from-final-v1"


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


class TickNetLFromFinal(nn.Module):
    """10 block PDP [1, 2, 3, 3, 1]; viết tường minh từng block cho Mid32 (32x32) và Mid224 (224x224)."""

    def __init__(self, num_classes=5, *, cifar=False):
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
        # STAGE 1: 1 block, đầu ra 112 kênh.
        # Mid32: 32x32 -> 32x32; Mid224: 112x112 -> 112x112.
        # =====================================================================
        layers["stage1"] = nn.Sequential(OrderedDict([
            # Block 1 (in=24, hidden=24, out=112, stride=1)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=24, out_channels=24,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=24, stride=1,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=24, out_channels=112,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=112, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=24, out_channels=112,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 2: 2 block, đầu ra 64 kênh.
        # Mid32: giữ 32x32 (stride=1); Mid224: 112x112 -> 56x56 (stride=2).
        # =====================================================================
        layers["stage2"] = nn.Sequential(OrderedDict([
            # Block 2 (in=112, hidden=88, out=64, stride=1 if cifar else 2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=112, out_channels=88,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=88, stride=1 if cifar else 2,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=88, out_channels=64,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=64, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=112, out_channels=64,
                    kernel_size=1, stride=1 if cifar else 2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 3 (in=64, hidden=48, out=64, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=64, out_channels=48,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=48, stride=1,
                    kernels=(3,), paddings=(1,), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=48, out_channels=64,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=64, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 3: 3 block, đầu ra 144 kênh.
        # unit1 giảm kích thước: Mid32 32x32 -> 16x16; Mid224 56x56 -> 28x28.
        # =====================================================================
        layers["stage3"] = nn.Sequential(OrderedDict([
            # Block 4 (in=64, hidden=48, out=144, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=64, out_channels=48,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=48, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=48, out_channels=144,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=144, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=64, out_channels=144,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 5 (in=144, hidden=112, out=144, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=144, out_channels=112,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=112, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=112, out_channels=144,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=144, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 6 (in=144, hidden=112, out=144, stride=1)
            ("unit3", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=144, out_channels=112,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=112, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=112, out_channels=144,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=144, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 4: 3 block, đầu ra 288 kênh.
        # unit1 giảm kích thước: Mid32 16x16 -> 8x8; Mid224 28x28 -> 14x14.
        # =====================================================================
        layers["stage4"] = nn.Sequential(OrderedDict([
            # Block 7 (in=144, hidden=112, out=288, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=144, out_channels=112,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=112, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=112, out_channels=288,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=288, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=144, out_channels=288,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 8 (in=288, hidden=216, out=288, stride=1)
            ("unit2", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=288, out_channels=216,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=216, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=216, out_channels=288,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=288, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
            # Block 9 (in=288, hidden=216, out=288, stride=1)
            ("unit3", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=288, out_channels=216,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=216, stride=1,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=216, out_channels=288,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=288, reduction_ratio=16),
                shortcut=nn.Identity(),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # =====================================================================
        # STAGE 5: 1 block, đầu ra 512 kênh.
        # Mid32: 8x8 -> 4x4; Mid224: 14x14 -> 7x7.
        # =====================================================================
        layers["stage5"] = nn.Sequential(OrderedDict([
            # Block 10 (in=288, hidden=216, out=512, stride=2)
            ("unit1", CompressedPDPBlock(
                pw1=ConvBlock(
                    in_channels=288, out_channels=216,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=False, activation=None,
                ),
                dw=MixedDepthwise(
                    channels=216, stride=2,
                    kernels=(3, 5), paddings=(1, 2), dilation=1, groups=None,
                    bias=False, use_bn=True, activation="relu",
                ),
                pw2=ConvBlock(
                    in_channels=216, out_channels=512,
                    kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                se=SE(gate_channels=512, reduction_ratio=16),
                shortcut=ConvBlock(
                    in_channels=288, out_channels=512,
                    kernel_size=1, stride=2, padding=0, dilation=1, groups=1,
                    bias=False, use_bn=True, activation="relu",
                ),
                use_se=True,
                use_shortcut=True,
            )),
        ]))

        # Head: 512 -> 768 kênh; giữ nguyên kích thước không gian.
        layers["final_conv"] = ConvBlock(
            in_channels=512, out_channels=HEAD_CHANNELS,
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


TickNetL = TickNetLFromFinal


def build_model_l_from_final(num_classes=5, *, cifar=False):
    return TickNetLFromFinal(num_classes, cifar=cifar)
