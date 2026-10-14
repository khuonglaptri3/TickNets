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

1. B4 (3,7)-->(3,5),96/2=48 padding 3//2=1, 7//2=3 tính Ho vẫn là 16x16, Ch%Số Kernel = 0 nếu khác 0 thì đổi Ch lại
2. B4(3,5)-->(3,5,7), 96/3=32 padding 3//2=1, 5//2=2, 7//2=3, Ch%Số Kernel = 0 thì ko cần đổi Ch
3. B4 Ch 128-->120, Ch mới vẫn % số kernel=0 nên ko đổi, khi đổi Ci thì tính Ch = 0.75Ci
4. B4 Co 256-->320, đổi STAGES_CHANNELS lại, do Co của block 4 nên block 4 ko đổi, đổi Ci B5-->320, đổi B5 Ch=0.75*320=240, 240% số kernel=0 nên ko đổi
5. B4 s2-->1, B4-8 ở 32x32, B9 còn 16x16,B14 còn4x4
6. Thêm Block sau B8 ở cuối stage 3, copy paste block trước, rồi sửa STAGES_DEPTH
7. Bỏ B6, chỉnh STAGE_DEPTHS = (1, 2, 4, 5, 1) lại
8. Đổi stage 3 256-->320, B4 đổi Co 256-->320, B5-8 Co->320, Ch->240, cập nhật STAGE_CHANNELS = (160, 128, 320, 512, 768)
9. Stem 32->40, B01 Ci=Ch=40
10. B14 xuất 768->640, B14 Co=640,final_conv thành conv1x1_block(640, HEAD_CHANNELS), sửa STAGE_CHANNELS = (160, 128, 320, 512, 640)  
11. Head 1024->1280, HEADS_CHANNELS=1280
12. cifar 10 thay cifar 100, 10, num_classes=10
13. bỏ SE --> use_se=False
14. bỏ shortcut --> use_shortcut=False
15. bỏ pw1 --> use_pw1=False, bỏ pw2 --> use_pw2=False
16. đổi padding 

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
                    #   stride=stride, padding=0, #nếu đổi p ở mọi DW = 0
                    #   stride=stride, padding=0 if kernel == 3 else kernel // 2, #nếu đổi p ở DW3 = 0
                    #   stride=stride, padding=0 if kernel == 5 else kernel // 2, #nếu đổi p ở DW5 = 0
                    #   stride=stride, padding={3: 0, 5: 1}[kernel], #nếu đổi p ở DW5=1,DW3= 0
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

    # def __init__(self, in_channels, out_channels, stride, hidden_channels, kernels):
    def __init__(self, in_channels, out_channels, stride, hidden_channels, kernels, use_se=True, use_shortcut=True,use_pw1=True, use_pw2=True,):
        super().__init__()
        # Like Basic's first pointwise layer, the narrow projection is linear.
        #Bo Pw1 --> Ch = Ci
        self.pw1 = conv1x1_block(in_channels, hidden_channels,
                                use_bn=False, activation=None) if use_pw1 else nn.Identity()
        self.dw = MixedDepthwise(hidden_channels, stride, kernels)
        #Bo Pw2 --> Ch = Co
        self.pw2 = conv1x1_block(hidden_channels, out_channels) if use_pw2 else nn.Identity()
        self.se = SE(out_channels, 16) if use_se else nn.Identity()
        # self.se = SE(out_channels, 16)
        self.use_shortcut = use_shortcut
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else conv1x1_block(in_channels, out_channels, stride=stride)
        # )
        ) if use_shortcut else None

    def forward(self, x):
        # residual = self.shortcut(x)
        residual = self.shortcut(x) if self.use_shortcut else None
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        x = self.se(x)
        return x if residual is None else x + residual


class TickNetL(nn.Module):
    """14 block PDP; các stage được khai báo trực tiếp, không dùng vòng lặp dựng stage."""

    def __init__(self, num_classes=100, *, cifar=True):
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
        #Doi stem p=1 --> 0 hoac 2
        # if cifar: 
            #p=0-->stem 32x32-->sau B04 16x16-->sau B09 8x8-->sau B14 4x4
            # layers["init_conv"].conv.padding = (2, 2) # p = 2-->stem 34x34-->sau B04 17x17-->sau B09 9x9-->sau B14 5x5
            # layers["init_conv"].conv.padding = (0, 0) # p = 0-->stem 30x30-->sau B04 15x15-->sau B09 8x8-->sau B14 4x4
            
        
        
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
        # """"""
        # STAGE 1: 1 block, đầu ra 160 kênh.
        # CIFAR: 32x32 -> 32x32; ảnh 224: 112x112 -> 112x112.
        layers["stage1"] = nn.Sequential(OrderedDict([
            # Block 1
            ("unit1", CompressedPDPBlock(
                in_channels=32, out_channels=160, stride=1,
                hidden_channels=32, kernels=(3,),
            )),
        ]))
        """
        # B1 DW3 p1->0 s1
        # p3, bu3 = 0, 2
        # block = layers["stage1"].unit1
        # dw3 = block.dw.branches[0]
        # dw3.padding = (p3, p3)
        # if bu3 > 0:
        #     block.dw.branches[0] = nn.Sequential(
        #         dw3,
        #         nn.ZeroPad2d((0, bu3, 0, bu3)),
        #     )
        """
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
        # B2 DW3 p1->0 s1
        # p3, bu3 = 0, 2
        # block = layers["stage2"].unit1
        # dw3 = block.dw.branches[0]
        # dw3.padding = (p3, p3)
        # if bu3 > 0:
        #     block.dw.branches[0] = nn.Sequential(
        #         dw3,
        #         nn.ZeroPad2d((0, bu3, 0, bu3)),
        #     )
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
        """
        # đổi padding riêng DW5 của B04, số 0 là DW3
        # layers["stage3"].unit1.dw.branches[1].padding = (1, 1) 
        # layers["stage3"].unit1.dw.branches[1] = nn.Sequential(
            # dw5,
            # nn.ZeroPad2d((0, 1, 0, 1)),
        # )
        #DW5 p=2->1 p3,p5 = 1,1 bu3,bu5 = 0,1
        #DW5 p=2->0 p3,p5 = 1,0 bu3,bu5 = 0,2
        #DW3 p=1->0 p3,p5 = 0,2 bu3,bu5 = 1,0
        #DW3 p=1->0,DW5 p=2->1 p3,p5 = 0,1 bu3,bu5 = 1,1
        #DW3 p=1->0,DW5 p=2->0 p3,p5 = 0,0 bu3,bu5 = 1,2
        # p3, p5 = 1, 1
        # bu3, bu5 = 0, 1
        # b04 = layers["stage3"].unit1
        # dw3 = b04.dw.branches[0]
        # dw5 = b04.dw.branches[1]
        # dw3.padding = (p3, p3)
        # dw5.padding = (p5, p5)
        # if bu3 > 0:
        #     b04.dw.branches[0] = nn.Sequential(
        #         dw3,
        #         nn.ZeroPad2d((0, bu3, 0, bu3)),
        #     )
        # if bu5 > 0:
        #     b04.dw.branches[1] = nn.Sequential(
        #         dw5,
        #         nn.ZeroPad2d((0, bu5, 0, bu5)),
        #     )   
        """
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
        #Dw3 p=0 p3,p5=0,2
        #Dw5 p=0 p3,p5=1,0
        #Dw3p=0,Dw5p=1, p3,p5=0,1
        #DW3,5 p=0, p3,p5=0,0
        #DW3p=1,DW5p=1, p3,p5=1,1
        p3, p5 = 1, 1
        for stage_name in (
            "stage1", "stage2", "stage3", "stage4", "stage5"
        ):
            for block in layers[stage_name]:
                for index, dw in enumerate(block.dw.branches):
                    k = dw.kernel_size[0]
                    s = dw.stride[0]
                    p = {3: p3, 5: p5}[k]
                    # Áp dụng khi s=1 hoặc s=2,
                    # padding mới nhỏ hơn hoặc bằng padding gốc.
                    bu = 2 * (k // 2 - p) // s
                    dw.padding = (p, p)
                    if bu > 0:
                        block.dw.branches[index] = nn.Sequential(
                            dw,
                            nn.ZeroPad2d((0, bu, 0, bu)),
                        )
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
