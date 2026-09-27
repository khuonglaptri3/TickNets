# TickNet-C v1: ứng viên dùng nguyên backbone TickNet của thầy

## 1. Quyết định thiết kế

C tăng dung lượng lên khoảng **5,16 triệu tham số**, với chi phí khoảng
**0,82 GFLOPs cho ảnh 224x224**. Mục tiêu là thử một cấu hình lớn hơn trong
giới hạn đề: không quá 6 triệu tham số học được và dưới 1 GFLOPs.
Đây là ứng viên bổ sung; Basic và L vẫn là các lựa chọn độc lập.

**Backbone được dựng trực tiếp bằng `models.TickNet.TickNet`; cả 9 block đều
là `models.TickNet.FR_PDP_block` nguyên bản.** C chỉ truyền cấu hình mới vào
constructor của thầy và đặt tên phiên bản `ticknet-c-v1`.

Mã xây dựng nằm trong [`models/ticknet_c.py`](../models/ticknet_c.py):

```python
model = TickNet(
    num_classes=5,
    init_conv_channels=32,
    init_conv_stride=2,                 # 1 cho Mid32
    channels=((80,), (48,), (96, 128), (192, 224), (640, 768, 896)),
    strides=(2, 1, 2, 2, 2),            # (1, 1, 2, 2, 2) cho Mid32
    in_size=(224, 224),                 # (32, 32) cho Mid32
)
```

Các lớp `TickNet`, `FR_PDP_block`, `SE`, `ConvBlock` và `Classifier` được dùng
trực tiếp từ mã gốc. `TickNet.py`, `common.py` và `SE_Attention.py` giữ nguyên.

## 2. Cơ chế TickNet được kế thừa

Luồng trong mỗi block giữ đúng thứ tự của thầy:

```text
x -> PW1 1x1 (Cin -> Cin, linear)
  -> DW 3x3 (stride của block)
  -> PW2 1x1 (Cin -> Cout, BN + ReLU)
  -> SE
  -> cộng shortcut (identity hoặc PW projection theo mã gốc)
```

`TickNet.forward` vẫn chạy backbone rồi classifier như bản gốc. C giữ:

- Data BatchNorm và stem Conv 3x3 với **32 kênh**.
- Point-depth-point, depthwise **3x3**, SE và full residual trong `FR_PDP_block`.
- Head Conv 1x1 với **1024 kênh**, global average pooling và classifier gốc.
- Lịch stride 224x224 vốn có của **TickNet-small/large**: `(2, 1, 2, 2, 2)`.
- Lịch stride cho ảnh 32x32 vốn có của TickNet: `(1, 1, 2, 2, 2)`.

SE là thành phần kế thừa. Thay classifier thành 5 lớp chỉ phục vụ dataset.
Thay đổi kiến trúc của C nằm ở lịch kênh, số block và cách phân bổ stride giữa
các stage, trong cùng họ TickNet và cùng triển khai block của thầy.

## 3. Cấu hình backbone và lý do

| Thành phần | TickNet-Basic | TickNet-C |
|---|---|---|
| Stem | 32 kênh | 32 kênh |
| Stage 1 | 128 | 80 |
| Stage 2 | 64 | 48 |
| Stage 3 | 128 | 96 → 128 |
| Stage 4 | 256 | 192 → 224 |
| Stage 5 | 512 | 640 → 768 → 896 |
| Số block mỗi stage | 1, 1, 1, 1, 1 | 1, 1, 2, 2, 3 |
| Stride stage ở 224x224 | 1, 2, 2, 2, 2 | 2, 1, 2, 2, 2 |
| Head | 1024 | 1024 |

Lịch kênh C vẫn tạo dạng tick: **32 → 80 → 48 → 96 → 128 → 192 → 224 →
640 → 768 → 896 → 1024**. Sau đỉnh sớm 80 và đáy 48, các stage sau tăng dần
dung lượng. C giảm số kênh ở các tầng đầu và tăng kênh/số block ở tầng cuối.

Ở 224x224, dùng stride 2 tại stage 1 làm pointwise đầu stage 2 chạy trên
56x56. Đây là cấu hình stride đã có trong mã TickNet của thầy. Block vẫn thực
hiện pointwise trước depthwise theo đúng mã nguồn gốc.

| Sau thành phần | Kênh | Spatial Mid224 | Spatial Mid32 |
|---|---:|---|---|
| Stem | 32 | 112x112 | 32x32 |
| Stage 1 | 80 | 56x56 | 32x32 |
| Stage 2 | 48 | 56x56 | 32x32 |
| Stage 3 | 128 | 28x28 | 16x16 |
| Stage 4 | 224 | 14x14 | 8x8 |
| Stage 5 | 896 | 7x7 | 4x4 |
| Head | 1024 | 7x7 | 4x4 |
| Global pooling / classifier | 1024 / 5 | 1x1 | 1x1 |

Các block liền nhau có số kênh khác nhau, vì vậy nhánh projection `PwR`
của block gốc được sử dụng trong cả 9 block. Số tham số tăng thêm nằm trên
các nhánh được thực thi và được kiểm tra gradient.

Việc giảm độ phân giải sớm hoặc tăng dung lượng tầng cuối có thể ảnh hưởng
accuracy. Phân bổ này được chọn theo cấu trúc và phép đếm tính toán; chưa có
kết quả train để kết luận nó tốt hơn Basic hoặc L.

## 4. Số đo chính thức

| Mô hình | Tham số học được | GFLOPs Mid32 | GFLOPs Mid224 |
|---|---:|---:|---:|
| TickNet-Basic | 1.062.223 | 0,158428 | 0,988343 |
| TickNet-L v1 | 1.096.260 | 0,157821 | 0,796760 |
| **TickNet-C v1** | **5.155.467** | **0,256830** | **0,821054** |

C dùng khoảng **85,9% giới hạn tham số** và **82,1% giới hạn FLOPs ở 224x224**.
Số đo thuộc cấu hình C được dựng bằng `TickNet` gốc ở trên.

Quy ước đo giống Basic/L: batch 1, float32, một forward `eval`, RGB, 5 lớp;
**1 MAC = 2 FLOPs**, tính Conv2d/Linear thực thi, gồm cả projection và các
Linear trong SE. Không tính bias addition, BatchNorm, activation, pooling,
cộng residual, nhân scale SE hoặc vận chuyển bộ nhớ. Số FLOPs không phải
phép đo latency thực tế.

Hook đếm theo module được đối chiếu độc lập với
`torch.utils.flop_counter.FlopCounterMode`. Kết quả chi tiết theo layer/stage
và phiên bản PyTorch nằm trong [`model_profiles.json`](model_profiles.json).

```powershell
conda activate fresher
python profile_mid.py --models basic l c --output docs/model_profiles.json
python -m pytest -q
```

Kết quả kiểm chứng ngày 2026-09-27: **32 tests passed**. Kiểm thử xác nhận
kiểu đối tượng `TickNet`, 9 block `FR_PDP_block` và các hàm forward gốc;
forward/backward ở cả hai kích thước với gradient hữu hạn cho mọi tham số;
huấn luyện fixture một epoch, lưu/nạp checkpoint C và từ chối checkpoint sai
phiên bản kiến trúc. Sáu cấu hình Basic/L/C ở 32/224 có hai phép đếm FLOPs
độc lập khớp nhau. Đối chiếu Git với `upstream/main` xác nhận `TickNet.py`,
`common.py` và `SE_Attention.py` không thay đổi.

## 5. Sử dụng trong pipeline

```powershell
python train_mid.py --data-root data --variant Mid32 --model c --seed 42 --output-dir runs/c_mid32_seed42
python train_mid.py --data-root data --variant Mid224 --model c --seed 42 --output-dir runs/c_mid224_seed42
```

`config.json` lưu `model: c`, phiên bản `ticknet-c-v1`, số tham số, MACs,
FLOPs và phạm vi phép đếm. Pipeline kiểm tra giới hạn trước khi huấn luyện,
lưu checkpoint và kiểm tra phiên bản kiến trúc khi nạp lại checkpoint C.

Chưa chạy huấn luyện đầy đủ trên Mid32/Mid224; các lệnh trên là bước thực
nghiệm tiếp theo. Chọn mô hình và siêu tham số bằng validation tách từ train;
test chỉ đánh giá sau khi chốt cấu hình. Mọi nhận xét về accuracy của C còn
phải được chứng minh bằng thực nghiệm.
