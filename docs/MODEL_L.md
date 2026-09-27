# Thiết kế TickNet-L v1 từ TickNet-Basic

## 1. Mục tiêu và mốc so sánh

Chọn TickNet-Basic trong `models/TickNet.py` làm baseline; mã backbone gốc
được giữ nguyên. Cả Basic và L trong phép so sánh đều có đầu ra **5 lớp**.
L phải có không quá 6.000.000 tham số học được và dưới 1.000.000.000 FLOPs;
mục tiêu thiết kế là khoảng **0,8 GFLOPs cho một ảnh 224x224**.

Đây là ứng viên kiến trúc đã được kiểm tra tính toán và gradient. Chưa chạy
thí nghiệm huấn luyện đầy đủ, chưa có kết quả accuracy để kết luận L tốt hơn Basic.

Ứng viên [TickNet-C](MODEL_C.md) dùng trực tiếp lớp TickNet và block FR-PDP
nguyên bản của thầy, với khoảng 5,16 triệu tham số. C được cung cấp như một
lựa chọn riêng bên cạnh L trong tài liệu này.

## 2. Những gì kế thừa và những gì thay đổi

Basic đã có SE attention, residual, depthwise convolution, data batch normalization
và global average pooling. L kế thừa những thành phần này; **SE và đầu ra 5 lớp
không được trình bày là cải tiến mới**.

| Thành phần | TickNet-Basic | TickNet-L v1 | Lý do thiết kế |
|---|---|---|---|
| Stem | 32 kênh | 24 kênh | Giảm chi phí khi ảnh còn lớn |
| Kênh ở 5 stage | 128, 64, 128, 256, 512 | 112, 64, 144, 288, 512 | Thu hẹp đỉnh sớm, chuyển sức biểu diễn sang các stage sau |
| Số block mỗi stage | 1, 1, 1, 1, 1 | 1, 1, 2, 2, 1 | Bổ sung bước xử lý tại bản đồ đặc trưng nhỏ hơn |
| Pointwise đầu block | Cin → Cin | Cin → hidden, tỷ lệ 0,75 từ stage 2 | Giảm phép nhân ở pointwise tốn kém |
| Depthwise | 3x3 trên mọi kênh | Stage 1–2: 3x3; stage 3–5: nửa kênh 3x3, nửa kênh 5x5 | Kết hợp hai vùng tiếp nhận không gian với chi phí thấp |
| Head trước pooling | 1024 kênh | 768 kênh | Giảm phép tính cuối backbone |
| Shortcut khi cùng shape | Identity | Identity, không tạo projection dư | Mọi tham số được tạo đều tham gia forward |

Trong Basic ở 224x224, stage 2 chiếm khoảng **0,521 GFLOPs**, tương đương
52,7% chi phí Conv/Linear toàn mạng. Pointwise đầu stage này chạy trên đầu vào
112x112 với 128 kênh, trước khi depthwise giảm kích thước. L dùng 112 kênh đầu
vào và 88 kênh ẩn tại đây, giảm chi phí stage 2 còn khoảng **0,333 GFLOPs**.
Khoản tính toán tiết kiệm được cho phép thêm block tại stage 3 và stage 4.

Đây là lý do phân bổ kiến trúc, không phải bằng chứng thực nghiệm về accuracy.

## 3. Cấu trúc block và kích thước

Pointwise đầu giữ tính tuyến tính như Basic, đồng thời cho phép giảm số kênh:

```text
                    +---------- shortcut identity / PW 1x1 ----------+
                    |                                                 |
input -> PW 1x1 linear -> DW (3x3 hoặc chia kênh 3x3/5x5)              |
                            -> BN + ReLU -> PW 1x1 + BN + ReLU -> SE -> cộng
```

Với mixed depthwise, chia tensor theo trục kênh thành hai phần bằng nhau;
mỗi nhánh xử lý phần kênh riêng rồi nối lại. Không nhân đôi toàn bộ kênh
cho mỗi kernel. Hai nhánh cùng stride và padding giữ cùng kích thước đầu ra.
Pointwise sau depthwise trộn thông tin giữa các nhóm kênh.

`hidden = max(16, floor((0,75 * Cin + 4) / 8) * 8)` từ stage 2;
stage 1 dùng `hidden = Cin = 24`. Quy tắc làm tròn đảm bảo chia kênh đều.

| Phần | Kênh đầu ra | Số block | Kênh ẩn theo block | Kernel DW | Spatial Mid224 | Spatial Mid32 |
|---|---:|---:|---|---|---|---|
| Stem 3x3 | 24 | — | — | — | 112x112 | 32x32 |
| Stage 1 | 112 | 1 | 24 | 3x3 | 112x112 | 32x32 |
| Stage 2 | 64 | 1 | 88 | 3x3 | 56x56 | 32x32 |
| Stage 3 | 144 | 2 | 48, 112 | 3x3 + 5x5 | 28x28 | 16x16 |
| Stage 4 | 288 | 2 | 112, 216 | 3x3 + 5x5 | 14x14 | 8x8 |
| Stage 5 | 512 | 1 | 216 | 3x3 + 5x5 | 7x7 | 4x4 |
| Head 1x1 | 768 | — | — | — | 7x7 | 4x4 |
| Global average pooling | 768 | — | — | — | 1x1 | 1x1 |
| Classifier 1x1 | 5 | — | — | — | 1x1 | 1x1 |

Hai phiên bản dùng cùng cấu trúc kênh/block. Stride của stem và stage 2
được điều chỉnh cho Mid32 theo cách backbone gốc xử lý ảnh nhỏ.

## 4. Quy ước đo và kết quả

- Một forward ở chế độ `eval`, batch size 1, RGB, float32, 5 lớp, CPU.
- MAC của Conv2d = số phần tử output × (Cin / groups) × Kh × Kw.
- MAC của Linear = số phần tử output × số đầu vào.
- **1 MAC = 2 FLOPs**. G = 1.000.000.000; M = 1.000.000.
- Cộng toàn bộ Conv2d/Linear được thực thi, gồm stem, hai nhánh DW,
  shortcut projection, các Linear trong SE và classifier.
- Không tính bias addition, BatchNorm, activation, pooling, cộng residual,
  nhân scale SE hay vận chuyển bộ nhớ. Đây là phép đếm Conv/Linear có phạm vi
  xác định, không phải tổng tất cả phép toán hoặc thời gian chạy thực tế.
- Đếm tham số bằng `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- Đối chiếu bộ đếm hook theo module với `torch.utils.flop_counter.FlopCounterMode`
  theo toán tử; cả bốn cấu hình cho cùng kết quả. Không cần cài thêm profiler.

| Mô hình | Đầu vào | Tham số học được | GMACs | GFLOPs (2 × MACs) |
|---|---|---:|---:|---:|
| TickNet-Basic | 32x32 | 1.062.223 | 0,079214 | 0,158428 |
| TickNet-L v1 | 32x32 | 1.096.260 | 0,078910 | 0,157821 |
| TickNet-Basic | 224x224 | 1.062.223 | 0,494171 | 0,988343 |
| TickNet-L v1 | 224x224 | 1.096.260 | 0,398380 | 0,796760 |

L giảm khoảng **19,4% FLOPs ở 224x224**, giảm khoảng **0,38% ở 32x32**,
và tăng khoảng **3,2% tham số** do bổ sung block. Cả hai đầu vào đều đáp ứng
giới hạn theo quy ước trên; cấu hình 224x224 còn khoảng 20,3% so với ngưỡng 1G.
Không suy ra giảm latency hay tăng accuracy chỉ từ các số đo này.

Số nguyên đầy đủ, chi phí từng layer/stage và phiên bản PyTorch được lưu trong
[`model_profiles.json`](model_profiles.json). Tái lập:

```powershell
conda activate fresher
python profile_mid.py --output docs/model_profiles.json
python -m pytest -q
```

Kết quả tại mốc triển khai L v1 ngày 2026-09-27: **24 tests passed**. Bao gồm forward/backward
ở cả 32x32 và 224x224 với gradient hữu hạn cho mọi tham số của L; phép đếm
grouped convolution/Linear đối chiếu công thức; huấn luyện fixture một epoch
và nạp lại checkpoint Basic/L; từ chối checkpoint L sai phiên bản kiến trúc.
Profiler cũng được kiểm tra không làm đổi trọng số, thống kê BatchNorm hay
trạng thái train/eval của mô hình.

## 5. Tích hợp huấn luyện và bước thực nghiệm tiếp theo

```powershell
python train_mid.py --data-root data --variant Mid32 --model l --seed 42 --output-dir runs/l_mid32_seed42
python train_mid.py --data-root data --variant Mid224 --model l --seed 42 --output-dir runs/l_mid224_seed42
```

Các lệnh trên dành cho thí nghiệm sau bước thiết kế. `--model basic` vẫn là
mốc so sánh, và mặc định CLI vẫn là Basic để giữ hành vi các lệnh cũ.
`config.json` lưu phiên bản `ticknet-l-v1`, tham số, MACs, FLOPs và phạm vi đếm.
Pipeline kiểm tra giới hạn L trước huấn luyện, lưu checkpoint và kiểm tra
phiên bản kiến trúc khi nạp checkpoint L. Chỉ đánh giá test ở cuối quá trình.

Để đánh giá thiết kế: dùng cùng manifest, seed và điều kiện train cho Basic/L;
nếu cần tuning thì tách validation từ train. Có thể làm ablation tắt mixed
kernel hoặc bỏ các block lặp để phân tích đóng góp của từng thay đổi.
Những thí nghiệm này **chưa được thực hiện**. Kiểm thử gradient và huấn luyện
fixture nhỏ chỉ xác nhận pipeline hoạt động.

## 6. Nguồn và cách trình bày đóng góp

- Baseline: code TickNet-Basic của repository gốc, gồm `TickNet.py`,
  `common.py`, `SE_Attention.py`; thông tin bài báo có trong README.
- Ý tưởng chia nhóm kênh cho depthwise kernel khác nhau dựa trên
  [MixConv — Tan và Le, 2019](https://arxiv.org/abs/1907.09595).

Đóng góp trong bài làm là cấu hình backbone L và cách kết hợp/phân bổ block
khác với Basic đã chọn. Không khẳng định phát minh SE, mixed depthwise hay
một họ mạng chưa từng tồn tại. Việc đáp ứng yêu cầu khác biệt kiến trúc của
bài thi còn cần được trình bày bằng sơ đồ, bảng so sánh và kết quả thực nghiệm.
