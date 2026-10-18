# 03. Kiến trúc mô hình

[models/ticknet_l.py](../../models/ticknet_l.py) viết tường minh từng khối, thuận tiện kiểm tra/vấn đáp. [ticknet_l_large_v2.py](../../models/ticknet_l.py) giữ builder tương thích trainer và checkpoint final.

| Thành phần | Kênh đầu ra | Khối PDP | Feature map |
|---|---:|---:|---|
| Stem | 32 | — | 32×32 |
| Stage 1 | 160 | 1 | 32×32 |
| Stage 2 | 128 | 2 | 32×32 |
| Stage 3 | 256 | 5 | 16×16 |
| Stage 4 | 512 | 5 | 8×8 |
| Stage 5 | 768 | 1 | 4×4 |
| Final 1×1 + global pooling | 1024 | — | 1×1 |
| Classifier | 10/100 | — | Logits theo lớp |

Khối PDP dùng pointwise 1×1 nén kênh tuyến tính, depthwise, pointwise projection, SE và residual. Stage 1–2 dùng depthwise 3×3; stage 3–5 chia kênh cho kernel 3×3/5×5. Khối đổi kích thước/kênh dùng projection shortcut. Residual/SE kế thừa thiết kế gốc; phân bổ kênh/độ sâu, nén kênh và mixed kernel là phần phát triển trong dự án.

| Dataset | Tham số | Forward FLOPs |
|---|---:|---:|
| CIFAR-10 | 4.453.650 | 788.601.088 |
| CIFAR-100 | 4.545.900 | 788.785.408 |

Quy ước: **1 MAC=2 FLOPs**, batch 1, input (1,3,32,32), eval forward, chỉ Conv2d/Linear; không tính bias, BN, activation, pooling, cộng residual, nhân SE và di chuyển dữ liệu. [Profile đã lưu](../05_results/report_assets/model_profiles_final.json). Script checkpoint tính lại với bộ đếm toán tử PyTorch.

Hai file `checkpoint10.py`/`checkpoint100.py` load trọng số, tính tham số/FLOPs trên GPU và kiểm tra giới hạn trước inference. [Phân tích đầy đủ](../04_training/TRAINING_AND_MODEL_REPORT.md).
