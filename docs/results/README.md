# Tổng kết kết quả TickNet

Báo cáo mới nhất: [Kiểm định Basic/C/L ngày 2026-10-01](AUDIT_TICKNET_2026-10-01.md).
Báo cáo đối chiếu kiến trúc gốc, notebook Kaggle, checkpoint, số liệu và mức
độ thuyết phục của phương pháp phát triển TickNet-L.

| Mô hình | Top-1 Mid32 | Top-1 Mid224 | Tham số | GFLOPs Mid224 |
|---|---:|---:|---:|---:|
| Basic | 89,6% | 92,4% | 1.062.223 | 0,988343 |
| C | 91,2% | 94,4% | 5.155.467 | 0,821054 |
| L | 91,6% | 95,6% | 1.096.260 | 0,796760 |

Kết quả một seed, 250 ảnh test; FLOPs chỉ tính Conv/Linear với 1 MAC = 2 FLOPs.
Danh sách chia train/test đã xác minh khớp cả ba mô hình. L Mid32 khác batch
size, L khác số workers; chưa có nhiều seed hoặc ablation tách từng thay đổi.
Đánh giá lại sáu checkpoint cho Top-1 và confusion matrix khớp toàn bộ báo
cáo. Kiểm định ghép cặp chưa xác lập khác biệt có ý nghĩa ở mức 5%.

- [Bảng so sánh và biểu đồ](model_comparison/README.md).
- [Basic](model_basic/README.md), [C](model_c/README.md),
  [L](ticknet_l_midterm_seed42_20260928/README.md).
- [Checkpoint và checksum](checkpoints/README.md).
- [Mã kiểm chứng và bằng chứng](audit_20261001/).
