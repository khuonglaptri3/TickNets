# Tổng kết kết quả TickNet

Tài liệu này tổng hợp kết quả đối chiếu giữa mô hình cơ sở **TickNet-Basic** (mô hình gốc của thầy) và mô hình cải tiến **TickNet-L v1** (Model L được lựa chọn cho đồ án Giữa kỳ và Cuối kỳ).

| Mô hình | Top-1 Mid32 | Top-1 Mid224 | Tham số | GFLOPs Mid32 | GFLOPs Mid224 | Trạng thái đề tài |
|---|---:|---:|---:|---:|---:|:---:|
| **TickNet-Basic** (Gốc của thầy) | 89,6% | 92,4% | 1.062.223 | 0,158428 | 0,988343 *(sát trần 1G)* | Mốc đối chứng (Baseline) |
| **TickNet-L v1** (Đề xuất) | **91,6%** | **95,6%** | 1.096.260 | **0,157821** | **0,796760** *(nhẹ nhất)* | **Được chọn phát triển** |

Quy ước FLOPs: 1 MAC = 2 FLOPs, một ảnh, chế độ eval, chỉ tính Conv2d và Linear.

- **[Hồ sơ tổng hợp & Bằng chứng Model L (Được chọn phát triển)](Model_L/README.md)**
- **[Hồ sơ tổng hợp Baseline Model Basic của thầy](Model_Basic/README.md)**
- [Biên bản kiểm định độc lập ngày 2026-10-01](AUDIT_TICKNET_2026-10-01.md)
- [Bảng so sánh và đối chiếu](model_comparison/README.md)
- [Checkpoint và checksum SHA-256](checkpoints/README.md)
- [Mã nguồn kiểm chứng độc lập](audit_20261001/)

