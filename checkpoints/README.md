# Final CIFAR checkpoints — TickNet-L Large v2 (14 khối)

Hai checkpoint final được chọn từ **bốn run của TickNet-L Large v2**: SGD và Adam trên CIFAR-10, SGD và Adam trên CIFAR-100. Chọn **một checkpoint tốt nhất cho mỗi dataset**, dựa trên validation Top-1; nếu hòa, chọn validation loss thấp hơn. Test chỉ dùng để báo cáo sau khi lựa chọn.

## Hai checkpoint được chọn

| Dataset | Optimizer | Learning rate | Weight decay | Best val epoch | Val Top-1 | Test Top-1 | Test loss | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CIFAR-10** | **SGD** | 0.175 | 0.0001 | **187** | **96.90%** | **96.04%** | 0.172538 | 0.960355 |
| **CIFAR-100** | **SGD** | 0.175 | 0.0005 | **198** | **80.58%** | **81.23%** | 0.700414 | 0.812142 |

- **CIFAR-10:** SGD cao hơn Adam **1,24 điểm phần trăm trên validation** (96,90% so với 95,66%), nên chọn checkpoint SGD ở epoch 187.
- **CIFAR-100:** SGD cao hơn Adam **3,38 điểm phần trăm trên validation** (80,58% so với 77,20%), nên chọn checkpoint SGD ở epoch 198. Test Top-1 đạt **81,23%**, vượt mục tiêu 80%.

Dùng **`best_val.pt`** của hai run này cho inference và báo cáo kết quả. `last.pt` là trạng thái cuối epoch 200 để lưu quá trình train và phục vụ resume; không phải checkpoint được chọn để báo cáo test.

Tên run, đường dẫn checkpoint và SHA-256 lấy từ [selected_checkpoints.json](../docs/experiments/selected_checkpoints.json). Các số liệu trong README này thuộc **mô hình 14 khối**.

## Kiến trúc

TickNet-L Large v2 có độ sâu theo stage **(1, 2, 5, 5, 1)**, tổng cộng **14 khối PDP**, giữ thiết kế pointwise–depthwise–pointwise, depthwise nhiều kích thước kernel và SE.

| Dataset | Số tham số | GFLOPs / ảnh 32×32 |
| --- | ---: | ---: |
| CIFAR-10 | **4.453.650** | **0,788601088** |
| CIFAR-100 | **4.545.900** | **0,788785408** |

Cả hai cấu hình nằm dưới giới hạn **6 triệu tham số** và **1 GFLOP**. FLOPs được tính cho một lần forward, batch size 1, theo quy ước **1 MAC = 2 FLOPs**, chỉ tính Conv2d và Linear. Không tính BN, activation, pooling, cộng residual, nhân SE, bias và di chuyển dữ liệu. Xem [model_profiles_final.json](../docs/experiments/report_assets/model_profiles_final.json).

## Cấu hình train của hai run được chọn

- **Dữ liệu:** mỗi dataset có 50.000 ảnh train gốc, chia phân tầng thành 45.000 ảnh train và 5.000 ảnh validation; test chính thức gồm 10.000 ảnh.
- **Thiết lập chung:** seed 42, batch size 128, 200 epochs; SGD với learning rate ban đầu 0.175, momentum 0.9 và Nesterov; cosine learning rate giảm đến 0. Runtime ghi trong config là Tesla T4.
- **CIFAR-10:** weight decay 0.0001; random crop, horizontal flip và Cutout độ dài 16.
- **CIFAR-100:** weight decay 0.0005; RandAugment với 2 phép biến đổi, magnitude 7 và CutMix với α = 1, xác suất 0,5.
- **Validation và test:** không dùng augmentation train.

RandAugment và CutMix là kỹ thuật tăng cường dữ liệu; **optimizer của cả hai checkpoint được chọn là SGD**.

## Artifact cần giữ cùng checkpoint

Khi lưu trữ hai run được chọn, giữ nguyên các file output để có thể đối chiếu kết quả và trạng thái train:

```text
best_val.pt
last.pt
config.json
epochs.csv
progress.json
completion.json
test_metrics.json
confusion_matrix.csv
test_predictions.csv
```

Không sửa metadata checkpoint để làm mất dấu cấu hình hoặc mã nguồn đã dùng khi train. Khi đánh giá lại hoặc resume, dùng đúng kiến trúc và tuân thủ kiểm tra tương thích của trainer.

## Báo cáo và kiểm tra

Xem [báo cáo experiment mới](../docs/experiments/README.md) để so sánh đầy đủ bốn run, learning curves, confusion matrix và kết quả theo lớp. [archive_manifest.json](../docs/experiments/archive_manifest.json) trong bộ báo cáo mới chứa SHA-256 của **36 file gốc thuộc bốn run**; phạm vi này khác với hai checkpoint được chọn trong README này. Các notebook đã chạy được giữ ở [notebooks](../docs/experiments/notebooks/).

Từ thư mục gốc repo, chạy bước xác thực output và dựng lại báo cáo:

```bash
python scripts/build_large_v2_experiment_report.py
```

Script cần đủ bốn thư mục run trong `runs/` và **không huấn luyện lại**. Theo báo cáo experiment, các output đã được đối chiếu về 200 epochs, lựa chọn best checkpoint, khả năng load vào mô hình 14 khối, 10.000 dự đoán test mỗi run, confusion matrix và các chỉ số đã lưu.

Các số liệu test ở đây là **kết quả Kaggle đã lưu và được kiểm tra tính nhất quán**. Lần tạo báo cáo chưa chạy lại full inference CIFAR-100 vì workspace không có sẵn ảnh CIFAR-100.
