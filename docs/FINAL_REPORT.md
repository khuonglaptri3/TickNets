# TickNet-L Large v2 — Báo cáo Final Project

## 1. Bài toán

Phân loại CIFAR-10/CIFAR-100 từ TickNet-Basic; tối đa 6 triệu tham số học được, dưới 1 tỷ FLOPs cho một ảnh RGB 32×32. [Yêu cầu](01_introduction/README.md).

## 2. Dữ liệu

Chia phân tầng 50.000 ảnh train gốc thành 45.000 train / 5.000 validation, seed 42; test chính thức 10.000 ảnh mỗi dataset. Validation/test chỉ tensor + normalize. CIFAR-10 train dùng crop/flip/Cutout; CIFAR-100 dùng crop/flip/RandAugment/CutMix. [Preprocessing](02_dataset/README.md).

## 3. Mô hình

TickNet-L Large v2 có 14 khối PDP, độ sâu stage (1,2,5,5,1), kênh (160,128,256,512,768). Dùng pointwise compression, mixed depthwise 3×3/5×5 ở stage sau, SE và residual. [Kiến trúc](03_model/README.md).

| Dataset | Tham số | GFLOPs | Giới hạn đề bài |
|---|---:|---:|---|
| CIFAR-10 | 4.453.650 | 0.788601088 | Đạt |
| CIFAR-100 | 4.545.900 | 0.788785408 | Đạt |

FLOPs tính Conv2d/Linear, batch 1, eval forward, 1 MAC=2 FLOPs; không tính bias, BN, activation, pooling, residual add, nhân SE, di chuyển dữ liệu.

## 4. Thí nghiệm

200 epochs, batch 128, seed 42, cross-entropy, cosine LR đến 0. SGD + Nesterov: LR=0.175, momentum 0.9. Adam: LR=0.001, β=(0.9,0.999), ε=1e-8. Weight decay 1e-4 cho CIFAR-10, 5e-4 cho CIFAR-100. [Cấu hình/notebook](04_training/README.md), [báo cáo train chi tiết](04_training/TRAINING_AND_MODEL_REPORT.md).

## 5. Kết quả

| Dataset | Optimizer | Best epoch | Val Top-1 | Test Top-1 | Test loss | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| CIFAR-10 | **SGD** | 187 | **96.90%** | **96.04%** | 0.172538 | 0.960355 |
| CIFAR-10 | Adam | 199 | 95.66% | 95.01% | 0.202167 | 0.950145 |
| CIFAR-100 | **SGD** | 198 | **80.58%** | **81.23%** | 0.700414 | 0.812142 |
| CIFAR-100 | Adam | 200 | 77.20% | 78.04% | 0.827812 | 0.780109 |

Chọn epoch và optimizer theo validation Top-1, hòa thì chọn loss thấp hơn; test báo cáo sau lựa chọn. Hai checkpoint SGD là trọng số nộp bài. [Phân tích/figures](05_results/README.md), [manifest](05_results/selected_checkpoints.json), [archive](05_results/checkpoints_4_phase/).

![So sánh CIFAR-10](05_results/report_assets/cifar10_optimizer_comparison.png)

![So sánh CIFAR-100](05_results/report_assets/cifar100_optimizer_comparison.png)

## 6. Thảo luận

SGD cao hơn Adam 1.03 điểm phần trăm trên test CIFAR-10, 3.19 điểm trên CIFAR-100. Hai cấu hình có LR khác nhau, chỉ một seed mỗi cấu hình; chưa tách tác động optimizer hoặc từng thay đổi kiến trúc. [Giới hạn kết luận](06_discussion/README.md).

## 7. Tham khảo

TickNet-Basic, full-residual PDP và SE kế thừa Nguyen & Nguyen (Neurocomputing, 2024). [Nguồn và trích dẫn](07_references/README.md).

## 8. Tái tạo

```bash
python -m pip install -r requirements.txt
python checkpoint10.py
python checkpoint100.py
```

Hai script nạp checkpoint nộp bài, kiểm tra dữ liệu, chạy 10.000 ảnh test, in kết quả mới và đối chiếu báo cáo; JSON/CSV lưu ở `runs/`. Bảng trên lấy từ kết quả Kaggle đã lưu; [phụ lục](08_appendix/README.md) ghi cách xác thực và kết quả chạy lại.

Đã tái tạo **96.04% trên CIFAR-10** và **81.23% trên CIFAR-100** bằng inference mới qua mirror có fingerprint dữ liệu trùng khớp. Accuracy, Macro-F1 và loss khớp báo cáo; xem [biên bản chạy lại](08_appendix/validation/README.md).
