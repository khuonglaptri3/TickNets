# Thiết Kế Hệ Thống Thực Nghiệm Lưới (Grid Search) Cho Kỳ Thi Cuối Kỳ

**Môn học:** Deep Learning Final Examination  
**Ngày lập:** 2026-10-05  
**Kiến trúc mục tiêu:** `TickNet-L v1` ([`models/ticknet_l.py`](../../../models/ticknet_l.py))
**Tập dữ liệu:** CIFAR-10 (10 lớp) và CIFAR-100 (100 lớp)  

---

## 1. Mục Tiêu & Cơ Sở Thiết Kế

Đề bài Cuối kỳ ([`.doc/Final exam.docx`](../../../.doc/Final%20exam.docx)) yêu cầu train/test hai bộ CIFAR, ngân sách mô hình và khảo sát optimizer/LR. Giao thức nhóm lựa chọn:
1. Huấn luyện mô hình $L$ (`TickNet-L`) từ đầu trên **CIFAR-10** và **CIFAR-100**; from-scratch là lựa chọn thực nghiệm, không phải điều kiện được ghi trong DOCX.
2. Ràng buộc phần cứng: **Số tham số $\le 6M$**, **FLOPs $< 1G$**.
3. Khảo sát có hệ thống giữa các mức Learning Rate (ví dụ `0.1`, `0.15`,...) và cả 2 bộ tối ưu hóa: **SGD** và **Adam**.
4. Báo cáo chi tiết các thông số (momentum, learning rate, epochs, weight decay, loss, accuracy).

### Ma Trận Thí Nghiệm (8 Cấu hình Grid Search)

| Mã thí nghiệm (Config ID) | Tập dữ liệu | Optimizer | Learning Rate ban đầu | Momentum / Betas | Weight Decay | Scheduler (200 Epochs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cifar10_sgd_lr010` | CIFAR-10 | SGD | `0.10` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar10_sgd_lr015` | CIFAR-10 | SGD | `0.15` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar10_adam_lr0001` | CIFAR-10 | Adam | `0.001` (1e-3) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar10_adam_lr00003` | CIFAR-10 | Adam | `0.0003` (3e-4) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar100_sgd_lr010` | CIFAR-100 | SGD | `0.10` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar100_sgd_lr015` | CIFAR-100 | SGD | `0.15` | momentum=0.9 | 1e-4 | CosineAnnealingLR ($T_{\max}=200$) |
| `cifar100_adam_lr0001` | CIFAR-100 | Adam | `0.001` (1e-3) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |
| `cifar100_adam_lr00003` | CIFAR-100 | Adam | `0.0003` (3e-4) | betas=(0.9, 0.999) | 1e-4 | CosineAnnealingLR ($T_{\max}=200, \eta_{\min}=0$) |

---

## 2. Kiến Trúc Kỹ Thuật Module Huấn Luyện `train_cifar.py`

### 2.1. Quản lý Siêu Tham Số & Cấu hình CLI / JSON
- Hỗ trợ tải cấu hình từ JSON file (`--config configs/final/<id>.json`).
- CLI có thể ghi đè các tham số: `--dataset`, `--optimizer`, `--learning-rate`, `--epochs`, `--batch-size`, `--val-fraction`, `--seed`, `--output-dir`.

### 2.2. Chiến Lược Phân Chia Tập Dữ Liệu (Split Policy)
- Sử dụng phân tầng stratified: `val_fraction=0.1` (45.000 train / 5.000 val trên CIFAR-10; 450 train / 50 val mỗi lớp trên CIFAR-100).
- Checkpoint được tuyển chọn hoàn toàn dựa trên tập validation (`best_val.pt` được chọn theo tiêu chí `val_top1` cao nhất, giải hòa bằng `val_loss` thấp hơn).
- Tập test chính thức (10.000 ảnh) tuyệt đối không tham gia vào quá trình chọn epoch hoặc tinh chỉnh siêu tham số.

### 2.3. Đầu Ra Dữ Liệu (Output Artifacts)
Mỗi thư mục output (`--output-dir`) sẽ tạo ra các file chuẩn:
1. `config.json`: Toàn bộ cấu hình huấn luyện, git hash, model complexity (tham số, FLOPs), phiên bản thư viện.
2. `epochs.csv`: Lịch sử từng epoch (epoch, learning_rate, train_loss, train_top1, val_loss, val_top1).
3. `best_val.pt`: Trọng số checkpoint tốt nhất trên tập validation.
4. `last.pt`: Checkpoint đầy đủ (model, optimizer, scheduler, rng) để resume nếu bị ngắt phiên trên Kaggle.
5. `test_metrics.json`: Kết quả đánh giá cuối cùng trên 10.000 ảnh test (Top-1, loss, macro_f1).
6. `confusion_matrix.csv`: Ma trận nhầm lẫn kích thước $10 \times 10$ (CIFAR-10) hoặc $100 \times 100$ (CIFAR-100).

---

## 3. Khả Năng Khôi Phục & Triển Khai Kaggle (Resume Capability)
- Cờ `--resume <path/to/last.pt>`: Khôi phục chính xác trạng thái model, optimizer, scheduler, và generator tại ranh giới epoch bị ngắt.
- Cờ `--stop-after-epoch <N>`: Dành cho việc chạy smoke test hoặc chạy từng chặng epoch trên môi trường giới hạn thời gian (GPU timeout).
- Cờ `--evaluate <checkpoint>`: Chạy suy luận độc lập trên tập test 10.000 ảnh và xuất ma trận nhầm lẫn.
