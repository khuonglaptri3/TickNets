# Kế Hoạch Triển Khai Thực Nghiệm Lưới (Grid Search) CIFAR-10 & CIFAR-100

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hoàn chỉnh hạ tầng huấn luyện và thực nghiệm lưới (Grid Search) cho mô hình `TickNet-L` trên cả CIFAR-10 và CIFAR-100, khảo sát cả 2 optimizer SGD và Adam với các dải learning rate khác nhau, hỗ trợ chạy trên máy nội bộ và Kaggle GPU.

**Architecture:** Tạo script `train_cifar.py` kế thừa chuẩn kỹ thuật của TickNets (quản lý seed, logging từng epoch, lưu `best_val.pt` theo validation, checkpoint `last.pt` có khả năng resume, tính toán độ phức tạp mô hình). Tạo 8 file cấu hình JSON cho 8 thực nghiệm. Tạo unit tests kiểm định quy trình và notebook Kaggle để chạy thực tế trên GPU.

**Tech Stack:** Python 3.11, PyTorch, TorchVision, PyTest.

**Spec:** `docs/superpowers/specs/2026-10-05-cifar-grid-search-design.md`

## Global Constraints
- Kiến trúc mô hình: `TickNet-L v1` (`cifar=True`) cho cả 2 bộ dữ liệu.
- Ràng buộc tham số: $\le 6.000.000$ params (Hiện tại CIFAR-10: 1.100.105, CIFAR-100: 1.169.315).
- Ràng buộc FLOPs: $< 1.000.000.000$ FLOPs (Hiện tại ~0.158 GFLOPs).
- Optimizers: Bắt buộc hỗ trợ đầy đủ cả `sgd` và `adam`.
- Scheduler: `CosineAnnealingLR` với chu kỳ 200 epochs.
- Phân chia: `val_fraction=0.1` phân tầng, chọn checkpoint bằng validation, đánh giá test độc lập.

---

### Task 1: Xây dựng Module Huấn Luyện `train_cifar.py`

**Files:**
- Create: `train_cifar.py`

- [x] **Step 1.1**: Viết hàm `parse_args` hỗ trợ cả CLI và nạp cấu hình từ JSON.
- [x] **Step 1.2**: Viết hàm `run_epoch` tính toán loss, Top-1 accuracy, xử lý training và eval.
- [x] **Step 1.3**: Viết logic lưu trữ `epochs.csv`, `best_val.pt`, `last.pt`, `test_metrics.json`, `confusion_matrix.csv`, `config.json`.
- [x] **Step 1.4**: Viết hàm `evaluate` và cơ chế resume từ checkpoint.

---

### Task 2: Tạo Các File Cấu Hình `configs/final/*.json`

**Files:**
- Create: `configs/final/cifar10_sgd_lr010.json`
- Create: `configs/final/cifar10_sgd_lr015.json`
- Create: `configs/final/cifar10_adam_lr0001.json`
- Create: `configs/final/cifar10_adam_lr00003.json`
- Create: `configs/final/cifar100_sgd_lr010.json`
- Create: `configs/final/cifar100_sgd_lr015.json`
- Create: `configs/final/cifar100_adam_lr0001.json`
- Create: `configs/final/cifar100_adam_lr00003.json`

- [x] **Step 2.1**: Tạo 4 cấu hình cho CIFAR-10 (SGD lr 0.10, 0.15; Adam lr 1e-3, 3e-4).
- [x] **Step 2.2**: Tạo 4 cấu hình cho CIFAR-100 (SGD lr 0.10, 0.15; Adam lr 1e-3, 3e-4).

---

### Task 3: Viết Unit Tests `tests/test_train_cifar.py`

**Files:**
- Create: `tests/test_train_cifar.py`

- [x] **Step 3.1**: Test kiểm tra khởi tạo optimizer (SGD vs Adam với đúng momentum, betas, weight decay).
- [x] **Step 3.2**: Test chạy smoke training 1 epoch cho cả CIFAR-10 và CIFAR-100 trên tập dữ liệu nhỏ (tạo artifacts `epochs.csv`, `last.pt`, `best_val.pt`, `config.json`).
- [x] **Step 3.3**: Test kiểm tra chế độ `--evaluate` đọc checkpoint và xuất ra ma trận nhầm lẫn chuẩn $10 \times 10$ và $100 \times 100$.

---

### Task 4: Tạo Notebook Kaggle Cho Huấn Luyện GPU

**Files:**
- Create: `docs/kaggle/Kaggle_Final_Exam_Grid_Search.ipynb`

- [x] **Step 4.1**: Tạo notebook hoàn chỉnh có thể mở trên Kaggle với GPU T4 x2 hoặc P100.
- [x] **Step 4.2**: Tích hợp lệnh tự động chạy tuần tự 8 cấu hình và lưu kết quả vào `/kaggle/working/runs/`.

---

### Task 5: Chạy Kiểm Thử, Xác Nhận và Commit

- [x] **Step 5.1**: Chạy `python3 -m pytest tests/test_train_cifar.py -v`.
- [x] **Step 5.2**: Chạy lại toàn bộ test suite `python3 -m pytest tests -q`.
- [x] **Step 5.3**: Commit toàn bộ lên nhánh `feature/final-exam-model-l`.
