# Kế Hoạch Triển Khai Toàn Diện Nâng Cấp Hệ Thống Cuối Kỳ (Final Exam System Evolution)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện toàn diện hệ thống huấn luyện, kiểm thử, phân tích dữ liệu, module hóa notebook Kaggle và bộ tài liệu kỹ thuật cuối kỳ cho mô hình `TickNet-L v1` và mô hình nền tảng tác giả `TickNet-Basic` trên CIFAR-10 & CIFAR-100, đồng thời bảo tồn nguyên vẹn trích dẫn học thuật của tác giả gốc.

**Architecture:** Mở rộng pipeline tiền xử lý với Cutout ($16\times16$ theo DeVries & Taylor 2017), tích hợp Nesterov momentum cho SGD trong `train_cifar.py`, cấu hình ma trận 10 thí nghiệm, chia tách Kaggle thành 5 notebook độc lập (1 Baseline + 4 Phases), viết mới toàn bộ 11 tài liệu kỹ thuật chuyên sâu trong `docs/critical/`, tái cấu trúc `README.md` với đầy đủ hướng dẫn thực thi và khôi phục trích dẫn BibTeX/Abstract của bài báo gốc Neurocomputing 2024.

**Tech Stack:** Python 3.11, PyTorch 2.0+, TorchVision, NumPy, Pandas, PyTest, Jupyter Notebook, GitHub Flavored Markdown.

**Spec:** [`docs/superpowers/specs/2026-10-05-cifar-final-exam-full-system-spec.md`](../specs/2026-10-05-cifar-final-exam-full-system-spec.md)

## Global Constraints

- Ràng buộc tham số: $\le 6.000.000$ params (TickNet-L v1 đạt 1.100.105 trên CIFAR-10 và 1.169.315 trên CIFAR-100).
- Ràng buộc FLOPs: $< 1.000.000.000$ FLOPs (TickNet-L v1 đạt ~0.158 GFLOPs với chuẩn 1 MAC = 2 FLOPs).
- Không sử dụng pretrained weights (100% train from scratch, Kaiming Uniform/Normal).
- Phân chia tập dữ liệu 90% train / 10% val phân tầng stratified, bảo đảm zero data leakage.
- Giữ nguyên văn trích dẫn tác giả gốc Thanh Tuan Nguyen & Thanh Phuong Nguyen (Neurocomputing 2024).

## Review Focus

1. Kiểm tra tính độc lập và toàn vẹn của lớp biến đổi `Cutout(16x16)` khi chạy với tensor ảnh PyTorch.
2. Kiểm tra cờ `nesterov=True` chỉ được kích hoạt khi optimizer là `sgd` và có `momentum > 0`.
3. Kiểm tra tính khả thi của 4 phase Kaggle độc lập tránh lỗi GPU timeout 12h.
4. Kiểm tra sự đồng nhất giữa các thông số trong 10 file JSON cấu hình và mô tả tài liệu.
5. Kiểm tra tính tương thích ngược và 100% test pass của bộ 47 automated tests.

---

### Task 1: Tích Hợp Kỹ Thuật Cutout (DeVries & Taylor, 2017) Vào `models/cifar_data.py`

**Files:**
- Modify: `models/cifar_data.py`
- Test: `tests/test_cifar_data.py`

- [x] **Step 1.1**: Định nghĩa lớp `Cutout(n_holes=1, length=16)` tạo mặt nạ 0.0 ngẫu nhiên che phủ vùng $16\times16$ trên tensor ảnh $3\times32\times32$.
- [x] **Step 1.2**: Tích hợp tham số `use_cutout=True` và `cutout_length=16` vào hàm `build_cifar_transforms` cho pipeline huấn luyện của CIFAR-10 và CIFAR-100.
- [x] **Step 1.3**: Bổ sung cờ cấu hình `cutout` vào `get_cifar_dataloaders` và truyền giá trị cấu hình tương ứng.
- [x] **Step 1.4**: Viết unit test xác thực `Cutout` che đúng kích thước và không làm thay đổi shape của tensor.

---

### Task 2: Kích Hoạt Nesterov Momentum Cho SGD Trong `train_cifar.py`

**Files:**
- Modify: `train_cifar.py`
- Test: `tests/test_train_cifar.py`

- [x] **Step 2.1**: Cập nhật hàm `build_optimizer` trong `train_cifar.py` bổ sung tham số `nesterov=(args.momentum > 0)` khi `optimizer == 'sgd'`.
- [x] **Step 2.2**: Ghi nhận thuộc tính `nesterov` vào log cấu hình `config.json` của mỗi lần chạy.
- [x] **Step 2.3**: Viết test kiểm tra đối tượng `torch.optim.SGD` sinh ra có `param_groups[0]['nesterov'] == True`.

---

### Task 3: Đồng Bộ Toàn Bộ 10 File Cấu Hình Huấn Luyện `configs/final/*.json`

**Files:**
- Create/Modify: `configs/final/*.json`

- [x] **Step 3.1**: Cập nhật 8 file cấu hình Grid Search cho `TickNet-L` (`cifar10_sgd_lr010.json`, `cifar10_sgd_lr015.json`, `cifar10_adam_lr0001.json`, `cifar10_adam_lr00003.json`, `cifar100_sgd_lr010.json`, `cifar100_sgd_lr015.json`, `cifar100_adam_lr0001.json`, `cifar100_adam_lr00003.json`) tích hợp `"cutout": true, "cutout_length": 16`.
- [x] **Step 3.2**: Tạo 2 file cấu hình baseline cho mô hình của tác giả: `configs/final/baseline_cifar10_sgd_lr010.json` và `configs/final/baseline_cifar100_sgd_lr010.json` với `"model": "basic"`, `"optimizer": "sgd"`, `"learning_rate": 0.10`, `"momentum": 0.9`, `"weight_decay": 0.0001`, `"cutout": true`.

---

### Task 4: Module Hóa Notebook Kaggle (Author Baseline + 4 Phase Độc Lập)

**Files:**
- Delete: `docs/kaggle/baseline_Basic/`
- Create: `docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`
- Create: `docs/kaggle/Phase1_CIFAR10_SGD.ipynb`
- Create: `docs/kaggle/Phase2_CIFAR10_Adam.ipynb`
- Create: `docs/kaggle/Phase3_CIFAR100_SGD.ipynb`
- Create: `docs/kaggle/Phase4_CIFAR100_Adam.ipynb`

- [x] **Step 4.1**: Xóa thư mục cũ giữa kỳ `docs/kaggle/baseline_Basic/` không còn phù hợp với yêu cầu cuối kỳ.
- [x] **Step 4.2**: Xây dựng notebook `Kaggle_Author_TickNet_Baseline.ipynb` huấn luyện mô hình gốc `TickNet-Basic` của tác giả trên CIFAR-10 và CIFAR-100 với SGD Nesterov lr=0.10, Cutout 16, tự động nén kết quả thành `author_ticknet_baseline_results.zip`.
- [x] **Step 4.3**: Tách nhỏ ma trận Grid Search thành 4 Phase độc lập:
  - `Phase1_CIFAR10_SGD.ipynb`: Chạy 2 thử nghiệm CIFAR-10 (SGD lr=0.10, lr=0.15) $\to$ xuất file zip `phase1_cifar10_sgd_results.zip`.
  - `Phase2_CIFAR10_Adam.ipynb`: Chạy 2 thử nghiệm CIFAR-10 (Adam lr=0.001, lr=0.0003) $\to$ xuất file zip `phase2_cifar10_adam_results.zip`.
  - `Phase3_CIFAR100_SGD.ipynb`: Chạy 2 thử nghiệm CIFAR-100 (SGD lr=0.10, lr=0.15) $\to$ xuất file zip `phase3_cifar100_sgd_results.zip`.
  - `Phase4_CIFAR100_Adam.ipynb`: Chạy 2 thử nghiệm CIFAR-100 (Adam lr=0.001, lr=0.0003) $\to$ xuất file zip `phase4_cifar100_adam_results.zip`.

---

### Task 5: Viết Lại và Đồng Bộ Toàn Bộ 11 Tài Liệu Kỹ Thuật Cốt Lõi `docs/critical/*.md`

**Files:**
- Overwrite: `docs/critical/*.md` (11 files)

- [x] **Step 5.1**: Đồng bộ `COMMANDS_GUIDE.md` với các cờ CLI mới nhất của `train_cifar.py`, hướng dẫn tải dữ liệu, chạy 4 phase Kaggle và trích xuất artifact.
- [x] **Step 5.2**: Đồng bộ `MODEL_L.md` với thiết kế 7 block, channel elasticity `[112, 64, 144, 288, 512]`, chứng minh tham số $\le 6$M và FLOPs $< 1$G.
- [x] **Step 5.3**: Đồng bộ `DATALOADER.md` với cơ chế nạp dữ liệu phân tầng 90/10 trên 45.000 train / 5.000 val cho CIFAR.
- [x] **Step 5.4**: Đồng bộ `PREPROCESSING.md` với đặc tả toán học của Cutout 16x16 và ma trận chuẩn hóa màu.
- [x] **Step 5.5**: Đồng bộ `HYPERPARAMETER_TUNING.md` với phân tích so sánh SGD Nesterov vs Adam và cơ chế Cosine Annealing.
- [x] **Step 5.6**: Đồng bộ `MODEL_INITIALIZATION.md` với quy tắc khởi tạo Kaiming Normal/Uniform cho huấn luyện from scratch.
- [x] **Step 5.7**: Đồng bộ `TRAINING_AND_ARCHITECTURE.md` với bảng so sánh trực diện TickNet-L v1 và TickNet-Basic.
- [x] **Step 5.8**: Đồng bộ `VALIDATION_AND_METRICS.md` với công thức Top-1, Loss, Macro-F1 và Ma trận nhầm lẫn.
- [x] **Step 5.9**: Đồng bộ `DATASET_SPLIT.md` và `DATASET_SPLITTING.md` với bảng kê chi tiết số lượng mẫu và cam kết zero leakage.
- [x] **Step 5.10**: Đồng bộ `DATASET_CLEANING.md` với kiểm tra tính toàn vẹn MD5 checksum từ máy chủ Đại học Toronto.

---

### Task 6: Viết Lại `README.md` Toàn Diện & Khôi Phục Trích Dẫn Học Thuật Của Tác Giả

**Files:**
- Overwrite: `README.md`

- [x] **Step 6.1**: Viết lại cấu trúc tổng quan `README.md` bao gồm: Giới thiệu mục tiêu cuối kỳ, phương pháp luận kiến trúc, chiến lược huấn luyện Cutout, ma trận Grid Search, hướng dẫn thực thi CLI & Kaggle, danh mục artifacts đầu ra và sitemap tài liệu.
- [x] **Step 6.2**: Khôi phục lại khối trích dẫn học thuật của bài báo tác giả gốc:
  - Huy hiệu DOI `10.1016/j.neucom.2024.127942` và hộp thông báo nguồn gốc bài báo ngay đầu file.
  - Mục chuyên biệt `## 8. Original Paper, Abstract & Citation` lưu trữ đầy đủ thông tin bài báo, nguyên văn Abstract của tác giả và khối mã BibTeX chuẩn:
    ```bibtex
    @article{neucoTickNetNguyen23,
      author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
      title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
      journal      = {Neurocomputing},
      volume       = {596},
      pages        = {127942},
      year         = {2024},
      url          = {https://doi.org/10.1016/j.neucom.2024.127942}
    }
    ```

---

### Task 7: Mở Rộng Bộ Kiểm Thử Tự Động Toàn Hệ Thống

**Files:**
- Update: `tests/test_cifar_data.py`
- Update: `tests/test_train_cifar.py`

- [x] **Step 7.1**: Kiểm thử kiểm tra toàn bộ pipeline Cutout, stratified splitting, normalization.
- [x] **Step 7.2**: Kiểm thử tính toán FLOPs và parameter count cho cả `basic` và `l` trên cả CIFAR-10 và CIFAR-100.
- [x] **Step 7.3**: Kiểm thử smoke training 1 epoch cho cả SGD Nesterov và Adam, kiểm tra tính toàn vẹn của artifacts đầu ra.
- [x] **Step 7.4**: Chạy toàn bộ test suite `PYTHONPATH=. pytest -v` $\to$ **Xác nhận 47/47 tests passed (100%)**.

---

### Task 8: Quản Trị Nhánh Git & Đồng Bộ Remote GitHub

- [x] **Step 8.1**: Commit toàn bộ các thay đổi vào nhánh `feature/final-exam-model-l`.
- [x] **Step 8.2**: Đẩy lên remote repository `origin/feature/final-exam-model-l` và xác nhận commit head mới nhất `6919d57d`.
