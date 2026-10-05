# CIFAR-10 & CIFAR-100 DataLoader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng module DataLoader chuẩn hóa cho CIFAR-10 và CIFAR-100 hỗ trợ tải tự động qua torchvision, áp dụng đúng pipeline tiền xử lý (RandomCrop padding 4, RandomHorizontalFlip, Normalize), tùy chọn chia train/validation phân tầng, và viết unit test kiểm định toàn diện.

**Architecture:** Tạo module `models/cifar_data.py` cung cấp các hàm tải dữ liệu và cấu hình biến đổi ảnh cho CIFAR-10 và CIFAR-100. Tích hợp bộ tiền xử lý chuẩn mực (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`) với các giá trị mean/std chuẩn của từng bộ dữ liệu. Hỗ trợ tạo `train_loader`, `val_loader` (tùy chọn) và `test_loader` với batch size, workers, pin_memory và seed có khả năng tái lập.

**Tech Stack:** Python 3.11+, PyTorch 2.x, TorchVision, PyTest.

**Spec:** Yêu cầu từ `.doc/Final exam.docx` (huấn luyện và kiểm thử mô hình L trên CIFAR-10 và CIFAR-100).

## Global Constraints

- Hỗ trợ tải tự động qua `torchvision.datasets.CIFAR10` và `torchvision.datasets.CIFAR100` với cờ `download=True`.
- Tiền xử lý huấn luyện bắt buộc: `transforms.RandomCrop(32, padding=4)`, `transforms.RandomHorizontalFlip()`, `transforms.ToTensor()`, `transforms.Normalize(mean, std)`.
- Tiền xử lý kiểm thử: `transforms.ToTensor()`, `transforms.Normalize(mean, std)`.
- Giá trị mean và std chuẩn:
  - CIFAR-10: `mean=(0.4914, 0.4822, 0.4465)`, `std=(0.2470, 0.2435, 0.2616)`
  - CIFAR-100: `mean=(0.5071, 0.4867, 0.4408)`, `std=(0.2675, 0.2565, 0.2761)`
- Đảm bảo tính tái lập (reproducibility) thông qua `generator` và `worker_init_fn` với seed cố định.
- Tương thích 100% với kiến trúc `TickNet-L` (`cifar=True`).

## Review Focus

1. **Khả năng tự động tải (Auto-download)**: Khi chạy trong môi trường mới hoặc Kaggle, dữ liệu tự tải về `data_root` mà không bị crash.
2. **Kích thước và kiểu dữ liệu tensor**: Batch trả về từ dataloader phải có shape `[B, 3, 32, 32]`, dtype `torch.float32`, nhãn có shape `[B]`, dtype `torch.long`.
3. **Phân phối giá trị sau Normalize**: Tensor sau chuẩn hóa có kỳ vọng xấp xỉ 0 và độ lệch chuẩn xấp xỉ 1 trên toàn tập.
4. **Chia tập Train/Val phân tầng**: Khi kích hoạt `val_fraction > 0`, tỷ lệ giữa các lớp trong train và val phải được bảo toàn cân bằng.
5. **Data augmentation không áp dụng lên test**: Tập test chỉ chuyển sang tensor và chuẩn hóa, không crop ngẫu nhiên hay lật ảnh.

---

### Task 1: Thiết kế và Xây dựng `models/cifar_data.py`

**Files:**
- Create: `models/cifar_data.py`

- [x] **Step 1.1**: Định nghĩa các hằng số chuẩn hóa `CIFAR10_MEAN`, `CIFAR10_STD`, `CIFAR100_MEAN`, `CIFAR100_STD` và hàm `get_cifar_transforms(dataset_name, augment=True)`.
- [x] **Step 1.2**: Xây dựng hàm `build_cifar_datasets(data_root, dataset_name, val_fraction=0.0, seed=42, download=True)` xử lý tải và chia tập (nếu có validation).
- [x] **Step 1.3**: Xây dựng hàm chính `build_cifar_loaders(data_root, dataset_name, batch_size=128, val_fraction=0.0, seed=42, num_workers=2, pin_memory=True, download=True)` trả về `(train_loader, val_loader, test_loader)`.

---

### Task 2: Viết Unit Tests kiểm định toàn diện `tests/test_cifar_data.py`

**Files:**
- Create: `tests/test_cifar_data.py`

- [x] **Step 2.1**: Viết test kiểm tra hàm transforms (shape, type, augment vs eval transforms).
- [x] **Step 2.2**: Viết test mock dataset để kiểm tra logic chia train/validation phân tầng mà không cần tải 150MB dữ liệu thật trong unit test.
- [x] **Step 2.3**: Viết test tích hợp kết nối với `build_ticknet_l(num_classes, cifar=True)` đảm bảo batch từ loader truyền trơn tru qua forward pass của mạng.

---

### Task 3: Chạy Kiểm thử và Xác nhận Tích hợp

- [x] **Step 3.1**: Chạy `python3 -m pytest tests/test_cifar_data.py -v` và kiểm tra 100% test pass.
- [x] **Step 3.2**: Chạy lại toàn bộ test suite `python3 -m pytest tests -q` đảm bảo không gây regression.
- [x] **Step 3.3**: Commit các thay đổi vào git branch `feature/final-exam-model-l`.
