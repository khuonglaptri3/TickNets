# Đặc Tả Kỹ Thuật Hệ Thống Toàn Diện Kỳ Thi Cuối Kỳ: TickNet-L & Author Baseline

**Dự án:** Deep Learning Final Examination  
**Ngày cập nhật:** 2026-10-05  
**Nhánh Git:** `feature/final-exam-model-l`  
**Mục tiêu:** Đặc tả toàn bộ các thay đổi kiến trúc, kỹ thuật tiền xử lý, tối ưu hóa, module hóa Kaggle, tài liệu kỹ thuật và trích dẫn học thuật cho hệ thống huấn luyện và đánh giá trên CIFAR-10 & CIFAR-100.

---

## 1. Bối Cảnh & Ràng Buộc Đồ Án Cuối Kỳ

Theo yêu cầu chính thức từ đề thi cuối kỳ:
1. **Huấn luyện từ đầu (Train from Scratch 100%):** Không sử dụng bất kỳ trọng số pretrained nào (kể cả trọng số giữa kỳ). Khởi tạo trọng số tuân theo chuẩn Kaiming Normal/Uniform.
2. **Ràng buộc ngân sách phần cứng:**
   - **Số lượng tham số (Parameters):** $\le 6,000,000$ (6M).
   - **Chi phí tính toán (FLOPs forward):** $< 1,000,000,000$ (1G FLOPs cho ảnh đầu vào kích thước $32 \times 32 \times 3$). Quy ước tính toán: $1 \text{ MAC} = 2 \text{ FLOPs}$.
3. **Bộ dữ liệu mục tiêu:** Đánh giá đồng thời trên hai tập dữ liệu benchmark chuẩn:
   - **CIFAR-10:** 10 lớp, 60.000 ảnh ($32 \times 32$).
   - **CIFAR-100:** 100 lớp, 60.000 ảnh ($32 \times 32$).
4. **Phân chia dữ liệu không rò rỉ (Zero Data Leakage):**
   - Tập huấn luyện (Train Set): 45.000 ảnh (90% tập official train, phân tầng stratified).
   - Tập kiểm định (Validation Set): 5.000 ảnh (10% tập official train, phân tầng stratified). Dùng để điều phối scheduler và chọn checkpoint `best_val.pt`.
   - Tập kiểm thử chính thức (Official Test Set): 10.000 ảnh. Được đánh giá duy nhất một lần ở cuối quá trình trên checkpoint `best_val.pt`.

---

## 2. Đặc Tả Nâng Cấp Pipeline Huấn Luyện & Tiền Xử Lý

### 2.1. Tăng Cường Dữ Liệu: Tích Hợp Kỹ Thuật Cutout (DeVries & Taylor, 2017)
- **Tập tin hiện thực:** [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py).
- **Mô tả toán học:** Lớp biến đổi `Cutout(n_holes=1, length=16)` tạo một mặt nạ hình vuông kích thước $16 \times 16$ tại tọa độ ngẫu nhiên $(y, x)$ trên ảnh $32 \times 32$, gán toàn bộ các giá trị trong vùng này về 0.0 (hoặc giá trị trung bình sau chuẩn hóa).
  $$M_{i,j} = \begin{cases} 0 & \text{nếu } |i - y| \le 8 \text{ và } |j - x| \le 8 \\ 1 & \text{ngược lại} \end{cases}$$
- **Quy trình Transform huấn luyện đầy đủ:**
  1. `RandomCrop(32, padding=4, padding_mode='reflect')`
  2. `RandomHorizontalFlip(p=0.5)`
  3. `ToTensor()` (chuyển đổi miền pixel về $[0.0, 1.0]$)
  4. `Normalize(mean, std)` (CIFAR-10: `mean=[0.4914, 0.4822, 0.4465]`, `std=[0.2470, 0.2435, 0.2616]`; CIFAR-100: `mean=[0.5071, 0.4865, 0.4409]`, `std=[0.2673, 0.2564, 0.2762]`)
  5. `Cutout(n_holes=1, length=16)`

### 2.2. Nâng Cấp Bộ Tối Ưu Hóa: Kích Hoạt Nesterov Momentum Cho SGD
- **Tập tin hiện thực:** [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py).
- **Đặc tả thuật toán:** Khi bộ tối ưu hóa là `sgd` với hệ số động lượng `momentum > 0`, cờ `nesterov=True` được kích hoạt mặc định.
  $$v_{t} = \mu v_{t-1} + g_t$$
  $$\theta_t = \theta_{t-1} - \eta (g_t + \mu v_t)$$
  Trong đó $\mu = 0.9$, $\eta$ là tốc độ học (0.10 hoặc 0.15), và suy giảm trọng số $\lambda = 1 \times 10^{-4}$.
- **Cơ chế Adam:** Sử dụng $\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\lambda = 1 \times 10^{-4}$ với $\eta \in \{0.001, 0.0003\}$.
- **Lịch trình LR:** Cosine Annealing Decay về 0 trong suốt 200 epochs:
  $$\eta_t = \frac{1}{2} \eta_{0} \left(1 + \cos\left(\frac{t \pi}{T_{\max}}\right)\right)$$

---

## 3. Đặc Tả Ma Trận Cấu Hình Thực Nghiệm (10 Cấu Hình)

Hệ thống cấu hình trong thư mục [`configs/final/`](file:///home/intern-tdkhuong/Desktop/TickNets/configs/final) bao gồm 2 nhóm:

### 3.1. Nhóm Baseline Của Tác Giả (Author TickNet-Basic Baseline)
1. `baseline_cifar10_sgd_lr010.json`: Mô hình `basic`, CIFAR-10, SGD (lr=0.10, Nesterov=0.9, Cutout=16, 200 epochs).
2. `baseline_cifar100_sgd_lr010.json`: Mô hình `basic`, CIFAR-100, SGD (lr=0.10, Nesterov=0.9, Cutout=16, 200 epochs).

### 3.2. Nhóm Grid Search Mô Hình Đề Xuất (TickNet-L v1)
1. `cifar10_sgd_lr010.json`: Mô hình `l`, CIFAR-10, SGD (lr=0.10, Nesterov=0.9).
2. `cifar10_sgd_lr015.json`: Mô hình `l`, CIFAR-10, SGD (lr=0.15, Nesterov=0.9).
3. `cifar10_adam_lr0001.json`: Mô hình `l`, CIFAR-10, Adam (lr=0.001).
4. `cifar10_adam_lr00003.json`: Mô hình `l`, CIFAR-10, Adam (lr=0.0003).
5. `cifar100_sgd_lr010.json`: Mô hình `l`, CIFAR-100, SGD (lr=0.10, Nesterov=0.9).
6. `cifar100_sgd_lr015.json`: Mô hình `l`, CIFAR-100, SGD (lr=0.15, Nesterov=0.9).
7. `cifar100_adam_lr0001.json`: Mô hình `l`, CIFAR-100, Adam (lr=0.001).
8. `cifar100_adam_lr00003.json`: Mô hình `l`, CIFAR-100, Adam (lr=0.0003).

---

## 4. Đặc Tả Phân Tách Notebook Kaggle 4 Phase & Baseline

Nhằm khắc phục tình trạng GPU timeout (giới hạn 12 giờ chạy của Kaggle) và tránh rủi ro mất mát toàn bộ tiến trình khi chạy 8 thử nghiệm trên cùng một notebook, kiến trúc Kaggle được module hóa thành 5 notebook độc lập:

1. [`docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb):
   - Chạy mô hình nguyên bản của tác giả (`TickNet-Basic`) trên cả CIFAR-10 và CIFAR-100 (SGD lr=0.10).
   - Tự động đóng gói kết quả vào `author_ticknet_baseline_results.zip`.
2. [`docs/kaggle/Phase1_CIFAR10_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase1_CIFAR10_SGD.ipynb):
   - Chạy `cifar10_sgd_lr010` và `cifar10_sgd_lr015`.
   - Xuất file nén `phase1_cifar10_sgd_results.zip`.
3. [`docs/kaggle/Phase2_CIFAR10_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase2_CIFAR10_Adam.ipynb):
   - Chạy `cifar10_adam_lr0001` và `cifar10_adam_lr00003`.
   - Xuất file nén `phase2_cifar10_adam_results.zip`.
4. [`docs/kaggle/Phase3_CIFAR100_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase3_CIFAR100_SGD.ipynb):
   - Chạy `cifar100_sgd_lr010` và `cifar100_sgd_lr015`.
   - Xuất file nén `phase3_cifar100_sgd_results.zip`.
5. [`docs/kaggle/Phase4_CIFAR100_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase4_CIFAR100_Adam.ipynb):
   - Chạy `cifar100_adam_lr0001` và `cifar100_adam_lr00003`.
   - Xuất file nén `phase4_cifar100_adam_results.zip`.

---

## 5. Đặc Tả Tái Cấu Trúc Bộ Tài Liệu Kỹ Thuật (`docs/critical/`)

Đã tiến hành đồng bộ và viết lại toàn bộ 11 tài liệu kỹ thuật cốt lõi:
1. `COMMANDS_GUIDE.md`: Sổ tay lệnh thực thi cho train, eval, resume, kiểm tra tài nguyên và Kaggle.
2. `MODEL_L.md`: Thiết kế kiến trúc TickNet-L v1, chứng minh công thức FLOPs và tham số.
3. `DATALOADER.md`: Cơ chế nạp dữ liệu CIFAR, phân tầng stratified 90/10, worker isolation và non-blocking transfers.
4. `PREPROCESSING.md`: Đặc tả chi tiết kỹ thuật Cutout 16x16 theo DeVries & Taylor (2017) và ma trận chuẩn hóa màu.
5. `HYPERPARAMETER_TUNING.md`: Lý luận hội tụ giữa SGD Nesterov và Adam, phân tích bề mặt tối ưu hóa (loss landscape).
6. `MODEL_INITIALIZATION.md`: Khởi tạo Kaiming Normal/Uniform, chứng minh việc huấn luyện from scratch không phụ thuộc pretrained.
7. `TRAINING_AND_ARCHITECTURE.md`: Bảng đối chiếu chi tiết giữa kiến trúc Basic của tác giả và TickNet-L v1.
8. `VALIDATION_AND_METRICS.md`: Công thức toán học Top-1 accuracy, Macro-F1, Cross-Entropy Loss và ma trận nhầm lẫn.
9. `DATASET_SPLIT.md`: Bảng kê phân bổ mẫu chi tiết từng lớp cho CIFAR-10 và CIFAR-100.
10. `DATASET_SPLITTING.md`: Cam kết kiểm chứng không rò rỉ dữ liệu giữa train/val/test.
11. `DATASET_CLEANING.md`: Quy trình xác thực tính toàn vẹn nhị phân MD5 và tính chuẩn hóa của dữ liệu CIFAR.

---

## 6. Đặc Tả Bảo Toàn Trích Dẫn Học Thuật & Vinh Danh Tác Giả Gốc

Tài liệu gốc `README.md` bắt buộc phải thể hiện sự tôn trọng quyền tác giả và nguồn gốc học thuật của công trình TickNets:
- **Tác giả:** Thanh Tuan Nguyen, Thanh Phuong Nguyen.
- **Tên bài báo:** *Efficient tick-shape networks of full-residual point-depth-point blocks for image classification*.
- **Tạp chí xuất bản:** *Neurocomputing*, Volume 596, Năm 2024, Mã bài: 127942.
- **DOI:** `10.1016/j.neucom.2024.127942`.
- **Mã BibTeX:** `@article{neucoTickNetNguyen23, ...}`.
- **Abstract nguyên văn:** Được lưu trữ đầy đủ trong Mục 8 của `README.md` nhằm phục vụ trích dẫn trong Báo cáo cuối kỳ.
