# CIFAR-10 & CIFAR-100: Mô Tả Dữ Liệu và Phân Chia Tập Train / Val / Test

Tài liệu này ghi nhận đặc tả cấu trúc dữ liệu, nguồn gốc tải từ Đại học Toronto, mã băm kiểm định MD5, và phương pháp phân chia tập **Train (45.000) / Validation (5.000) / Test (10.000)** cho đồ án cuối kỳ.

---

## 1. Nguồn Dữ liệu & Tính Toàn Vẹn Chuẩn Quốc tế

Hai bộ dữ liệu chuẩn quốc tế được tải tự động qua script [`scripts/download_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/scripts/download_cifar.py):
1. **CIFAR-10 (`cifar-10-python.tar.gz`):**
   - URL chính: `https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz`
   - URL phụ: `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`
   - Mã băm MD5: `c58f30108f718f92721af3b95e74349a`
   - Kích thước ảnh: $32 \times 32$ pixels, 3 kênh màu RGB.
   - 10 lớp: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.
2. **CIFAR-100 (`cifar-100-python.tar.gz`):**
   - URL chính: `https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz`
   - URL phụ: `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
   - Mã băm MD5: `eb9058c3a382ffc7106e4002c42a8d85`
   - Kích thước ảnh: $32 \times 32$ pixels, 3 kênh màu RGB.
   - 100 lớp chi tiết (fine classes), nhóm trong 20 siêu lớp (coarse superclasses).

---

## 2. Phương Pháp Phân Chia Dữ Liệu (Stratified Random Hold-Out)

Quy trình phân chia dữ liệu tuân thủ chuẩn mực học máy thực nghiệm nghiêm ngặt:
- **Tập Test (10.000 ảnh):** Giữ nguyên bản 100% tập Test chính thức của CIFAR do Alex Krizhevsky công bố. Tập này được bảo vệ nguyên vẹn, **tuyệt đối không tham gia vào quá trình huấn luyện hay tinh chỉnh siêu tham số**.
- **Tập Train gốc (50.000 ảnh):** Được phân chia phân tầng (Stratified Split) với tỷ lệ $9 : 1$ bằng hạt giống ngẫu nhiên `seed=42`:
  - **Tập Train thực nghiệm:** $45.000$ ảnh ($90\%$).
  - **Tập Validation:** $5.000$ ảnh ($10\%$).

---

## 3. Bảng Thống kê Phân Bổ Mẫu Chi Tiết

| Bộ Dữ liệu | Số Lớp | Train Sub-set (90%) | Validation Set (10%) | Held-Out Test Set | Tổng số mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | 10 lớp | **45.000 ảnh** (4.500 / lớp) | **5.000 ảnh** (500 / lớp) | **10.000 ảnh** (1.000 / lớp) | 60.000 ảnh |
| **CIFAR-100**| 100 lớp| **45.000 ảnh** (450 / lớp) | **5.000 ảnh** (50 / lớp) | **10.000 ảnh** (100 / lớp) | 60.000 ảnh |

### Vai trò Chức năng của Từng Tập:
1. **Train (45.000 ảnh):** Cập nhật trọng số mạng nơ-ron qua thuật toán tối ưu (SGD + Nesterov hoặc Adam) kèm tăng cường dữ liệu `RandomCrop`, `RandomHorizontalFlip` và `Cutout (16x16)`.
2. **Validation (5.000 ảnh):** Đo lường năng lực học sau mỗi epoch, làm tiêu chuẩn để lưu lại checkpoint tốt nhất `best_val.pt`.
3. **Test (10.000 ảnh):** Hộp đen đánh giá khách quan cuối cùng; được kiểm thử 1 lần duy nhất trên checkpoint `best_val.pt` để báo cáo điểm thi và trả lời vấn đáp Oral Exam.
