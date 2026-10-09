# 4. Dataset Splitting: Cơ Chế Phân Tầng và Phòng Chống Rò Rỉ Dữ Liệu Cuối Kỳ

Tài liệu này tổng hợp phương pháp luận, cơ sở toán học và bằng chứng mã nguồn của thuật toán phân chia phân tầng (Stratified Splitting) trên hai tập dữ liệu **CIFAR-10** và **CIFAR-100**.

---

## 1. Phương pháp Phân tầng theo Lớp (Per-Class Stratification)

### 1.1. Tại sao phải Phân tầng?
Nếu chia 50.000 ảnh train ngẫu nhiên không điều kiện, xác suất xuất hiện độ lệch số lượng mẫu giữa các lớp là rất lớn. Với CIFAR-100 (mỗi lớp chỉ có 500 ảnh trong tập train gốc), sự chênh lệch ngẫu nhiên này có thể làm một số lớp có ít hơn 40 ảnh validation, khiến tín hiệu đánh giá mô hình bị nhiễu động nghiêm trọng.

### 1.2. Thuật toán Triển khai trong Mã nguồn
Triển khai tại hàm `stratified_split_indices` trong [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py):

```python
def stratified_split_indices(
    targets: List[int],
    num_classes: int,
    val_fraction: float,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_to_indices[int(label)].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for c in sorted(class_to_indices):
        indices = class_to_indices[c]
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices
```

### 1.3. Đặc tính Kỹ thuật của Thuật toán:
1. **Cố định Seed Khử Nhiễu Hệ điều hành:** Sử dụng `random.Random(42)`, đảm bảo thứ tự phân tách độc lập hoàn toàn với nền tảng (Linux, Windows, macOS, Kaggle).
2. **Cân Bằng Phân Bố Tuyệt Đối:**
   - Trên CIFAR-10: Mỗi lớp có chính xác 4.500 ảnh Train và 500 ảnh Validation.
   - Trên CIFAR-100: Mỗi lớp có chính xác 450 ảnh Train và 50 ảnh Validation.
3. **Không Giao Thoa (Disjoint Sets):**
   $$\text{Train}_{\text{indices}} \cap \text{Val}_{\text{indices}} = \emptyset$$
   Được chứng thực qua unit test `test_stratified_split_indices_proportions_and_disjoint`.

---

## 2. Nguyên Tắc Vàng: Chống Rò Rỉ Dữ Liệu (Zero Data Leakage)

1. **Cô lập Tuyệt Đối Tập Test:**
   - Tập Test (10.000 ảnh) không hề tham gia vào quá trình tính toán thống kê hay chọn lọc siêu tham số.
   - Hằng số chuẩn hóa Mean và Std được tính toán trên toàn bộ không gian ảnh chuẩn, không bị rò rỉ nhãn phân loại.
2. **Tách Biến đổi Giữa Train và Val:**
   - Lớp `TransformedSubset` trong [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) nhận mảng ảnh gốc (raw numpy arrays) và chỉ áp dụng các phép biến đổi riêng biệt khi một batch được nạp:
     - Tập Train nhận `train_transform` (gồm RandomCrop, Flip, Cutout).
     - Tập Val nhận `eval_transform` (chỉ ToTensor và Normalize).
   - Đảm bảo dữ liệu Validation luôn phản ánh ảnh thực tế không bị che mờ hay cắt xén nhân tạo.
