# 5. Preprocessing: CIFAR-10 & CIFAR-100 (Resize / Normalize / Data Augmentation & Cutout)

Tài liệu này ghi nhận chi tiết thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn thực tế trong đồ án cuối kỳ đối với mục **Preprocessing** (tiền xử lý dữ liệu) và **Data Augmentation** cho hai tập dữ liệu **CIFAR-10** và **CIFAR-100**.

---

## 1. Bảng Tổng Hợp Kỹ Thuật Tiền Xử Lý Dữ Liệu Cuối Kỳ

| Hạng mục | Tập Train | Tập Validation & Test | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- | :--- |
| **Kích thước ảnh** | Nguyên bản $32 \times 32$ | Nguyên bản $32 \times 32$ | Dữ liệu gốc CIFAR là $32 \times 32$, bảo toàn 100% pixel gốc, không nội suy phóng đại |
| **Data Augmentation 1** | **RandomCrop(32, pad=4, reflect)** | Không áp dụng | Đệm phản xạ 4 pixel xung quanh rồi cắt ngẫu nhiên $32 \times 32$, tạo tính bất biến dịch chuyển |
| **Data Augmentation 2** | **RandomHorizontalFlip(p=0.5)** | Không áp dụng | Lật ngang ngẫu nhiên $50\%$ số ảnh, tăng gấp đôi tính đa dạng đối xứng |
| **Data Augmentation 3** | **Cutout(1 hole, length=16)** | Không áp dụng | Che ngẫu nhiên mảng $16 \times 16$ pixel (DeVries & Taylor, 2017), ép mô hình học đặc trưng toàn cục |
| **Chuyển đổi Tensor** | `transforms.ToTensor()` | `transforms.ToTensor()` | Chuyển mảng điểm ảnh $[0, 255]$ sang tensor float32 $[0.0, 1.0]$ |
| **Chuẩn hóa (Normalize)** | `transforms.Normalize(mean, std)` | `transforms.Normalize(mean, std)` | Chuẩn hóa theo hằng số phân bố chuẩn xác của từng tập dữ liệu (CIFAR-10 vs CIFAR-100) |
| **Tính Tái Lập** | Generator cố định + `seed_worker` | Độc lập, cố định | Đảm bảo kết quả huấn luyện có thể tái lập bit-for-bit |

---

## 2. Chi tiết Kỹ thuật & Bằng chứng Mã nguồn ([`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py))

### 2.1. Hằng số Chuẩn hóa Chuẩn Quốc tế
Mỗi tập dữ liệu sở hữu đặc trưng màu sắc và ánh sáng riêng biệt:
```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
```
Chuẩn hóa giúp đưa phân bố dữ liệu về kỳ vọng $\approx 0$ và phương sai $\approx 1$, giúp gradient truyền ngược ở các lớp đầu không bị tiêu biến (vanishing) hay phát nổ (exploding).

### 2.2. Kỹ thuật Điều hòa Cutout (DeVries & Taylor, 2017)
Được triển khai trong class `Cutout` tại [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py):
```python
class Cutout(object):
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask_tensor = torch.from_numpy(mask).to(dtype=img.dtype, device=img.device).expand_as(img)
        return img * mask_tensor
```
* **Nguyên lý hoạt động:** Trong mỗi ảnh huấn luyện sau khi chuẩn hóa, chọn ngẫu nhiên một tâm $(x, y)$ và che một vùng vuông $16 \times 16$ pixel thành giá trị 0 (tương ứng với giá trị trung bình sau normalize).
* **Hiệu quả thực nghiệm:** Ngăn chặn hiện tượng mô hình phụ thuộc vào một chi tiết nhỏ mang tính "học vẹt" (ví dụ: chỉ nhìn thấy mỏ chim là đoán chim, nếu che mỏ thì mô hình buộc phải học thêm hình thái cánh và lông).

### 2.3. Pipeline Biến đổi Hoàn chỉnh (`get_cifar_transforms`)
```python
def get_cifar_transforms(dataset_name: str, *, augment: bool = True, cutout: bool = True, cutout_length: int = 16) -> transforms.Compose:
    canon_name = normalize_dataset_name(dataset_name)
    mean, std = CIFAR_STATS[canon_name]

    if augment:
        tf_list = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if cutout and cutout_length > 0:
            tf_list.append(Cutout(n_holes=1, length=cutout_length))
        return transforms.Compose(tf_list)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
```

---

## 3. Bằng chứng Kiểm thử (Verification Evidence)
Quy trình tiền xử lý được kiểm định bằng unit test:
1. `tests/test_cifar_data.py::test_get_cifar_transforms_shape_and_type`: Xác nhận ảnh sau tiền xử lý luôn có kích thước `(3, 32, 32)`, kiểu dữ liệu `torch.float32`, và giá trị chuẩn hóa vượt ra ngoài khoảng $[0, 1]$.
2. `tests/test_cifar_data.py::test_cutout_transform`: Xác nhận vùng pixel $16 \times 16$ bị che triệt để về 0.0 trong khi phần còn lại của ảnh được giữ nguyên vẹn.
