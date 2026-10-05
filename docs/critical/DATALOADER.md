# 6. DataLoader: CIFAR-10 & CIFAR-100 (Batch / Shuffle / Workers / Pin_Memory / Cutout)

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn (source code proof) thực tế trong đồ án cuối kỳ đối với mục **DataLoader**, phục vụ huấn luyện mô hình trên hai tập chuẩn quốc tế **CIFAR-10** và **CIFAR-100**.

---

## 1. Bảng tổng hợp cấu hình DataLoader trong Đồ án Cuối kỳ

| Hạng mục | Tập Train | Tập Validation | Tập Test | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- | :--- | :--- |
| **Kích thước mẫu** | **45.000 ảnh** (90% tập train) | **5.000 ảnh** (10% phân tầng) | **10.000 ảnh** (Chuẩn Test) | Tuyệt đối không rò rỉ dữ liệu (No Data Leakage); test giữ nguyên gốc |
| **Batch Size** | 128 (tùy biến qua CLI) | 128 (cùng kích thước) | 128 (cùng kích thước) | Tối ưu hóa throughput bộ nhớ GPU T4/P100 |
| **Drop Last** | `drop_last=False` | `drop_last=False` | `drop_last=False` | Không bỏ sót dữ liệu; batch lẻ được cân trọng số mẫu trong `run_epoch` |
| **Shuffle** | `shuffle=True` (kèm `Generator(seed)`) | `shuffle=False` | `shuffle=False` | Xáo trộn chống overfit khi train; giữ tuần tự ổn định khi eval |
| **Data Augmentation** | **RandomCrop + Flip + Cutout** | Không áp dụng (chỉ Normalize) | Không áp dụng (chỉ Normalize) | Tăng cường dữ liệu chống overfitting, tăng khả năng khái quát hóa |
| **Num Workers** | 2 tiến trình (kèm `seed_worker`) | 2 tiến trình (kèm `seed_worker`) | 2 tiến trình | Tối ưu nạp song song; `seed_worker` đảm bảo tính tái lập 100% |
| **Pin Memory** | Bật tự động khi có CUDA | Bật tự động khi có CUDA | Bật tự động khi có CUDA | Tăng tốc DMA transfer từ RAM máy chủ lên VRAM GPU qua PCIe |

---

## 2. Chi tiết kỹ thuật & Bằng chứng mã nguồn (Code Proof)

Toàn bộ logic tạo DataLoader được đóng gói tập trung trong module [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) và tích hợp vào quy trình huấn luyện tại [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py).

### 2.1. Phân chia Phân tầng (Stratified Validation Split)
Triển khai tại [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) hàm `stratified_split_indices`:
- Tập train 50.000 ảnh của CIFAR được phân chia theo tỷ lệ $9 : 1$ ($45.000$ train và $5.000$ validation).
- Mỗi lớp trong CIFAR-10 có chính xác $4.500$ ảnh train và $500$ ảnh validation.
- Mỗi lớp trong CIFAR-100 có chính xác $450$ ảnh train và $50$ ảnh validation.
- Tập Validation được dùng làm tiêu chí đánh giá chọn `best_val.pt`. Tập Test 10.000 ảnh hoàn toàn độc lập và chỉ được đánh giá 1 lần duy nhất.

### 2.2. Chuỗi Xử lý Ảnh & Tăng cường Dữ liệu Cutout (DeVries & Taylor, 2017)
Triển khai tại [`models/cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/cifar_data.py) hàm `get_cifar_transforms`:
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
Chuỗi biến đổi cho tập **Train**:
1. `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`
2. `transforms.RandomHorizontalFlip(p=0.5)`
3. `transforms.ToTensor()`
4. `transforms.Normalize(mean=mean, std=std)` (Chuẩn hóa chuẩn theo từng bộ dữ liệu)
5. `Cutout(n_holes=1, length=16)` (Che ngẫu nhiên 1 ô $16 \times 16$ pixel)

Chuỗi biến đổi cho tập **Validation & Test**:
1. `transforms.ToTensor()`
2. `transforms.Normalize(mean=mean, std=std)`

### 2.3. Hằng số Chuẩn hóa Toàn vẹn (Standard Normalization Constants)
Mỗi bộ dữ liệu được chuẩn hóa theo giá trị trung bình và độ lệch chuẩn chuẩn mực:
- **CIFAR-10:**
  - Mean: `(0.4914, 0.4822, 0.4465)`
  - Std: `(0.2470, 0.2435, 0.2616)`
- **CIFAR-100:**
  - Mean: `(0.5071, 0.4867, 0.4408)`
  - Std: `(0.2675, 0.2565, 0.2761)`

### 2.4. Tính Toán Metric Có Trọng Số Mẫu (Sample-Weighted Metrics)
Khi `drop_last=False`, batch cuối cùng thường có số lượng mẫu ít hơn `batch_size`. Hàm `run_epoch` trong [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py) tính toán loss và accuracy có trọng số chính xác:
```python
loss_sum += loss.item() * labels.numel()
correct += (preds == labels).sum().item()
total += labels.numel()
...
loss_avg = loss_sum / total
top1_acc = 100.0 * correct / total
```
Tránh hoàn toàn sai lệch thống kê do lấy trung bình số học giữa các batch kích thước không đồng đều.

---

## 3. Bằng chứng Kiểm thử (Verification Evidence)
Hệ thống DataLoader được bảo vệ bởi test suite tự động trong [`tests/test_cifar_data.py`](file:///home/intern-tdkhuong/Desktop/TickNets/tests/test_cifar_data.py):
- `test_normalize_dataset_name`: Chuẩn hóa tên dataset.
- `test_get_cifar_transforms_shape_and_type`: Đảm bảo output luôn là tensor `(3, 32, 32)` float32.
- `test_cutout_transform`: Xác nhận mảng pixel bị che về 0 chính xác.
- `test_stratified_split_indices_proportions_and_disjoint`: Đảm bảo không trùng lặp index giữa train và val.
- `test_cifar_batches_forward_pass_ticknet_l`: Kiểm tra tương thích forward pass với `TickNet-L`.
