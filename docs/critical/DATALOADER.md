# 6. DataLoader: Batch / Shuffle / Workers / Pin_Memory

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn (source code proof) thực tế trong đồ án đối với mục **DataLoader**, bao gồm kích thước Batch (`batch_size`, `drop_last`), cơ chế xáo trộn (`shuffle`, generator isolation), đa tiến trình nạp dữ liệu (`num_workers`, `seed_worker`) và tối ưu hóa bộ nhớ GPU (`pin_memory`).

---

## 1. Bảng tổng hợp cấu hình DataLoader trong đồ án

| Hạng mục | Tập Train | Tập Test | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- | :--- |
| **Batch Size** | Mặc định 64 (tùy biến qua `--batch-size`) | 64 (cùng kích thước với train) | Kiểm soát tài nguyên VRAM, phù hợp cả CPU lẫn GPU |
| **Drop Last** | `drop_last=False` (giữ trọn 25.000 mẫu) | `drop_last=False` (giữ trọn 250 mẫu) | Không bỏ sót dữ liệu; batch lẻ được cân trọng số mẫu trong `run_epoch` |
| **Shuffle** | `shuffle=True` (kèm `Generator(seed)`) | `shuffle=False` (kèm `Generator(seed + 1)`) | Xáo trộn chống overfit khi train; giữ tuần tự ổn định khi test |
| **Num Workers** | Tùy biến qua `--num-workers` (kèm `seed_worker`) | Tùy biến qua `--num-workers` (kèm `seed_worker`) | Tối ưu nạp song song; `seed_worker` đảm bảo tính tái lập 100% |
| **Pin Memory** | Tự động kích hoạt khi có CUDA (`device.type == "cuda"`) | Tự động kích hoạt khi có CUDA (`device.type == "cuda"`) | Tăng tốc DMA transfer từ RAM lên GPU qua PCIe |
| **Verification** | Chế độ `--check-data` kiểm tra batch thô | Chế độ `--check-data` kiểm tra batch thô | Xác thực shape `(B, 3, H, W)` và class mapping trước khi train |

---

## 2. Chi tiết kỹ thuật & Bằng chứng mã nguồn (Code Proof)

Toàn bộ logic tạo DataLoader được đóng gói trong hàm `build_mid_loaders` tại [models/mid_data.py#L46-L77](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L46-L77) và tích hợp vào quy trình huấn luyện tại [train_mid.py#L76-L90](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L76-L90).

### 2.1. Batch: Xử lý Batch không chẵn & Cân trọng số chính xác

* **Khởi tạo tham số**: Tham số `--batch-size` được định nghĩa trong [train_mid.py#L50](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L50) (mặc định = 64), bắt buộc `batch_size >= 1`.
* **Giữ trọn mẫu với `drop_last=False`** ([models/mid_data.py#L71-L72](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L71-L72)):
  ```python
  common = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker,
                pin_memory=pin_memory, drop_last=False)
  ```
  * Đối với **Train** (25.000 ảnh): Gồm $390$ batch đủ 64 mẫu và $1$ batch cuối có $40$ mẫu ($390 \times 64 + 40 = 25.000$).
  * Đối với **Test** (250 ảnh): Gồm $3$ batch đủ 64 mẫu và $1$ batch cuối có $58$ mẫu ($3 \times 64 + 58 = 250$).
* **Thuật toán tính metric theo trọng số mẫu (Sample-Weighted Metrics)**:
  Khi `drop_last=False`, nếu chỉ cộng trung bình loss giữa các batch thì batch cuối (40 mẫu) sẽ bị tính ngang quyền với batch 64 mẫu, gây sai số thống kê.
  Hàm `run_epoch` trong [train_mid.py#L35-L41](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L35-L41) giải quyết triệt để điều này bằng cách nhân trọng số số lượng mẫu thực tế:
  ```python
  batch_count = labels.numel()
  loss_sum += loss.item() * batch_count
  correct += (logits.argmax(dim=1) == labels).sum().item()
  count += batch_count
  ...
  return {"loss": loss_sum / count, "top1": 100.0 * correct / count, "samples": count}
  ```
  *(Được kiểm chứng qua unit test: `test_epoch_metrics_weight_partial_batches_by_sample_count` trong `tests/test_mid_pipeline.py`).*

---

### 2.2. Shuffle: Độc lập luồng sinh số ngẫu nhiên (RNG Stream Isolation)

Triển khai tại [models/mid_data.py#L73-L75](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L73-L75):
```python
train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)

# A separate generator prevents test iteration from changing the train RNG stream.
test_loader = DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **common)
```
* **Tập Train (`shuffle=True`)**:
  - Dùng `torch.Generator().manual_seed(seed)` riêng biệt.
  - Xáo trộn ngẫu nhiên thứ tự ảnh trước mỗi epoch để phá vỡ mối tương quan thứ tự, chống overfitting.
* **Tập Test (`shuffle=False`)**:
  - Dùng `torch.Generator().manual_seed(seed + 1)` riêng biệt.
  - Giữ thứ tự mẫu duyệt qua tuần tự (`range(len(test))`), giúp việc theo dõi lỗi phân loại và tính confusion matrix giữa các lần đánh giá luôn cố định và nhất quán.
* **Nguyên tắc cách ly RNG**: Việc test loader có bộ generator riêng ngăn việc gọi `iter(test_loader)` làm lệch luồng số ngẫu nhiên của `train_loader`.

---

### 2.3. Workers: Đa tiến trình song song & Bảo toàn tính tái lập (`seed_worker`)

* **Tham số điều khiển**:
  - `--num-workers`: Số tiến trình con đọc dữ liệu song song (mặc định = 0).
  - `--threads`: Số luồng CPU tính toán nội bộ (PyTorch intra-op threads), thiết lập qua `torch.set_num_threads(args.threads)` ([train_mid.py#L76](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L76)).
* **Cơ chế `seed_worker` chống trùng lặp ngẫu nhiên** ([models/mid_data.py#L29-L32](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L29-L32)):
  ```python
  def seed_worker(worker_id: int) -> None:
      worker_seed = torch.initial_seed() % (2**32)
      random.seed(worker_seed)
      np.random.seed(worker_seed)
  ```
  Khi nạp dữ liệu bằng nhiều tiến trình con (`num_workers > 0`), hàm `seed_worker` tự động gán seed độc lập cho thư viện `random` và `numpy` của từng worker dựa trên `initial_seed` của PyTorch. Nhờ đó, các phép data augmentation (`RandomCrop`, `RandomHorizontalFlip`) được thực thi trên nhiều worker vẫn **hoàn toàn tái lập bit-for-bit**.

---

### 2.4. Pin Memory: Tự động tối ưu hóa bộ nhớ GPU (Pinned Memory / DMA)

* **Thiết lập tự động thích ứng** tại [train_mid.py#L81](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L81):
  ```python
  train_loader, test_loader = build_mid_loaders(
      args.data_root, args.variant, batch_size=args.batch_size,
      seed=args.seed, num_workers=args.num_workers,
      augment=not args.no_augment, pin_memory=device.type == "cuda"
  )
  ```
* **Bản chất kỹ thuật**:
  - Khi phát hiện thiết bị là GPU (`device.type == "cuda"`), cờ `pin_memory=True` tự động được bật.
  - PyTorch sẽ cấp phát bộ nhớ trang cố định (page-locked / pinned memory) trên RAM của máy chủ.
  - Dữ liệu ảnh sau đó được sao chép trực tiếp sang bộ nhớ VRAM của GPU thông qua bus PCIe bằng cơ chế **DMA (Direct Memory Access)**, bỏ qua sự can thiệp của CPU, giúp GPU không bị nghẽn (bottleneck) khi chờ nạp batch.
  - Khi chạy trên CPU (`device.type == "cpu"`), `pin_memory=False` để giải phóng tài nguyên RAM.

---

### 2.5. Chế độ kiểm tra nhanh dữ liệu (`--check-data`)

Tại [train_mid.py#L83-L90](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L83-L90), hệ thống cung cấp chế độ kiểm tra dữ liệu độc lập không cần train:
```bash
python train_mid.py --data-root data --variant Mid224 --seed 42 --check-data
```
Lệnh này nạp thử ngay 1 batch đầu tiên từ cả `train_loader` và `test_loader`, kiểm tra và in ra thông số:
- Số lượng mẫu của train (25.000) và test (250).
- Shape tensor: `[64, 3, 224, 224]`.
- Bảng ánh xạ nhãn (`class_to_idx`).
- Hạt giống ngẫu nhiên (`seed`).

---

## 3. Liên kết tham chiếu trong dự án
- [models/mid_data.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py): Định nghĩa `build_mid_loaders`, `seed_worker`, cấu hình `DataLoader`.
- [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py): Điều phối tham số batch, device, `pin_memory` và vòng lặp `run_epoch`.
- [tests/test_mid_pipeline.py](file:///home/intern-tdkhuong/Desktop/TickNets/tests/test_mid_pipeline.py): Bộ unit test tự động xác thực tính tái lập của shuffle, native shape và tính chính xác của sample-weighted metric.
