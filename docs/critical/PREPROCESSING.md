# 5. Preprocessing: Resize / Normalize / Augmentation

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn (source code proof) thực tế trong đồ án đối với mục **Preprocessing** (tiền xử lý dữ liệu), bao gồm xử lý kích thước (Resize), chuẩn hóa (Normalize), tăng cường dữ liệu (Augmentation) và khả năng tái lập (Reproducibility).

---

## 1. Bảng tổng hợp kỹ thuật Preprocessing trong đồ án

| Hạng mục | Tập Train | Tập Test | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- | :--- |
| **Resize** | Không resize ngầm; bắt buộc native size qua `RequireImageSize` | Bắt buộc native size qua `RequireImageSize` | Bảo toàn 100% pixel gốc, chống mờ biên hoặc méo ảnh do nội suy |
| **Normalize** | `ToTensor()` đưa về $[0, 1]$; Chuẩn hóa phân bố qua tầng **`data_bn`** của mạng | `ToTensor()` đưa về $[0, 1]$; Chuẩn hóa qua tham số đóng băng của **`data_bn`** | Tự động học mean/std thích ứng theo từng batch, không dùng ImageNet mean/std tĩnh |
| **Augmentation** | `RandomCrop` (reflect padding, pad = size//8) + `RandomHorizontalFlip(p=0.5)` | **Tuyệt đối không augment** | Tăng độ phong phú dữ liệu train, giữ nguyên vẹn dữ liệu test để đánh giá khách quan |
| **Reproducibility** | Generator seeded (`seed`) + `seed_worker` | Generator seeded (`seed + 1`) | Đảm bảo kết quả huấn luyện có thể tái lập bit-for-bit, hai luồng RNG không can thiệp nhau |

---

## 2. Chi tiết kỹ thuật & Bằng chứng mã nguồn (Code Proof)

Toàn bộ pipeline tiền xử lý và nạp dữ liệu được triển khai chính thức tại [models/mid_data.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py) và được điều phối bởi [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py).

### 2.1. Resize: Kiểm soát kích thước gốc (Native Size Validation)

* **Vấn đề thông thường**: Các pipeline xử lý ảnh truyền thống thường tự động áp dụng `transforms.Resize((size, size))`. Điều này tiềm ẩn rủi ro: nếu nạp sai thư mục hoặc ảnh sai kích thước, ảnh sẽ bị nội suy bilinear/bicubic làm mờ biên, méo cấu trúc hoặc mất chi tiết tần số cao.
* **Giải pháp trong đồ án**: Đồ án xây dựng lớp kiểm soát kích thước tùy biến `RequireImageSize` tại [models/mid_data.py#L35-L44](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L35-L44):
  ```python
  class RequireImageSize:
      """Catch wrong dataset roots instead of silently resizing their images."""
      def __init__(self, size: int):
          self.size = size

      def __call__(self, image):
          if image.size != (self.size, self.size):
              raise ValueError(f"Expected native {self.size}x{self.size} image, got {image.size}")
          return image
  ```
* **Triển khai thực tế**:
  - Đối với `Mid32`: Bắt buộc kích thước ảnh nguyên bản là $32 \times 32$.
  - Đối với `Mid224`: Bắt buộc kích thước ảnh nguyên bản là $224 \times 224$.
  - Nếu dữ liệu không đúng kích thước định dạng, pipeline lập tức quăng ngoại lệ `ValueError` thay vì âm thầm resize, đảm bảo 100% tính toàn vẹn của byte ảnh gốc.

---

### 2.2. Normalize: Tầng `data_bn` trong kiến trúc mạng

* **Chuyển đổi dải giá trị**:
  Sử dụng `transforms.ToTensor()`, đưa giá trị điểm ảnh từ dạng số nguyên $[0, 255]$ về Tensor dạng số thực `float32` trong khoảng $[0.0, 1.0]$.
* **Không dùng `transforms.Normalize(mean, std)` cố định của ImageNet**:
  Thay vì trừ trung bình và chia độ lệch chuẩn tĩnh của tập ImageNet (vốn có thể bị lệch phân bố - domain shift - so với ảnh trong bộ dữ liệu Mid), đồ án tích hợp trực tiếp một tầng **`data_bn`** (`nn.BatchNorm2d(num_features=3)`) tại đầu vào của mạng:
  - Trong **TickNet gốc / Candidate C** ([models/TickNet.py#L65-L67](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py#L65-L67)):
    ```python
    # data batchnorm
    if self.use_data_batchnorm:
        self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))
    ```
  - Trong **TickNet-L v1** ([models/ticknet_l.py#L81-L83](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py#L81-L83)):
    ```python
    nn.Sequential(
        OrderedDict([
            ("data_bn", nn.BatchNorm2d(3)),
            ("init_conv", conv_3x3(3, init_channels, stride=init_stride)),
    ```
* **Ghi vết cấu hình huấn luyện**:
  Trong [train_mid.py#L125](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L125) và tài liệu [docs/DATASET_SPLIT.md#L117](file:///home/intern-tdkhuong/Desktop/TickNets/docs/DATASET_SPLIT.md#L117):
  ```python
  normalization="ToTensor [0,1]; model has data_bn"
  ```
* **Lợi ích**:
  - Tầng `data_bn` học tham số scale ($\gamma$) và shift ($\beta$) tối ưu nhất cho bài toán phân loại 5 lớp.
  - Tự động cập nhật `running_mean` và `running_var` theo đúng phân bố thực tế của dữ liệu huấn luyện.

---

### 2.3. Data Augmentation: Tăng cường có kiểm soát

Được cấu hình trong hàm `build_mid_loaders` tại [models/mid_data.py#L56-L63](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L56-L63):

```python
size = IMAGE_SIZES[variant]
train_ops = [RequireImageSize(size)]
if augment:
    train_ops.extend([
        transforms.RandomCrop(size, padding=size // 8, padding_mode="reflect"),
        transforms.RandomHorizontalFlip()
    ])
train_ops.append(transforms.ToTensor())

root = Path(data_root) / variant
train = datasets.ImageFolder(root / "train", transform=transforms.Compose(train_ops))
test = datasets.ImageFolder(root / "test", transform=transforms.Compose([RequireImageSize(size), transforms.ToTensor()]))
```

1. **`RandomCrop` với tỷ lệ đệm động & chế độ `reflect`**:
   - Tỷ lệ đệm linh hoạt: `padding = size // 8`.
     - Với `Mid32`: padding = 4 $\rightarrow$ Pad thành $40 \times 40$ rồi crop ngẫu nhiên về $32 \times 32$.
     - Với `Mid224`: padding = 28 $\rightarrow$ Pad thành $280 \times 280$ rồi crop ngẫu nhiên về $224 \times 224$.
   - Chế độ `reflect` (đệm phản chiếu qua biên): Tránh tạo viền đen nhân tạo (zero-padding), bảo toàn tính liên tục của hoa văn và lông thú ở vùng rìa.
2. **`RandomHorizontalFlip(p=0.5)`**:
   - Lật ảnh theo trục ngang với xác suất 50%. Phù hợp ngữ nghĩa với 5 lớp sinh vật (`bird`, `cat`, `dog`, `frog`, `horse`), giúp mô hình bất biến với góc nhìn trái/phải.
3. **Tập Test không Augment**:
   - Giữ nguyên bản để làm thước đo đánh giá hiệu năng khách quan (Held-out Test evaluation).
4. **Hỗ trợ Ablation Study**: Có cờ `--no-augment` trong `train_mid.py` cho phép tắt toàn bộ augmentation khi cần so sánh độ ảnh hưởng của data augmentation tới độ chính xác.

---

### 2.4. Reproducibility: Độc lập luồng ngẫu nhiên (RNG Isolation)

Để ngăn chặn hiện tượng mất tính tái lập khi chạy đa luồng hoặc ngẫu nhiên hóa dữ liệu:

1. **Khởi tạo Seed toàn diện (`seed_everything`)** ([models/mid_data.py#L17-L26](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L17-L26)):
   ```python
   def seed_everything(seed: int) -> None:
       os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
       random.seed(seed)
       np.random.seed(seed % (2**32))
       torch.manual_seed(seed)
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.benchmark = False
       torch.backends.cudnn.deterministic = True
       torch.use_deterministic_algorithms(True)
   ```
2. **Khởi tạo Worker độc lập (`seed_worker`)** ([models/mid_data.py#L29-L32](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L29-L32)):
   ```python
   def seed_worker(worker_id: int) -> None:
       worker_seed = torch.initial_seed() % (2**32)
       random.seed(worker_seed)
       np.random.seed(worker_seed)
   ```
3. **Cách ly Generator giữa Train và Test** ([models/mid_data.py#L71-L75](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py#L71-L75)):
   ```python
   train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
   # A separate generator prevents test iteration from changing the train RNG stream.
   test_loader = DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(seed + 1), **common)
   ```
   * Bộ sinh ngẫu nhiên của Test dùng `seed + 1`, đảm bảo việc duyệt dữ liệu kiểm thử ở các epoch đánh giá không bao giờ làm thay đổi chuỗi số ngẫu nhiên của tập Train.

---

## 3. Liên kết tham chiếu trong dự án
- [models/mid_data.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/mid_data.py): Triển khai các transform, `RequireImageSize`, DataLoader seeded.
- [models/TickNet.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py): Kiến trúc backbone TickNet tích hợp tầng `data_bn`.
- [models/ticknet_l.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py): Kiến trúc TickNet-L v1 tích hợp tầng `data_bn`.
- [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py): Pipeline huấn luyện chính thức ghi vết cấu hình chuẩn hóa.
- [docs/DATASET_SPLIT.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/DATASET_SPLIT.md): Tài liệu hóa phương thức chuẩn hóa và tiền xử lý.
