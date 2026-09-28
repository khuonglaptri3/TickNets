# 7. Model Initialization: Pretrained vs. Scratch

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn (source code proof) thực tế trong đồ án đối với mục **Model Initialization** (khởi tạo mô hình), bao gồm lựa chọn huấn luyện từ đầu (Train from Scratch) hay dùng trọng số có sẵn (Pretrained), thuật toán khởi tạo trọng số Kaiming / Xavier và cơ chế nạp checkpoint.

---

## 1. Bảng tổng hợp Model Initialization trong đồ án

| Tiêu chí | Cơ chế trong đồ án | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- |
| **Chiến lược chính** | **Train from Scratch** (100% từ đầu) | Không dùng trọng số pretrained cho mạng chính; mô hình tùy biến hoàn toàn mới |
| **Backbone Init** | **Kaiming Uniform** (He init) + Zero bias | Phù hợp tối ưu cho hàm kích hoạt ReLU trong các khối PDP (Full-Residual Point-Depth-Point) |
| **Classifier Init** | **Xavier Normal** (Glorot init, gain=1.0) | Đảm bảo phương sai đầu ra ổn định trước khi qua hàm mất mát CrossEntropyLoss |
| **Input Init** | **BatchNorm2d (`data_bn`)** ($\gamma=1, \beta=0$) | Chuẩn hóa tương thích với dải giá trị $[0, 1]$ từ `ToTensor()` |
| **Pretrained ngoại vi** | **MobileNetV3-Small** (ImageNet-1K) | **Chỉ dùng trong module `cleaning/`** làm công cụ kiểm định đối tượng/nhãn độc lập |
| **Checkpoint Reload** | Nạp lại qua cờ `--evaluate <last.pt>` | Khóa chặt tính toàn vẹn bằng SHA-256 của `split_manifest.csv` và `architecture_revision` |

---

## 2. Chi tiết kỹ thuật & Bằng chứng mã nguồn (Code Proof)

### 2.1. Tại sao huấn luyện từ đầu (Train from Scratch)?

Mô hình mục tiêu trong đồ án không dùng pretrained ImageNet từ torchvision zoo mà được huấn luyện từ đầu vì 3 lý do cốt lõi:

1. **Bản chất kiến trúc mạng mới (Novel Architecture)**:
   - Các biến thể **TickNet-Basic**, **TickNet-L**, **TickNet-C** sử dụng cấu trúc khối độc quyền `FR_PDP_block` (Full-Residual Point-Depth-Point block) với cơ chế SE Attention (Squeeze-and-Excitation).
   - Lịch trình phân bổ kênh dạng "tick-shape" (thu hẹp kênh ở stem/stage đầu rồi bung nở mạnh ở các stage sau).
   - Vì là kiến trúc nghiên cứu mới, **không tồn tại trọng số pretrained sẵn có** trên các kho mô hình chuẩn (như ImageNet-1K/22K).
2. **Thích ứng với hai độ phân giải (Dual Resolution Adaptability)**:
   - Mạng được cấu hình linh hoạt cho cả `Mid32` ($32 \times 32$, stride stem = 1) và `Mid224` ($224 \times 224$, stride stem = 2). Việc huấn luyện from scratch giúp trọng số hội tụ tự nhiên theo độ phân giải mục tiêu.
3. **Phân biệt rạch ròi với Module Data Cleaning**:
   - Chỉ có công cụ kiểm định dữ liệu thô `cleaning/object_filter.py` là dùng `MobileNet_V3_Small_Weights.DEFAULT` của torchvision để làm "trọng tài" phát hiện nhãn nghi ngờ. Toàn bộ các mô hình phân loại chính của đồ án (TickNet) đều train from scratch.

---

### 2.2. Thuật toán khởi tạo trọng số (Weight Initialization Algorithms)

Mỗi tầng trong mạng được chỉ định chiến lược khởi tạo toán học riêng biệt:

#### A. Backbone: Kaiming Uniform (He Initialization)
Được triển khai trong hàm `init_params()` tại [models/TickNet.py#L90-L97](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py#L90-L97) và [models/ticknet_l.py#L103-L107](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py#L103-L107):

```python
# [models/TickNet.py: dòng 90-97]
def init_params(self):
    # backbone
    for name, module in self.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
```

* **Cơ sở lý thuyết**:
  - Đối với các lớp tích chập (Conv2D) đi kèm phi tuyến ReLU trong các khối `FR_PDP_block`, phân bố chuẩn Gaussian hay phân bố đều thông thường dễ khiến phương sai kích hoạt bị triệt tiêu (vanishing) hoặc bùng nổ (exploding) khi mạng đi sâu.
  - Phép khởi tạo `kaiming_uniform_` lấy mẫu từ phân bố đều $\mathcal{U}(-\text{bound}, \text{bound})$ với $\text{bound} = \sqrt{\frac{6}{\text{fan\_in}}}$, bù đắp chính xác phần năng lượng bị triệt tiêu một nửa bởi hàm ReLU ($x < 0$).
  - Toàn bộ vector bias của các tầng Conv được gán triệt để về $0$ (`torch.nn.init.constant_(module.bias, 0)`).

#### B. Head phân loại (Classifier): Xavier Normal (Glorot Initialization)
Được triển khai tại [models/common.py#L80-L81](file:///home/intern-tdkhuong/Desktop/TickNets/models/common.py#L80-L81):

```python
# [models/common.py: dòng 66-81]
class Classifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                bias=True)

    def init_params(self):
        torch.nn.init.xavier_normal_(self.conv.weight, gain=1.0)
```

* **Cơ sở lý thuyết**:
  - Tầng Classifier của TickNet sử dụng phép tích chập $1 \times 1$ chuyển $1024$ kênh đặc trưng về đúng $5$ logits phân loại.
  - Để các giá trị logit không bị quá lớn (gây bão hòa hàm softmax và làm gradient ban đầu bị suy giảm), trọng số được khởi tạo theo phân bố chuẩn Xavier:
    $$\mathcal{N}\left(0, \sigma^2\right) \quad \text{với} \quad \sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$$
  - Giữ cho phương sai của tín hiệu đầu ra cân bằng với phương sai đầu vào trước khi tính `CrossEntropyLoss`.

#### C. Tầng chuẩn hóa đầu vào `data_bn`
- Tầng `BatchNorm2d(3)` đặt ngay tại cổng vào của mạng được khởi tạo mặc định theo chuẩn PyTorch:
  - Vector trọng số tỉ lệ: $\gamma = 1.0$.
  - Vector độ dời: $\beta = 0.0$.
  - `running_mean` khởi tạo bằng $0$, `running_var` khởi tạo bằng $1$.

---

### 2.3. Quy trình nạp Checkpoint & Bảo đảm tính toàn vẹn (Checkpoint Reloading)

Khi không huấn luyện mới từ đầu mà cần đánh giá lại mô hình (`python train_mid.py --evaluate <checkpoint.pt>`), pipeline thực hiện kiểm tra chéo nghiêm ngặt tại [train_mid.py#L99-L110](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L99-L110):

```python
if args.evaluate:
    checkpoint = torch.load(args.evaluate, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != mapping or checkpoint["config"]["variant"] != args.variant:
        raise ValueError("Checkpoint class mapping or variant does not match the selected dataset")
    recorded_hash = checkpoint["config"].get("split_manifest_sha256")
    if not recorded_hash or recorded_hash != manifest_hash:
        raise ValueError("Checkpoint split manifest does not match this dataset; use its original prepared split")
    args.model = checkpoint["config"]["model"]
    recorded_revision = checkpoint["config"].get("architecture_revision")
    if args.model in CUSTOM_MODEL_NAMES and recorded_revision != MODEL_REVISIONS[args.model]:
        raise ValueError(f"Checkpoint TickNet-{args.model.upper()} architecture revision does not match this implementation")
```

1. **Khóa khớp định danh lớp**: Kiểm tra `class_to_idx` phải trùng khớp với bộ 5 lớp `CLASSES`.
2. **Khóa khớp nguồn gốc chia (Provenance Protection)**: Kiểm tra mã SHA-256 của `split_manifest.csv` lưu trong checkpoint phải khớp từng ký tự với file manifest hiện hành. Ngăn chặn việc nạp checkpoint được huấn luyện trên một cách chia dữ liệu khác.
3. **Khóa khớp phiên bản kiến trúc (`architecture_revision`)**: Đảm bảo checkpoint nạp vào phải khớp phiên bản thiết kế (`ticknet-basic-upstream`, `ticknet-l-v1`, `ticknet-c-v1`), tránh hiện tượng lệch kích thước tensor trọng số.

---

## 3. Liên kết tham chiếu trong dự án
- [models/TickNet.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py): Triển khai `init_params()` với Kaiming Uniform cho Backbone.
- [models/ticknet_l.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py): Khởi tạo Kaiming Uniform cho TickNet-L v1.
- [models/ticknet_c.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_c.py): Xây dựng TickNet-C kế thừa cơ chế khởi tạo của TickNet gốc.
- [models/common.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/common.py): Triển khai Xavier Normal cho `Classifier`.
- [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py): Pipeline huấn luyện from scratch và nạp checkpoint an toàn.
