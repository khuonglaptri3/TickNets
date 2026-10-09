# 7. Model Initialization: Khởi Tạo Trọng Số và Chiến Lược Train from Scratch Cuối Kỳ

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn thực tế đối với mục **Model Initialization** (khởi tạo mô hình), khẳng định cam kết **Train from Scratch 100% trên CIFAR-10 & CIFAR-100** và phân tích thuật toán khởi tạo Kaiming / Xavier.

---

## 1. Bảng Tổng Hợp Khởi Tạo Mô Hình Cuối Kỳ

| Tiêu chí | Cơ chế trong đồ án | Căn cứ thiết kế & Triển khai trong code |
| :--- | :--- | :--- |
| **Chiến lược chính** | **Train from Scratch 100%** | Tuyệt đối **không dùng checkpoint giữa kỳ hay pretrain ImageNet**; khởi tạo trọng số mới hoàn toàn |
| **Backbone Conv Init**| **Kaiming Uniform (He init)** + Zero bias | Phù hợp tối ưu cho hàm kích hoạt ReLU trong các khối FR-PDP và Stem Conv |
| **Classifier Head Init**| **Xavier Normal cho weight; bias mặc định của Conv2d** | Khởi tạo tầng phân loại 10 lớp (CIFAR-10) và 100 lớp (CIFAR-100) |
| **Batch Normalization** | Weight $\gamma=1.0$, Bias $\beta=0.0$ | Bảo toàn thang đo phương sai ban đầu trước khi cập nhật running mean/var |
| **Squeeze-and-Excitation**| Khởi tạo mặc định nn.Linear (Kaiming Uniform với a=sqrt(5), bias Uniform) | Không có bước khởi tạo lại riêng cho SE Linear |
| **Hạt giống ngẫu nhiên** | `seed_everything(args.seed=42)` | Cố định toàn bộ seed cho PyTorch (CPU/CUDA), NumPy và Python random |

---

## 2. Tại sao nhóm chọn Train from Scratch trên CIFAR?

Đây là lựa chọn của nhóm để so sánh các cấu hình từ cùng trạng thái khởi tạo.
Đề DOCX không ghi lệnh cấm pretrained và cho phép kế thừa kiến trúc nhóm đã đề xuất ở giữa kỳ.

1. **Tính nhất quán của thực nghiệm:**
   - Đề thi cuối kỳ yêu cầu đánh giá năng lực hội tụ của mô hình `TickNet-L` với các siêu tham số khác nhau (SGD 0.1, 0.15; Adam 0.001, 0.0003). Nếu dùng checkpoint nạp sẵn từ nhiệm vụ khác, việc so sánh tốc độ học và đường cong hội tụ sẽ hoàn toàn mất giá trị.
2. **Sự Khác Biệt về Cấu Trúc Head Phân Loại:**
   - Giữa kỳ mô hình chỉ phân loại 5 lớp đối tượng.
   - Cuối kỳ mô hình phải phân loại **10 lớp (CIFAR-10)** và **100 lớp (CIFAR-100)**.
   - Việc khởi tạo from scratch cho phép toàn bộ mạng từ các tầng trích xuất cạnh/màu sắc (low-level features) đến các tầng ngữ nghĩa trừu tượng (high-level features) thích ứng tối ưu với miền dữ liệu CIFAR.

---

## 3. Thuật Toán Khởi Tạo Chi Tiết & Bằng Chứng Mã Nguồn

Triển khai tại [`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py) và [`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py):

### 3.1. Tầng Tích Chập (Conv2d): Kaiming Uniform (He Initialization)
```python
# models/ticknet_l.py: chỉ backbone Conv2d được khởi tạo lại ở vòng lặp này.
for module in self.backbone.modules():
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
self.classifier.init_params()  # common.py: xavier_normal_(classifier.conv.weight)
# BN và SE Linear giữ khởi tạo mặc định của constructor PyTorch.
```

- **Cơ sở Lý thuyết:** Với hàm kích hoạt phi tuyến ReLU ($f(x) = \max(0, x)$), một nửa số nơ-ron bị triệt tiêu về 0 khi $x < 0$. Nếu khởi tạo phân bố chuẩn thông thường với phương sai $\frac{1}{\text{fan\_in}}$, phương sai đầu ra sau mỗi tầng sẽ giảm đi một nửa, dẫn đến hiện tượng gradient bị triệt tiêu khi mạng sâu.
- Phép khởi tạo `kaiming_uniform_` lấy mẫu từ khoảng $[-\text{bound}, \text{bound}]$ với $\text{bound} = \sqrt{\frac{6}{\text{fan\_in}}}$, nhân đôi phương sai để bù đắp chính xác phần năng lượng bị triệt tiêu.

### 3.2. Tính Toàn Vẹn Không Có Tham Số Bị Ngắt Kết Nối
Bộ kiểm thử trong [`tests/test_ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/tests/test_ticknet_l.py) hàm `test_l_trains_at_both_native_resolutions_with_no_disconnected_parameters` chạy 1 bước forward + backward và kiểm tra:
```python
for name, param in model.named_parameters():
    assert param.grad is not None, f"Parameter {name} has no gradient!"
    assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"
```
Đảm bảo 100% tham số trong mạng đều tham gia vào đồ thị tính toán và nhận gradient cập nhật đầy đủ ngay từ epoch đầu tiên.
