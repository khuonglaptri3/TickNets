# 8. Training Pipeline & So sánh Kiến trúc với TickNet gốc

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn (source code proof) thực tế trong đồ án đối với:
1. **Quy trình Huấn luyện (Training Pipeline)**: Vòng lặp `forward` $\rightarrow$ `loss` $\rightarrow$ `backward` $\rightarrow$ `optimizer.step()`, kiểm soát số học và bộ lập lịch Cosine.
2. **So sánh Kiến trúc với TickNet ban đầu**: Phân tích sự khác biệt giữa bản gốc `models/TickNet.py` (TickNet-Basic) và hai ứng viên nghiên cứu **TickNet-L v1** và **TickNet-C v1**.

---

## 1. Quy trình Huấn luyện (Training Loop Mechanics)

Toàn bộ vòng lặp huấn luyện và đánh giá được chuẩn hóa tại hàm `run_epoch` trong [train_mid.py#L19-L42](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L19-L42):

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; training/evaluation aborted")
            if training:
                loss.backward()
                optimizer.step()
            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_count
    if not count:
        raise ValueError("Cannot evaluate an empty data loader")
    return {"loss": loss_sum / count, "top1": 100.0 * correct / count, "samples": count}
```

### 1.1. Chi tiết từng bước kỹ thuật

| Bước | Thực thi trong mã nguồn | Chi tiết kỹ thuật & Tối ưu hóa |
| :--- | :--- | :--- |
| **`zero_grad`** | `optimizer.zero_grad(set_to_none=True)` | Sử dụng `set_to_none=True` giải phóng bộ nhớ tensor gradient thay vì gán 0, giảm băng thông bộ nhớ và tăng tốc độ xử lý |
| **`forward`** | `logits = model(images)` | Dữ liệu đi qua mạng: Input $\rightarrow$ `data_bn` $\rightarrow$ Stem $\rightarrow$ 5 Stages PDP blocks $\rightarrow$ Head $\rightarrow$ GAP $\rightarrow$ Classifier (5 logits) |
| **`loss`** | `loss = criterion(logits, labels)` | `criterion = torch.nn.CrossEntropyLoss()` kết hợp LogSoftmax và NLLLoss một cách ổn định số học |
| **Bảo vệ số học** | `if not torch.isfinite(loss): raise ...` | Chặn đứng ngay lập tức nếu xuất hiện `NaN` hoặc `Inf` loss, bảo vệ checkpoint không bị corrupt |
| **`backward`** | `loss.backward()` | Kích hoạt Autograd tính đạo hàm riêng $\frac{\partial \mathcal{L}}{\partial w}$ cho toàn bộ tham số có `requires_grad=True` |
| **`step`** | `optimizer.step()` | Cập nhật trọng số theo thuật toán SGD (Momentum = 0.9, Weight Decay = $10^{-4}$, LR = 0.1) |
| **`scheduler`** | `scheduler.step()` | `CosineAnnealingLR(optimizer, T_max=epochs)` hạ dần tốc độ học theo đường cong cosine về 0 |

---

## 2. So sánh Kiến trúc với TickNet ban đầu (`models/TickNet.py`)

Trong đồ án có 3 phiên bản mô hình được xây dựng và đo lường độ phức tạp độc lập:
1. **TickNet-Basic** (Mô hình gốc trong `models/TickNet.py`).
2. **TickNet-L v1** (Ứng viên tinh gọn tính toán trong `models/ticknet_l.py`).
3. **TickNet-C v1** (Ứng viên mở rộng dung lượng dùng nguyên vẹn backbone của thầy trong `models/ticknet_c.py`).

### 2.1. Bảng đối chiếu thông số đo đạc thực tế (`docs/model_profiles.json`)

*Ràng buộc của bài toán: Tham số học được $\le \mathbf{6.000.000}$ (6M) và FLOPs $< \mathbf{1.000.000.000}$ (1 GFLOP) trên ảnh $224 \times 224$ (quy ước $1 \text{ MAC} = 2 \text{ FLOPs}$).*

| Thông số kỹ thuật | TickNet-Basic (Gốc) | TickNet-L v1 | TickNet-C v1 |
| :--- | :---: | :---: | :---: |
| **File mã nguồn** | [models/TickNet.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py) | [models/ticknet_l.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py) | [models/ticknet_c.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_c.py) |
| **Tổng tham số (Params)** | **1.062.223** (~1.06M) | **1.096.260** (~1.10M) | **5.155.467** (~5.16M) |
| **FLOPs (Mid224 - 224x224)** | **0.988343 GFLOPs** (98.8% trần) | **0.796760 GFLOPs** (-19.4%) | **0.821054 GFLOPs** (-16.9%) |
| **FLOPs (Mid32 - 32x32)** | **0.158428 GFLOPs** | **0.157821 GFLOPs** | **0.256830 GFLOPs** |
| **Số lượng PDP Block** | 5 block (1 block/stage) | 7 block (thêm ở stage 3 & 4) | 9 block (sâu nhất) |
| **Độ rộng kênh Stem** | 32 kênh | **24 kênh** | 32 kênh |
| **Cấu trúc Depthwise Conv** | Toàn bộ kernel $3 \times 3$ | **Mixed DW: Nửa $3 \times 3$, nửa $5 \times 5$** | Toàn bộ kernel $3 \times 3$ |
| **Pointwise đầu block (Pw1)** | Tuyến tính ($C_{\text{in}} \rightarrow C_{\text{in}}$) | **Nén kênh ($C_{\text{in}} \rightarrow 0.75 C_{\text{in}}$)** | Tuyến tính ($C_{\text{in}} \rightarrow C_{\text{in}}$) |
| **Độ rộng kênh Head** | 1024 kênh | **768 kênh** | 1024 kênh |

---

### 2.2. Điểm nghẽn của TickNet gốc (`models/TickNet.py`)

Khi phân tích hồ sơ tính toán chuyên sâu tại [docs/MODEL_L.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_L.md):
- Trong bản gốc `TickNet-Basic` ở độ phân giải 224x224, **Stage 2 chiếm tới 0.521 GFLOPs** (tương đương **52.7%** tổng chi phí tính toán toàn mạng!).
- Lý do: Tầng Pointwise 1x1 đầu tiên của Stage 2 phải tính toán trên bản đồ đặc trưng còn quá lớn ($112 \times 112$) với $128$ kênh đầu vào trước khi Depthwise hạ mẫu.
- Kết quả là TickNet-Basic bị "nghẽn cổ chai tính toán", tiêu tốn đến **0.988 GFLOPs** (sát trần 1.0 GFLOPs của đề bài) dù mạng chỉ có 1 triệu tham số và vẻn vẹn 5 block.

---

### 2.3. Hai hướng cải tiến so với bản gốc: TickNet-L và TickNet-C

#### A. Hướng 1: Tối ưu hiệu quả tính toán — `TickNet-L v1`
Nhằm hạ thấp FLOPs để bổ sung thêm tầng xử lý sâu hơn:
1. **Thu hẹp Stem & Head**: Hạ Stem từ 32 xuống 24 kênh, hạ Head từ 1024 xuống 768 kênh.
2. **Nén kênh Pointwise (Pointwise Bottleneck)**: Tầng Pw1 từ Stage 2 áp dụng tỷ lệ nén $0.75 \times C_{\text{in}}$, giảm mạnh chi phí tại Stage 2 từ 0.521 GFLOPs xuống còn 0.333 GFLOPs.
3. **Tăng độ sâu (7 blocks)**: Khoản FLOPs tiết kiệm được được tái đầu tư để tăng số block ở Stage 3 ($144$ kênh) và Stage 4 ($288$ kênh) từ 1 block lên 2 blocks.
4. **Mixed Depthwise Convolution ($3 \times 3 + 5 \times 5$)**: Từ Stage 3 đến Stage 5, nửa số kênh chạy qua DW $3 \times 3$, nửa số kênh chạy qua DW $5 \times 5$. Kỹ thuật này giúp mô hình mở rộng vùng tiếp nhận không gian (receptive field) để bắt trọn các đặc trưng quy mô lớn mà không làm tăng chi phí tính toán.
5. **Kết quả**: Chi phí trên 224x224 giảm xuống **0.796 GFLOPs**, mạng sâu hơn và tiếp nhận trường nhìn tốt hơn.

#### B. Hướng 2: Tối đa hóa dung lượng biểu diễn — `TickNet-C v1`
Nhằm khai thác tối đa giới hạn 6 triệu tham số mà vẫn tuân thủ 100% mã nguồn của thầy:
1. **Kế thừa 100% code của thầy**: Không đổi thứ tự layer, không đổi kernel size, dùng trực tiếp `TickNet` và `FR_PDP_block` từ `models/TickNet.py`.
2. **Tăng độ sâu lên 9 blocks**: Phân bổ số block theo cấu hình:
   $$\text{Stages} = ((80,), (48,), (96, 128), (192, 224), (640, 768, 896))$$
3. **Mở rộng dung lượng tham số (5.16M params)**: Bung nở số kênh cực mạnh ở Stage 5 ($896$ kênh), giúp mạng học được các biểu diễn đặc trưng phức tạp hơn nhiều so với 1M params của Basic.
4. **Kiểm soát stride thông minh**: Dùng lịch stride `(2, 1, 2, 2, 2)` (kế thừa từ bản Large của thầy), giúp FLOPs ở 224x224 chỉ dừng ở mức **0.821 GFLOPs** (rất an toàn dưới ngưỡng 1.0 GFLOPs).

---

## 3. Sơ đồ Kiến trúc Trực quan Tương tác (Interactive Archify Diagrams)

Dự án tích hợp các sơ đồ kiến trúc động chuẩn Showcase tại thư mục `docs/architectures/`, mở trực tiếp bằng trình duyệt web:
- 📊 **[TickNet-Basic (Gốc)](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_basic.html)**: Mô tả 5 block kinh điển, thể hiện trực quan điểm nghẽn tính toán 52.7% FLOPs tại Stage 2.
- 🚀 **[TickNet-L v1 (Tối ưu FLOPs)](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_l.html)**: Mô tả 7 block, trực quan hóa cơ chế Pointwise Bottleneck 0.75x và phân nhánh Mixed DW 3x3 + 5x5.
- ⚡ **[TickNet-C v1 (Mở rộng Tham số)](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_c.html)**: Mô tả 9 block, trực quan hóa cấu trúc mở rộng 5.16M params, Stage 5 phình to 896ch và lịch stride `(2, 1, 2, 2, 2)`.
- 🔄 **[So sánh Tổng quan Toàn diện (Comparison)](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_comparison.html)**: So sánh trực quan 3 phương án kiến trúc trên cùng một bức tranh vĩ mô.

---

## 4. Liên kết tham chiếu trong dự án
- [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py): Vòng lặp `run_epoch` chứa forward, loss, backward, step.
- [models/TickNet.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py): Kiến trúc gốc của thầy (TickNet-Basic).
- [models/ticknet_l.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py): Kiến trúc tối ưu FLOPs TickNet-L v1 (Mixed DW 3x3/5x5).
- [models/ticknet_c.py](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_c.py): Kiến trúc mở rộng 5.16M tham số TickNet-C v1.
- [docs/MODEL_L.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_L.md) & [docs/MODEL_C.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_C.md): Báo cáo thiết kế và phân tích chi phí FLOPs từng stage.
- [docs/model_profiles.json](file:///home/intern-tdkhuong/Desktop/TickNets/docs/model_profiles.json): Số đo tham số và FLOPs chính thức đo bằng forward pass.
- [docs/architectures/README.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/README.md): Hướng dẫn sử dụng và tương tác với các sơ đồ kiến trúc Archify.
