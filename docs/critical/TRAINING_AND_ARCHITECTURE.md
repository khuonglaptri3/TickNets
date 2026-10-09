# 8. Training Pipeline & So sánh Kiến trúc TickNet-L với TickNet gốc của Tác giả

Tài liệu này ghi nhận chi tiết thiết kế, cơ chế kỹ thuật và bằng chứng mã nguồn thực tế trong đồ án cuối kỳ đối với:
1. **Pipeline Huấn luyện Cuối kỳ ([`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py))**: Vòng lặp `forward` $\rightarrow$ `loss` $\rightarrow$ `backward` $\rightarrow$ `optimizer.step()`, bảo vệ số học, lập lịch Cosine Annealing, và lưu trữ artifact.
2. **So sánh Kiến trúc Đối đầu**: Phân tích chi tiết giữa mô hình gốc của tác giả **TickNet-Basic** ([`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py)) và mô hình cải tiến **TickNet-L v1** ([`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py)) trên CIFAR-10 & CIFAR-100.

---

## 1. Cơ chế Vòng Lặp Huấn luyện (Training Loop Mechanics)

Được chuẩn hóa tại hàm `run_epoch` trong [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py):

```python
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered during training")

            if training:
                loss.backward()
                optimizer.step()

            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_count

    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct / total,
        "samples": total,
    }
```

### 1.1. Chi tiết Từng Bước Kỹ thuật
- **`zero_grad(set_to_none=True)`:** Xóa gradient bằng cách gán `None` thay vì tạo tensor số 0, giúp tiết kiệm bộ nhớ đệm và tăng tốc độ xử lý GPU.
- **`non_blocking=True`:** Truyền tensor bất đồng bộ giữa CPU và GPU qua kênh DMA khi bật `pin_memory=True`.
- **Bảo vệ số học (`torch.isfinite`):** Chặn đứng ngay lập tức nếu xuất hiện `NaN` hoặc `Inf` loss, bảo vệ checkpoint không bị sai lệch trọng số.
- **Tính toán Metric có trọng số mẫu:** Đảm bảo batch cuối cùng (batch lẻ) không làm lệch trung bình loss và accuracy của epoch.
- **Lập lịch học Cosine Annealing:** `scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)` giảm dần tốc độ học về 0 theo chu kỳ trơn tru.

---

## 2. So sánh Đối đầu Kiến trúc: TickNet-L v1 vs TickNet-Basic

Mã nguồn hỗ trợ cả hai mô hình thông qua cờ `--model {l, basic}`:

| Đặc tính Kỹ thuật | TickNet-Basic (Tác giả) | TickNet-L v1 (Đề xuất cuối kỳ) |
| :--- | :--- | :--- |
| **Tệp mã nguồn** | [`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py) | [`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py) |
| **Kênh Stem Conv** | 32 kênh | **24 kênh** (tiết kiệm chi phí ở độ phân giải lớn) |
| **Cấu hình Kênh 5 Stages** | [128, 64, 128, 256, 512] | **[112, 64, 144, 288, 512]** (chuyển trọng tâm về stage 3 & 4) |
| **Số khối Blocks mỗi Stage** | [1, 1, 1, 1, 1] (5 blocks) | **[1, 1, 2, 2, 1] (7 blocks sâu hơn)** |
| **Cơ chế Pointwise Conv** | Không nén ($C_{in} \rightarrow C_{in}$) | **Thắt cổ chai (Bottleneck $0.75 \times C_{in}$)** |
| **Cơ chế Depthwise Conv** | Thuần $3 \times 3$ trên mọi kênh | **Mixed DW (chia đôi kênh chạy song song $3 \times 3$ và $5 \times 5$)** |
| **Kênh Conv trước Pooling** | 1024 kênh | **768 kênh** |
| **Số tham số trên CIFAR-10** | **1.067.348** ($\le 6\text{M}$) | **1.100.105** ($\le 6\text{M}$) |
| **Chi phí FLOPs trên CIFAR-10**| **0.1584 GFLOPs** ($< 1\text{G}$) | **0.1578 GFLOPs** ($< 1\text{G}$) |
| **Số tham số trên CIFAR-100**| **1.159.598** ($\le 6\text{M}$) | **1.169.315** ($\le 6\text{M}$) |
| **Chi phí FLOPs trên CIFAR-100**| **0.1586 GFLOPs** ($< 1\text{G}$) | **0.1580 GFLOPs** ($< 1\text{G}$) |

---

## 3. Hệ thống Lưu trữ Artifacts Hoàn chỉnh

Sau mỗi thực nghiệm huấn luyện, pipeline [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py) tự động xuất 6 artifacts phục vụ nghiệm thu và báo cáo khoa học:
1. `config.json`: Toàn bộ siêu tham số, phiên bản code, ngày giờ, số tham số, và số FLOPs.
2. `epochs.csv`: Nhật ký từng epoch gồm Learning Rate, Train Loss, Train Top-1 Acc, Val Loss, Val Top-1 Acc.
3. `best_val.pt`: Checkpoint trọng số đạt kết quả validation tốt nhất (dùng để nộp và evaluate).
4. `last.pt`: Checkpoint epoch cuối cùng (có đầy đủ state của optimizer và scheduler để phục vụ `--resume`).
5. `test_metrics.json`: Độ chính xác Top-1 (%), Loss trung bình, và điểm Macro F1 trên tập Test độc lập.
6. `confusion_matrix.csv`: Ma trận nhầm lẫn kích thước $10 \times 10$ hoặc $100 \times 100$.
