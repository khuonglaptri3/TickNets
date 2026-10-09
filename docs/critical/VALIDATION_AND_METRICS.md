# 9. Validation & Metrics: Đánh giá Năng lực Mô hình trên CIFAR-10 & CIFAR-100

Tài liệu này ghi nhận chi tiết cơ sở toán học, hiện trạng triển khai trong đồ án cuối kỳ đối với hệ thống độ đo: **Top-1 Accuracy, Loss, Macro F1-Score, và Ma trận nhầm lẫn (Confusion Matrix)** trên hai tập dữ liệu **CIFAR-10** (10 lớp) và **CIFAR-100** (100 lớp).

---

## 1. Cơ sở Toán học của Hệ thống Độ đo (Evaluation Metrics)

Trong bài toán phân loại ảnh nhiều lớp:
- **CIFAR-10:** $|\mathcal{C}| = 10$ lớp đối tượng.
- **CIFAR-100:** $|\mathcal{C}| = 100$ lớp đối tượng.

Đối với từng lớp mục tiêu $c \in \mathcal{C}$:
- **$TP_c$ (True Positive)**: Ảnh thực tế là lớp $c$ và mô hình dự đoán chính xác là lớp $c$.
- **$FP_c$ (False Positive)**: Ảnh thực tế KHÔNG phải là $c$ nhưng mô hình dự đoán nhầm thành $c$ (*Báo động giả / Lỗi Loại I*).
- **$FN_c$ (False Negative)**: Ảnh thực tế là lớp $c$ nhưng mô hình bỏ sót và dự đoán sang lớp khác (*Bỏ sót / Lỗi Loại II*).
- **$TN_c$ (True Negative)**: Ảnh thực tế không phải $c$ và mô hình dự đoán không phải $c$.

### 1.1. Độ chính xác Top-1 (Top-1 Accuracy)
$$\text{Top-1 Accuracy} = \frac{\sum_{c \in \mathcal{C}} TP_c}{N} \times 100\%$$
- Thể hiện tỷ lệ phần trăm mẫu mà xác suất dự đoán cao nhất trùng khớp với nhãn thực tế.

### 1.2. Độ mất mát Trung bình (Average Cross-Entropy Loss)
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \left(\frac{e^{z_{i, y_i}}}{\sum_{j=1}^{C} e^{z_{i, j}}}\right)$$
- Đo lường độ tin cậy và mức độ phạt sai lệch của phân bố xác suất dự đoán so với phân bố one-hot thực tế.

### 1.3. Điểm số Macro F1 (Macro-Averaged F1 Score)
$$\text{F1}_c = \frac{2 \times TP_c}{2 \times TP_c + FP_c + FN_c}$$
$$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{F1}_c$$
- **Ý nghĩa khoa học:** F1-score là trung bình điều hòa giữa Precision và Recall. Macro-F1 tính trung bình F1 của tất cả các lớp với quyền số ngang nhau.
- **Đặc biệt quan trọng trên CIFAR-100:** Với 100 lớp (mỗi lớp chỉ có 100 ảnh test), Macro-F1 vạch trần việc mô hình có thiên vị các lớp dễ nhận biết và "bỏ cuộc" ở các lớp khó hay không.

---

## 2. Ma trận Nhầm lẫn (Confusion Matrix)

Triển khai tại hàm `evaluate` trong [`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py):
- **CIFAR-10:** Ma trận vuông $10 \times 10$ ($10.000$ mẫu test, $1.000$ mẫu/lớp).
- **CIFAR-100:** Ma trận vuông $100 \times 100$ ($10.000$ mẫu test, $100$ mẫu/lớp).
- **Hàng (Rows)**: Nhãn thực tế (Ground Truth $y$).
- **Cột (Columns)**: Nhãn mô hình dự đoán ($\hat{y}$).
- Đường chéo chính biểu diễn $TP_c$. Các phần tử ngoài đường chéo biểu diễn lỗi nhầm lẫn cụ thể giữa các cặp lớp tương đồng (ví dụ: *cat* nhầm sang *dog*, hoặc *automobile* nhầm sang *truck*).

Được tự động xuất ra file `confusion_matrix.csv` và `test_predictions.csv` để trực quan hóa biểu đồ Heatmap trong báo cáo cuối kỳ.

---

## 3. Bằng chứng Triển khai Mã nguồn ([`train_cifar.py`](file:///home/intern-tdkhuong/Desktop/TickNets/train_cifar.py))

```python
def evaluate(model, test_loader, device, output_dir, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    loss_sum, correct, total = 0.0, 0, 0
    predictions_rows = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            loss_sum += loss.item() * labels.numel()
            correct += (preds == labels).sum().item()
            total += labels.numel()

            for target, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                matrix[target][pred] += 1
                predictions_rows.append({"sample_index": len(predictions_rows), "target": target, "prediction": pred})

    # Macro F1 computation
    f1_sum = 0.0
    for i in range(num_classes):
        tp = matrix[i][i]
        fp_plus_fn = sum(matrix[i]) + sum(matrix[r][i] for r in range(num_classes)) - 2 * tp
        f1_sum += (2.0 * tp) / max(1, 2 * tp + fp_plus_fn)
    macro_f1 = f1_sum / num_classes

    result = {
        "top1": 100.0 * correct / total,
        "loss": loss_sum / total,
        "macro_f1": macro_f1,
        "samples": total,
        "correct": correct,
    }
    ...
```

---

## 4. Vai trò trong Đánh giá Đồ án Cuối kỳ

Theo rubric chấm điểm của môn học:
- **Oral Exam (30%):** Giảng viên sẽ chất vấn trực tiếp về nguyên nhân mô hình nhầm lẫn giữa các lớp trong Confusion Matrix và ý nghĩa của Macro-F1.
- **Hiệu năng trên CIFAR-10 (30%) & CIFAR-100 (30%):** Điểm số được xếp hạng đối đầu giữa các nhóm sinh viên dựa trên Top-1 Accuracy trên tập Test.
- **Chất lượng Báo cáo (10%):** Bảng tổng hợp số liệu Top-1, Loss, Macro-F1 và Confusion Matrix heatmap là bằng chứng khoa học không thể thiếu.
