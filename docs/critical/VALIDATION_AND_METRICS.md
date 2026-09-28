# 9. Validation: Accuracy, Precision, Recall, F1-Score & Confusion Matrix

Tài liệu này ghi nhận chi tiết cơ sở toán học, hiện trạng triển khai trong đồ án và luận giải chuyên sâu về câu hỏi: **"Tại sao phải sử dụng hệ thống độ đo (Accuracy, Precision, Recall, F1-score) và Ma trận nhầm lẫn (Confusion Matrix) trong bài toán phân loại ảnh?"**

---

## 1. Cơ sở toán học của các Độ đo đánh giá (Evaluation Metrics)

Bài toán trong đồ án là **phân loại 5 lớp đối tượng**:
$$\mathcal{C} = \{0: \text{bird}, \; 1: \text{cat}, \; 2: \text{dog}, \; 3: \text{frog}, \; 4: \text{horse}\}$$

Đối với mỗi lớp mục tiêu $c \in \mathcal{C}$, không gian dự đoán được chia thành 4 thành phần cơ bản:
- **$TP_c$ (True Positive)**: Mẫu thực tế là lớp $c$ và mô hình dự đoán chính xác là lớp $c$.
- **$FP_c$ (False Positive)**: Mẫu thực tế KHÔNG phải là $c$ nhưng mô hình dự đoán nhầm thành $c$ (*Báo động giả / Lỗi Loại I*).
- **$FN_c$ (False Negative)**: Mẫu thực tế là lớp $c$ nhưng mô hình bỏ sót và dự đoán sang lớp khác (*Bỏ sót / Lỗi Loại II*).
- **$TN_c$ (True Negative)**: Mẫu thực tế không phải $c$ và mô hình dự đoán không phải $c$.

### 1.1. Độ chính xác tổng thể (Accuracy / Top-1 Accuracy)
$$\text{Accuracy} = \frac{\sum_{c \in \mathcal{C}} TP_c}{N} = \frac{\text{Tổng số mẫu dự đoán đúng}}{\text{Tổng số mẫu toàn tập}}$$
- Trong `train_mid.py#L37-L41`: Được tính bằng `correct / count` (tỷ lệ phần trăm Top-1).

### 1.2. Độ chuẩn xác (Precision / Positive Predictive Value)
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$
- **Ý nghĩa thực tế**: *"Trong tất cả các ảnh mà mô hình tuyên bố là 'Mèo', có bao nhiêu phần trăm thực sự là Mèo?"*
- Đo lường **độ tin cậy** của lời dự đoán. Precision thấp đồng nghĩa với việc mô hình hay đoán bừa, dễ bị ảo giác gán nhãn sai.

### 1.3. Độ nhạy / Độ bao phủ (Recall / Sensitivity / True Positive Rate)
$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
- **Ý nghĩa thực tế**: *"Trong tất cả các bức ảnh 'Mèo' thực tế có trong tập dữ liệu, mô hình tìm ra và nhận diện được bao nhiêu phần trăm?"*
- Đo lường **khả năng không bỏ sót**. Recall thấp nghĩa là mô hình bị "mù" trước một bộ phận mẫu của lớp đó.

### 1.4. Điểm số F1 (F1-Score / Balanced F-Score)
$$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c} = \frac{2 TP_c}{2 TP_c + FP_c + FN_c}$$
- Là **trung bình điều hòa (Harmonic Mean)** giữa Precision và Recall.
- **Đặc tính toán học**: F1-Score phạt rất nặng nếu một trong hai chỉ số Precision hoặc Recall bị lệch quá thấp. Một mô hình chỉ đạt F1 cao khi và chỉ khi **vừa đoán chuẩn (Precision cao), vừa không bỏ sót (Recall cao)**.
- **Macro-F1 (Đa lớp)**:
  $$\text{Macro-F1} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} F1_c$$
  Đánh giá bình đẳng vai trò của cả 5 lớp đối tượng.

---

## 2. Tại sao phải sử dụng Ma trận nhầm lẫn (Confusion Matrix) & Hệ độ đo này?

### 2.1. Cạm bẫy của việc chỉ dùng duy nhất Accuracy
Accuracy là chỉ số phổ biến nhất nhưng lại có **nhược điểm chí tử: che giấu bản chất lỗi phân loại**:
- Giả sử mô hình đạt **Accuracy = 80.0%**. Con số này trông rất khả quan, nhưng thực tế có thể xảy ra kịch bản:
  - Lớp `bird`: Đúng $50/50$ ($100\%$)
  - Lớp `frog`: Đúng $50/50$ ($100\%$)
  - Lớp `horse`: Đúng $50/50$ ($100\%$)
  - Lớp `cat`: Đúng $35/50$ ($70\%$)
  - Lớp `dog`: Đúng **$15/50$ ($30\%$)** $\rightarrow$ Mô hình hoàn toàn thất bại trong việc phân biệt chó!
- Nếu chỉ nhìn vào con số $80\%$, người làm nghiên cứu sẽ không thể phát hiện ra mạng đang bị suy thoái nghiêm trọng ở lớp `dog`.

### 2.2. Vai trò vượt trội của Ma trận nhầm lẫn (Confusion Matrix)
Ma trận nhầm lẫn là một bảng vuông $5 \times 5$:
- **Hàng (Rows)**: Nhãn thực tế (Ground Truth Labels).
- **Cột (Columns)**: Nhãn mô hình dự đoán (Predicted Labels).

```text
               DỰ ĐOÁN (Predicted)
              Bird   Cat   Dog   Frog  Horse
THỰC  Bird   [ 48     0     1     1      0  ]  -> Recall Bird  = 48/50 = 96%
TẾ    Cat    [  0    38    10     0      2  ]  -> Recall Cat   = 38/50 = 76%
(True)Dog    [  1    11    35     0      3  ]  -> Recall Dog   = 35/50 = 70%
      Frog   [  0     0     0    49      1  ]  -> Recall Frog  = 49/50 = 98%
      Horse  [  0     1     3     0     46  ]  -> Recall Horse = 46/50 = 92%
                |     |     |     |      |
             Prec. Prec. Prec. Prec.  Prec.
```

**3 Giá trị cốt lõi chỉ có được từ Confusion Matrix:**
1. **Chẩn đoán cặp lớp dễ nhầm lẫn (Semantic Ambiguity Diagnosis)**:
   - Ma trận chỉ đích danh các ô ngoài đường chéo: Ví dụ `Cat` bị đoán thành `Dog` (10 ảnh) và `Dog` bị đoán thành `Cat` (11 ảnh). Điều này phản ánh sự tương đồng hình thái học (tai, mõm, bốn chân, lông) giữa 2 loài động vật ăn thịt nhỏ.
2. **Phân tích hình học đặc trưng (Feature Disentanglement)**:
   - Giúp đánh giá xem biểu diễn không gian ẩn (latent space) của mạng TickNet có tách biệt rạch ròi các cụm đặc trưng hay đang bị dính chùm giữa các loài thú bốn chân.
3. **Cơ sở cho việc tinh chỉnh kiến trúc hoặc Data Augmentation**:
   - Nếu `Cat` và `Dog` hay nhầm nhau, ta có thể bổ sung các phép augmentation tăng độ tương phản vùng mặt hoặc tinh chỉnh head phân loại.

---

## 3. Hiện trạng Triển khai trong Đồ án TickNets

### 3.1. Những gì đã có sẵn trong Pipeline:
- Trong [train_mid.py#L35-L41](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L35-L41), hàm `run_epoch` theo dõi liên tục qua từng epoch:
  - `loss`: Cross-entropy loss trung bình có trọng số.
  - `top1`: Top-1 Accuracy chính xác theo số lượng mẫu thực tế.
- Kết quả được ghi nhận vào `epochs.csv` và `test_metrics.json`.

### 3.2. Đoạn mã mở rộng tính toán trọn bộ Metrics khi Đánh giá Checkpoint:
Khi chạy đánh giá mô hình cuối cùng, việc trích xuất trọn bộ Precision, Recall, F1 và Confusion Matrix có thể thực thi đơn giản như sau:

```python
import torch
from sklearn.metrics import classification_report, confusion_matrix

@torch.inference_mode()
def evaluate_full_metrics(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    class_names = ["bird", "cat", "dog", "frog", "horse"]
    
    # 1. Ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    
    # 2. Báo cáo Precision, Recall, F1 theo từng lớp và Macro-F1
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    return cm, report
```

---

## 4. Tóm tắt giá trị học thuật cho Báo cáo Giữa kỳ
Khi trình bày mục **Validation & Evaluation Metrics**, việc sử dụng kết hợp bộ tứ:
$$\{\text{Accuracy}, \; \text{Per-Class Precision}, \; \text{Per-Class Recall}, \; \text{Macro-F1}\} \; + \; \text{Confusion Matrix}$$
chứng minh:
1. **Tính khách quan và toàn diện**: Không bị "bẫy số liệu" bởi một con số Accuracy duy nhất.
2. **Hiểu sâu sắc về dữ liệu và mô hình**: Phân tích được các ca nhầm lẫn biên (Edge cases) giữa các loài động vật tương đồng.
3. **Tiêu chuẩn học thuật quốc tế**: Đúng chuẩn trình bày của các hội nghị thị giác máy tính hàng đầu (CVPR, ICCV, ECCV).
