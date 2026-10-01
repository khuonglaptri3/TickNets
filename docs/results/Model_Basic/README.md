# Hồ sơ Kết quả & Kiểm định Kỹ thuật: TickNet-Basic (Baseline của thầy)

Thư mục này tổng hợp toàn bộ kết quả thực nghiệm, biểu đồ đánh giá, trọng số checkpoint, sơ đồ kiến trúc và dữ liệu kiểm định cho mô hình **TickNet-Basic** (mô hình gốc của thầy/tác giả), đóng vai trò là **Baseline mốc so sánh** cho toàn bộ đề tài Giữa kỳ và Cuối kỳ.

---

## 1. Tổng quan & Bảng thông số cốt lõi

TickNet-Basic giữ nguyên 100% kiến trúc gốc của tác giả từ `models/TickNet.py`, `models/common.py` và `models/SE_Attention.py`. Mô hình được cấu hình đầu ra 5 lớp để huấn luyện và đánh giá trên hai tập dữ liệu **Mid32** ($32 \times 32$) và **Mid224** ($224 \times 224$).

| Tiêu chí đánh giá | **TickNet-Basic (Baseline của thầy)** | TickNet-C | TickNet-L v1 | Ngưỡng Giới hạn Đề thi |
| :--- | :---: | :---: | :---: | :---: |
| **Số tham số (Learnable Params)** | **1.062.223 (~1,06M)** | 5.155.467 | 1.096.260 | $\le$ **6.000.000 (6M)** |
| **GFLOPs Mid32 (32×32)** | **0,158428 G** | 0,256830 G | 0,157821 G | **< 1,0 G** |
| **GFLOPs Mid224 (224×224)** | **0,988343 G** *(sát trần 1G)* | 0,821054 G | 0,796760 G | **< 1,0 G** |
| **Top-1 Accuracy Mid32** | **89,6% (224/250)** | 91,2% (228/250) | 91,6% (229/250) | Đánh giá xếp hạng lớp |
| **Macro F1 Mid32** | **0,8956** | 0,9117 | 0,9158 | Đánh giá phân loại |
| **Top-1 Accuracy Mid224** | **92,4% (231/250)** | 94,4% (236/250) | 95,6% (239/250) | Đánh giá xếp hạng lớp |
| **Macro F1 Mid224** | **0,9240** | 0,9439 | 0,9559 | Đánh giá phân loại |

*Quy ước đo FLOPs: 1 MAC = 2 FLOPs, batch size 1, chế độ eval, chỉ tính Conv2d và Linear.*

---

## 2. Cấu trúc thư mục kết quả `Model_Basic`

```text
Model_Basic/
├── README.md                                          # Tài liệu tổng hợp toàn diện (Hồ sơ này)
├── TickNet_Model_Baseline_Basic_Architecture.drawio   # Sơ đồ thiết kế kiến trúc trực quan (Draw.io)
│
├── report_assets/                                     # Biểu đồ và dữ liệu phục vụ viết báo cáo
│   ├── mid32_learning_curves.png                      # Đồ thị học Mid32 (Loss & Top-1 qua 200 epochs)
│   ├── mid32_confusion_matrix.png                     # Heatmap ma trận nhầm lẫn 5 lớp (Mid32)
│   ├── mid32_confusion_matrix.csv                     # Số liệu ma trận nhầm lẫn (Mid32)
│   ├── mid32_classification_report.csv                # Precision, Recall, F1 từng lớp (Mid32)
│   ├── mid224_learning_curves.png                     # Đồ thị học Mid224 (Loss & Top-1 qua 200 epochs)
│   ├── mid224_confusion_matrix.png                    # Heatmap ma trận nhầm lẫn 5 lớp (Mid224)
│   ├── mid224_confusion_matrix.csv                    # Số liệu ma trận nhầm lẫn (Mid224)
│   ├── mid224_classification_report.csv               # Precision, Recall, F1 từng lớp (Mid224)
│   ├── dataset_properties_by_class.csv                # Phân bố số lượng ảnh train/test theo từng lớp
│   ├── midterm_required_summary.csv                   # Bảng tổng kết thông số chuẩn theo yêu cầu đề bài
│   └── model_profiles_basic.json                      # Hồ sơ tham số & FLOPs chi tiết từng tầng của Basic
│
├── comparisons/                                       # Đồ thị & số liệu đối chiếu Basic vs C và L
│   ├── accuracy_by_dataset.png                        # So sánh Top-1 Accuracy 3 mô hình
│   ├── macro_f1_by_dataset.png                        # So sánh Macro F1 3 mô hình
│   ├── accuracy_vs_compute_mid224.png                 # Đồ thị phân tán Trade-off Accuracy vs GFLOPs ở Mid224
│   ├── reported_metrics.csv                           # Bảng tổng hợp số liệu gốc chưa làm tròn
│   └── split_provenance.csv                           # Bằng chứng khớp split và hash dữ liệu
│
├── training_logs/                                     # Lịch sử huấn luyện 200 epochs & checkpoint gốc
│   ├── mid32/
│   │   ├── config.json                                # Cấu hình train (Seed 42, Batch 64, SGD, lr 0.1)
│   │   ├── epochs.csv                                 # Chi tiết Loss và Accuracy từng epoch (1 -> 200)
│   │   ├── test_metrics.json                          # Kết quả kiểm thử test độc lập
│   │   └── last.pt                                    # Trọng số checkpoint cuối cùng (Epoch 200, 8.3MB)
│   └── mid224/
│       ├── config.json                                # Cấu hình train (Seed 42, Batch 64, SGD, lr 0.1)
│       ├── epochs.csv                                 # Chi tiết Loss và Accuracy từng epoch (1 -> 200)
│       ├── test_metrics.json                          # Kết quả kiểm thử test độc lập
│       └── last.pt                                    # Trọng số checkpoint cuối cùng (Epoch 200, 8.3MB)
│
└── verification_and_predictions/                      # Dữ liệu kiểm định độc lập & dự đoán từng ảnh test
    ├── basic_mid32_predictions.csv                    # Dự đoán chi tiết từng ảnh test trên Mid32 (250 ảnh)
    ├── basic_mid224_predictions.csv                   # Dự đoán chi tiết từng ảnh test trên Mid224 (250 ảnh)
    ├── verification.json                              # Biên bản audit kiểm định độc lập ngày 2026-10-01
    └── source_checks.json                             # Kết quả kiểm tra đối chiếu git hash và unit tests
```

---

## 3. Phân tích kiến trúc TickNet-Basic & Điểm nghẽn tính toán (Bottleneck)

### 3.1. Cấu trúc khối FR-PDP gốc của thầy
Mỗi block FR-PDP trong TickNet-Basic tuân theo trình tự xử lý:
```text
x ───► PW1 1×1 (Cin → Cin, Tuyến tính)
       │
       ▼
      DW 3×3 (Kích thước kernel cố định, Stride tương ứng) + BN + ReLU
       │
       ▼
      PW2 1×1 (Cin → Cout) + BN + ReLU
       │
       ▼
      Squeeze-and-Excitation (SE Attention)
       │
       ▼
      (+) ◄── Shortcut (Identity nếu cùng shape, hoặc PW Projection)
```

- **Phân bổ kênh**: Stem 32 kênh $\to$ Stage 1 (128) $\to$ Stage 2 (64) $\to$ Stage 3 (128) $\to$ Stage 4 (256) $\to$ Stage 5 (512) $\to$ Head 1024 $\to$ GAP $\to$ Classifier (5 lớp).
- Mỗi stage chỉ gồm đúng **1 block** FR-PDP (tổng cộng 5 block).

### 3.2. Điểm nghẽn tính toán cốt lõi (Computational Bottleneck)
Phân tích chi tiết trong [model_profiles_basic.json](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_Basic/report_assets/model_profiles_basic.json) chỉ ra:
- Ở độ phân giải $224 \times 224$, riêng **Stage 2 tiêu tốn tới 0,52103 GFLOPs**, tương đương **52,7% tổng chi phí toàn mạng**.
- **Nguyên nhân**: Phép Pointwise PW1 ở đầu Stage 2 phải nhận tensor $112 \times 112$ với 128 kênh trước khi Depthwise thực hiện giảm kích thước (stride 2).
- **Hậu quả**: Tổng FLOPs của Basic ở $224 \times 224$ chạm mức **0,988343 GFLOPs** — cực kỳ sát ngưỡng trần $1,0\text{ G}$ của đề bài (chỉ còn dư 1,17%). Mô hình không thể tăng thêm chiều sâu (depth) ở các tầng sau nếu không tối ưu lại stage này.
- *Đây chính là cơ sở và động lực quan trọng để nhóm nghiên cứu và đề xuất biến thể cải tiến **TickNet-L v1**.*

---

## 4. Chi tiết kết quả thực nghiệm TickNet-Basic

### 4.1. Kết quả trên Mid224 ($224 \times 224$)
- **Top-1 Accuracy**: **92,4%** (231/250 ảnh test đúng).
- **Macro F1**: **0,9240**.
- **Hiệu năng từng lớp**:
  - `horse`: Precision 96,1%, Recall 98,0%, F1 0.970 (49/50 ảnh).
  - `frog`: Precision 97,9%, Recall 94,0%, F1 0.959 (47/50 ảnh).
  - `bird`: Precision 90,0%, Recall 90,0%, F1 0.900 (45/50 ảnh).
  - `cat`: Precision 89,8%, Recall 88,0%, F1 0.889 (44/50 ảnh).
  - `dog`: Precision 88,5%, Recall 92,0%, F1 0.902 (46/50 ảnh).

![Ma trận nhầm lẫn Mid224](report_assets/mid224_confusion_matrix.png)
![Đường cong học Mid224](report_assets/mid224_learning_curves.png)

### 4.2. Kết quả trên Mid32 ($32 \times 32$)
- **Top-1 Accuracy**: **89,6%** (224/250 ảnh test đúng).
- **Macro F1**: **0,8956**.
- **Đặc điểm phân loại**: Ở kích thước $32 \times 32$, mô hình nhận diện tốt `frog` (F1: 0.960) và `horse` (F1: 0.939); độ chính xác giảm ở `bird` (88,0%), `cat` (82,0%) và `dog` (86,0%) do các chi tiết nhận diện bị mờ khi giảm độ phân giải xuống 32×32.

![Ma trận nhầm lẫn Mid32](report_assets/mid32_confusion_matrix.png)
![Đường cong học Mid32](report_assets/mid32_learning_curves.png)

---

## 5. Quy trình huấn luyện & Kaggle Notebooks

- **Siêu tham số huấn luyện**: Seed 42, Batch size 64, Optimizer SGD (lr = 0.1, momentum = 0.9, weight decay = 1e-4), CosineAnnealingLR (200 epochs).
- **Kaggle Training Notebooks**:
  - [Basic Mid32 (Epochs 1–200)](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbaseline-mid32)
  - [Basic Mid224 (Epochs 1–100)](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbasic-mid224-epoch001-100-notebook)
  - [Basic Mid224 (Epochs 101–200)](https://www.kaggle.com/code/khngtrnnh/dl-btgk-modelbasic-mid224-epoch101-200-notebook)
- **Quy trình 2 phiên Kaggle**: Do giới hạn thời gian chạy 12 giờ của Kaggle, lượt chạy Mid224 được chia thành 2 session (1–100 và 101–200). Session 2 khôi phục đầy đủ trọng số model, optimizer, scheduler và RNG state từ epoch 100 để tiếp tục chu kỳ cosine annealing 200 epoch.

---

## 6. Kiểm định tính toàn vẹn độc lập

Theo báo cáo kiểm định [verification.json](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_Basic/verification_and_predictions/verification.json) ngày 2026-10-01:
- Cả hai checkpoint `last.pt` (epoch 200) của Basic đều được nạp thành công và kiểm tra SHA-256 nghiêm ngặt.
- Đo lại FLOPs bằng `torch.utils.flop_counter.FlopCounterMode` cho kết quả trùng khớp 100% với báo cáo:
  - Basic Mid32: **0,158428 GFLOPs**.
  - Basic Mid224: **0,988343 GFLOPs**.
- Chạy lại suy luận trên tập test local tái lập khớp hoàn toàn ma trận nhầm lẫn và Top-1 Accuracy: **89,6% (Mid32)** và **92,4% (Mid224)**.
