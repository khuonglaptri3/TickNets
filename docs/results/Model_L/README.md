# Hồ sơ Kết quả & Kiểm định Kỹ thuật: TickNet-L v1 (Model L)

Thư mục này tổng hợp toàn bộ kết quả thực nghiệm, biểu đồ đánh giá, trọng số checkpoint và các dữ liệu chứng minh **tính đúng đắn của phương pháp thực hiện** cho mô hình **TickNet-L v1** (Model L) thuộc bài thi Giữa kỳ và Cuối kỳ.

---

## 1. Tổng quan & Bảng thông số cốt lõi

TickNet-L v1 là biến thể cải tiến cấp độ khối (Block-level) từ kiến trúc TickNet-Basic gốc, được thiết kế nhằm tối ưu hóa ngân sách tính toán (FLOPs) và mở rộng trường tiếp nhận đa tỉ lệ thông qua Mixed Depthwise Convolution.

| Tiêu chí | TickNet-Basic (Baseline) | TickNet-C | **TickNet-L v1 (Được chọn)** | Ngưỡng yêu cầu Đề thi |
| :--- | :---: | :---: | :---: | :---: |
| **Số tham số (Learnable Params)** | 1.062.223 | 5.155.467 | **1.096.260 (~1,10M)** | $\le$ **6.000.000 (6M)** |
| **GFLOPs Mid32 (32×32)** | 0,158428 G | 0,256830 G | **0,157821 G** *(thấp nhất)* | **< 1,0 G** |
| **GFLOPs Mid224 (224×224)** | 0,988343 G | 0,821054 G | **0,796760 G** *(thấp nhất)* | **< 1,0 G** |
| **Top-1 Accuracy Mid32** | 89,6% (224/250) | 91,2% (228/250) | **91,6% (229/250)** | Đánh giá xếp hạng |
| **Macro F1 Mid32** | 0,8956 | 0,9117 | **0,9158** | Đánh giá phân loại |
| **Top-1 Accuracy Mid224** | 92,4% (231/250) | 94,4% (236/250) | **95,6% (239/250)** | Đánh giá xếp hạng |
| **Macro F1 Mid224** | 0,9240 | 0,9439 | **0,9559** | Đánh giá phân loại |

*Quy ước FLOPs: 1 MAC = 2 FLOPs, chế độ eval, batch size 1, Conv2d & Linear.*

---

## 2. Cấu trúc thư mục kết quả `Model_L`

```text
Model_L/
├── README.md                              # Hồ sơ tổng hợp toàn diện (Tài liệu này)
├── report_assets/                         # Biểu đồ, ma trận nhầm lẫn, báo cáo phân loại & profile
│   ├── mid32_learning_curves.png          # Đường cong huấn luyện Mid32 (Loss & Accuracy qua 200 epochs)
│   ├── mid32_confusion_matrix.png         # Ma trận nhầm lẫn 5 lớp trên tập test Mid32
│   ├── mid32_confusion_matrix.csv         # Dữ liệu số ma trận nhầm lẫn Mid32
│   ├── mid32_classification_report.csv    # Precision, Recall, F1 từng lớp trên Mid32
│   ├── mid224_learning_curves.png         # Đường cong huấn luyện Mid224 (Loss & Accuracy qua 200 epochs)
│   ├── mid224_confusion_matrix.png        # Ma trận nhầm lẫn 5 lớp trên tập test Mid224
│   ├── mid224_confusion_matrix.csv        # Dữ liệu số ma trận nhầm lẫn Mid224
│   ├── mid224_classification_report.csv   # Precision, Recall, F1 từng lớp trên Mid224
│   ├── dataset_properties_by_class.csv    # Số lượng ảnh train/test trên từng lớp (5.000 train / 50 test)
│   ├── midterm_required_summary.csv       # Bảng tổng kết thông số theo đúng mẫu báo cáo
│   └── model_profiles_l.json              # Chi tiết tham số và FLOPs từng layer/stage
├── comparisons/                           # Đồ thị và bảng số liệu so sánh trực diện L vs Basic & C
│   ├── accuracy_by_dataset.png            # Biểu đồ cột so sánh Accuracy giữa 3 mô hình
│   ├── macro_f1_by_dataset.png            # Biểu đồ so sánh chỉ số Macro F1
│   ├── accuracy_vs_compute_mid224.png     # Đồ thị phân tán Trade-off: Accuracy vs GFLOPs ở Mid224
│   ├── reported_metrics.csv               # Bảng số liệu thô chưa làm tròn của các mô hình
│   └── split_provenance.csv               # Bằng chứng khớp split và cấu hình train
├── training_logs/                         # Lịch sử huấn luyện đầy đủ 200 epochs & checkpoint gốc
│   ├── mid32/
│   │   ├── config.json                    # Cấu hình huấn luyện (Seed 42, Batch 128, SGD, lr 0.1)
│   │   ├── epochs.csv                     # Chi tiết Loss và Accuracy từng epoch (1 -> 200)
│   │   ├── test_metrics.json              # Kết quả đánh giá test độc lập
│   │   └── last.pt                        # Trọng số checkpoint cuối cùng (Epoch 200)
│   └── mid224/
│       ├── config.json                    # Cấu hình huấn luyện (Seed 42, Batch 64, SGD, lr 0.1)
│       ├── epochs.csv                     # Chi tiết Loss và Accuracy từng epoch (1 -> 200)
│       ├── test_metrics.json              # Kết quả đánh giá test độc lập
│       └── last.pt                        # Trọng số checkpoint cuối cùng (Epoch 200)
└── method_soundness_and_verification/     # BẰNG CHỨNG TÍNH ĐÚNG ĐẮN CỦA PHƯƠNG PHÁP & KIỂM ĐỊNH
    ├── METHOD_SOUNDNESS.md                # Tài liệu thuyết minh chi tiết tính đúng đắn toán học & kỹ thuật
    ├── block_equivalence.json             # Bằng chứng thu về block gốc FR-PDP với sai số = 0.0
    ├── verification.json                  # Kết quả kiểm định độc lập (FlopCounterMode, SHA-256, metrics)
    ├── source_checks.json                 # Kết quả kiểm tra tính toàn vẹn mã nguồn & unit tests
    ├── verify_evidence.py                 # Script tự động tính toán lại và xác minh mọi bằng chứng
    ├── l_mid32_predictions.csv            # Dự đoán chi tiết từng ảnh test trên Mid32
    └── l_mid224_predictions.csv           # Dự đoán chi tiết từng ảnh test trên Mid224
```

---

## 3. Tóm lược tính đúng đắn của phương pháp (Method Soundness)

Xem chi tiết tại: [method_soundness_and_verification/METHOD_SOUNDNESS.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/METHOD_SOUNDNESS.md).

### 3.1. Động lực thiết kế giải quyết nút thắt tính toán (Computational Bottleneck)
- **Điểm yếu của Basic gốc**: Ở ảnh $224 \times 224$, riêng Stage 2 chiếm tới **52,7% tổng FLOPs toàn mạng** (0,521 GFLOPs) vì phép Pointwise 1×1 chạy trên tensor lớn $112 \times 112$ với 128 kênh trước khi downsample.
- **Giải pháp của Model L**:
  1. Thêm tỷ lệ nén kênh tuyến tính: $\text{hidden} \approx 0,75 C_{in}$ tại Pointwise đầu (giảm kênh ẩn xuống 88 ở stage 2), giúp FLOPs Stage 2 giảm xuống **0,333 GFLOPs** (tiết kiệm ~36%).
  2. Khoản FLOPs dôi dư được tái phân bổ vào Stage 3 và Stage 4 (tăng từ 1 lên 2 block mỗi stage), giúp mạng sâu hơn ở các tầng trích xuất ngữ cảnh cao cấp.
  3. Tích hợp **Mixed Depthwise Convolution (MixConv)**: Tách kênh chạy song song DW $3 \times 3$ và DW $5 \times 5$, mở rộng trường tiếp nhận không gian đa tỉ lệ mà giữ nguyên tính chất nhẹ của depthwise.

### 3.2. Bằng chứng toán học & lập trình
- **Block Equivalence**: Khi đặt $\text{hidden} = C_{in}$ và $\text{kernels} = (3,)$, block của Model L thu về giống 100% với block FR-PDP nguyên bản của thầy, với sai số số học lớn nhất $\mathbf{= 0,0}$ trên mọi cấu hình kiểm thử ([block_equivalence.json](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/block_equivalence.json)).
- **Gradient Flow**: 32 bài unit test bao phủ forward/backward đều **PASSED**, gradient tồn tại và hữu hạn trên mọi tham số học được, đường tắt residual và SE attention hoạt động chính xác.
- **Tính toán FLOPs chuẩn xác**: Phép đếm hook khớp 100% với `torch.utils.flop_counter.FlopCounterMode` của PyTorch ([verification.json](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/verification.json)).

### 3.3. Kiểm định dữ liệu & Tái lập độc lập
- **Tính toàn vẹn dữ liệu**: 50.500 ảnh local được kiểm tra mã băm SHA-256 khớp tuyệt đối với manifest phân chia tập train/test. Cả 3 mô hình đều được huấn luyện trên cùng một tập dữ liệu phân chia chuẩn.
- **Tái lập kết quả**: Chạy lại suy luận từ checkpoint `last.pt` tái tạo chính xác 100% độ chính xác Top-1 (91,6% trên Mid32 và 95,6% trên Mid224) và toàn bộ confusion matrix ban đầu.

---

## 4. Chi tiết kết quả thực nghiệm Model L

### 4.1. Tập dữ liệu Mid224 ($224 \times 224$)
- **Top-1 Accuracy**: **95,6%** (239/250 ảnh test đúng).
- **Macro F1**: **0,9559**.
- **Hiệu năng từng lớp**:
  - `horse`: Precision 100%, Recall 100%, F1 1.000 (50/50 ảnh).
  - `frog`: Precision 98,0%, Recall 98,0%, F1 0.980 (49/50 ảnh).
  - `bird`: Precision 94,1%, Recall 96,0%, F1 0.950 (48/50 ảnh).
  - `dog`: Precision 92,0%, Recall 92,0%, F1 0.920 (46/50 ảnh).
  - `cat`: Precision 93,9%, Recall 92,0%, F1 0.929 (46/50 ảnh).

![Ma trận nhầm lẫn Mid224](report_assets/mid224_confusion_matrix.png)
![Đường cong học Mid224](report_assets/mid224_learning_curves.png)

### 4.2. Tập dữ liệu Mid32 ($32 \times 32$)
- **Top-1 Accuracy**: **91,6%** (229/250 ảnh test đúng).
- **Macro F1**: **0,9158**.
- **Đặc điểm phân loại**: Ở kích thước $32 \times 32$, mô hình nhận diện tốt `frog` (F1: 0.969) và `bird` (F1: 0.949); có sự nhầm lẫn giữa `dog` và `cat` (9 ảnh dog bị phân loại nhầm thành cat) do mất các đặc trưng râu và mắt nhỏ khi giảm độ phân giải.

![Ma trận nhầm lẫn Mid32](report_assets/mid32_confusion_matrix.png)
![Đường cong học Mid32](report_assets/mid32_learning_curves.png)

---

## 5. Chiến lược chuyển tiếp sang Đồ án Cuối kỳ (Final Exam)

Đề bài [.doc/Final exam.docx](file:///home/intern-tdkhuong/Desktop/TickNets/.doc/Final%20exam.docx) yêu cầu:
1. Huấn luyện mô hình $L$ trên **CIFAR-10** và **CIFAR-100** với các learning rate (0.1, 0.15,...) và 2 optimizers (**SGD**, **Adam**).
2. Quy định: *"The proposed model L is not the quite same as previous networks, except your proposed model in the midterm examination."*

### Ưu thế tuyệt đối của Model L khi bước vào Final Exam:
1. **Kích thước ảnh CIFAR là $32 \times 32$**:
   - Ở $32 \times 32$, Model L chỉ tốn **0,1578 GFLOPs**, trong khi Model C ngốn tới **0,2568 GFLOPs** (+62,7%).
   - Tốc độ huấn luyện của Model L sẽ nhanh hơn đáng kể, cho phép nhóm chạy đầy đủ toàn bộ lưới thí nghiệm (Grid search các learning rate 0.1, 0.15 và optimizers SGD, Adam trên cả CIFAR-10 lẫn CIFAR-100) mà không lo vượt hạn mức GPU Kaggle.
2. **Khả năng khái quát hóa trên CIFAR-100**:
   - Model C có tới 5,16 triệu tham số. Khi huấn luyện trên 100 lớp của CIFAR-100 (mỗi lớp chỉ có 500 ảnh train), mô hình quá nặng sẽ rất dễ bị **overfitting**.
   - Model L với dung lượng tinh gọn ~1,10 triệu tham số cùng cơ chế đa trường tiếp nhận (Mixed DW $3 \times 3$ và $5 \times 5$) sẽ học tổng quát hóa tốt hơn và bền vững hơn rất nhiều.
3. **Thuyết minh vấn đáp (Oral Exam - 30%)**:
   - Nhóm có thể tự tin bảo vệ tính kế thừa từ Midterm sang Final: giữ nguyên khối kiến trúc đã tối ưu nút thắt Stage 2, mở rộng bộ phân loại (classifier) sang 10 và 100 lớp, và phân tích chi tiết sự hội tụ của SGD vs Adam dưới các tốc độ học khác nhau.

---

## 6. Hướng dẫn tái lập kiểm chứng

Chạy script kiểm tra độc lập tại thư mục gốc của repository:

```bash
python docs/results/Model_L/method_soundness_and_verification/verify_evidence.py
```
