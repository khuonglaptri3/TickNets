# Tính đúng đắn của phương pháp và giải thuật thiết kế TickNet-L v1

Tài liệu này tổng hợp toàn bộ cơ sở lý thuyết, phân tích toán học, kiến trúc tính toán và bằng chứng thực nghiệm chứng minh **tính đúng đắn (methodological soundness and correctness)** của mô hình **TickNet-L v1** trong khuôn khổ môn học và bài thi Giữa kỳ / Cuối kỳ.

---

## 1. Cơ sở lý thuyết và động lực thiết kế kiến trúc

### 1.1. Phân tích điểm nghẽn tính toán (Bottleneck) trên TickNet-Basic gốc
Trong kiến trúc TickNet-Basic ở độ phân giải $224 \times 224$:
- Stage 2 tiêu thụ tới **0,52103 GFLOPs**, chiếm **52,7% tổng chi phí tính toán Conv/Linear của toàn mạng** (0,98834 GFLOPs).
- Nguyên nhân: Tầng Pointwise 1×1 đầu block (PW1) phải xử lý trên kích thước không gian lớn ($112 \times 112$) với $C_{in} = 128$ kênh trước khi Depthwise thực hiện giảm kích thước (stride 2).
- Hậu quả: Mạng bị "nghẽn tính toán" ở ngay tầng nông, làm cạn kiệt ngân sách FLOPs (< 1G), khiến mạng không thể tăng thêm độ sâu (depth) ở các tầng ngữ cảnh sâu hơn.

### 1.2. Giải pháp nén kênh tuyến tính (Linear Bottleneck)
Model L tái cấu trúc khối PDP bằng cách chèn tỷ lệ nén kênh có kiểm soát:
$$\text{hidden} = \max\left(16, \left\lfloor \frac{0,75 \times C_{in} + 4}{8} \right\rfloor \times 8\right)$$
- Tại Stage 2: Kênh đầu vào được điều chỉnh từ 128 xuống 112, và kênh ẩn $\text{hidden} = 88$.
- Chi phí của Stage 2 giảm mạnh từ **0,521 GFLOPs xuống 0,333 GFLOPs** (tiết kiệm ~36% chi phí tại stage này).
- Tầng PW1 giữ tính chất tuyến tính (không dùng hàm phi tuyến ReLU sau PW1 nén) để tránh phá hủy manifold thông tin trong không gian số chiều thấp (theo nguyên lý MobileNetV2).

### 1.3. Tái phân bổ ngân sách tính toán (FLOP Reallocation)
Lượng FLOPs tiết kiệm được ở tầng nông được tái đầu tư chiến lược vào các tầng sâu:
- **Tăng số block**: Stage 3 tăng từ 1 lên 2 block; Stage 4 tăng từ 1 lên 2 block (xử lý trên spatial map $28 \times 28$ và $14 \times 14$).
- **Tăng dung lượng biểu diễn**: Mạng học được các đặc trưng ngữ cảnh phức tạp hơn mà không làm bùng nổ FLOPs.
- Kết quả: Tổng FLOPs ở $224 \times 224$ giảm từ **0,988 GFLOPs xuống 0,797 GFLOPs** (giảm 19,4%), trong khi số tham số chỉ tăng rất nhẹ từ 1,06M lên 1,10M (+3,2%).

### 1.4. Cơ chế Mixed Depthwise Convolution (MixConv)
Thay vì áp dụng đồng nhất một kích thước kernel $3 \times 3$ trên toàn bộ kênh:
- Kênh ẩn được chia đôi thành 2 nhóm độc lập:
  - Nhóm 1: Áp dụng Depthwise $3 \times 3$ (bắt các chi tiết cục bộ, góc cạnh, texture).
  - Nhóm 2: Áp dụng Depthwise $5 \times 5$ (mở rộng trường tiếp nhận receptive field để bao quát hình dáng tổng thể của vật thể).
- `groups = channels` trên mỗi nhánh, giữ nguyên tính chất tách biệt kênh của Depthwise. Sau đó, tensor được nối lại (concatenate) và đưa qua Pointwise PW2 ($1 \times 1$) để trộn thông tin giữa các nhóm kênh.

---

## 2. Bằng chứng tính đúng đắn về mặt giải thuật và lập trình

### 2.1. Kiểm chứng thu về trường hợp cơ bản (Base-case Equivalence)
Để chứng minh khối `FR_PDP_block_L` không bị sai lệch cấu trúc hay lỗi toán học so với khối `FR_PDP_block` gốc của thầy:
- Khi cấu hình: $\text{hidden} = C_{in}$ và $\text{kernels} = (3,)$, sao chép trọng số từ block gốc sang block L.
- Thực hiện kiểm thử trên 4 cấu hình $(C_{in}, C_{out}, stride)$:
  - $(32, 32, 1)$: Sai số tuyệt đối lớn nhất $\mathbf{= 0,0}$
  - $(32, 64, 1)$: Sai số tuyệt đối lớn nhất $\mathbf{= 0,0}$
  - $(32, 64, 2)$: Sai số tuyệt đối lớn nhất $\mathbf{= 0,0}$
  - $(32, 32, 2)$: Sai số tuyệt đối lớn nhất $\mathbf{= 0,0}$
- Bằng chứng lưu tại file: [`block_equivalence.json`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/block_equivalence.json).

### 2.2. Kiểm chứng gradient và luồng truyền ngược (Backpropagation Check)
- Đã chạy 32 bài unit test bao phủ mô hình (`tests/test_ticknet_l.py`, `tests/test_mid_pipeline.py`,...): **32/32 tests PASSED**.
- Kiểm tra tính toán Forward và Backward ở cả 2 kích thước ($32 \times 32$ và $224 \times 224$).
- Gradient của tất cả tham số học được của Model L (`requires_grad=True`) đều tồn tại, có giá trị hữu hạn (không có NaN, không có Inf).
- Nhánh shortcut identity và projection kết nối trực tiếp, không làm mất luồng đạo hàm, module SE attention đặt chuẩn xác trước phép cộng residual.

### 2.3. Tính toán FLOPs và Tham số độc lập
- Quy ước chuẩn: Một lượt forward ở chế độ `eval`, batch size 1, float32, CPU, $1 \text{ MAC} = 2 \text{ FLOPs}$ (chỉ tính Conv2d và Linear).
- Đối chiếu hai phương pháp đếm độc lập:
  1. Module hook thủ công trong code.
  2. `torch.utils.flop_counter.FlopCounterMode` của PyTorch.
- Kết quả đối chiếu: **Trùng khớp 100%** đến từng phép toán:
  - Mid32: 1.096.260 params, 0,07891 GMACs $\to$ **0,157821 GFLOPs**.
  - Mid224: 1.096.260 params, 0,39838 GMACs $\to$ **0,796760 GFLOPs**.
- Bằng chứng lưu tại: [`verification.json`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/verification.json).

---

## 3. Bằng chứng kiểm định thực nghiệm và tái lập dữ liệu

### 3.1. Xác minh tính toàn vẹn của tập dữ liệu (Dataset Integrity)
- Đối chiếu SHA-256 của toàn bộ **50.500 file ảnh local** với manifest chuẩn: toàn bộ đều khớp.
- Tái tạo CSV schema của notebook Model L từ membership chuẩn thu được đúng hash SHA-256:
  `35bcc76fde54e97d54f7825cdb34376a3732fd79aa829624ea4fc982b4590bdf`.
- Kết luận: Cả ba mô hình Basic, C, L đều được huấn luyện trên **cùng một tập phân chia train/test (25.000 train / 250 test)**, đảm bảo tính công bằng thực nghiệm.

### 3.2. Tái lập độc lập kết quả suy luận từ Checkpoint
Kiểm định độc lập ngày 2026-10-01 đã nạp hai checkpoint `last.pt` (epoch 200) của Model L và thực hiện suy luận lại trên tập test độc lập:
- **Mid32**:
  - Top-1 Accuracy: **91,6%** (229/250 ảnh đúng).
  - Macro F1: **0,91575**.
  - Confusion Matrix tái lập: Trùng khớp 100% ma trận $5 \times 5$ ban đầu.
- **Mid224**:
  - Top-1 Accuracy: **95,6%** (239/250 ảnh đúng).
  - Macro F1: **0,95588**.
  - Confusion Matrix tái lập: Trùng khớp 100% ma trận $5 \times 5$ ban đầu.
- Chi tiết dự đoán từng ảnh được lưu lại tại:
  - [`l_mid32_predictions.csv`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/l_mid32_predictions.csv)
  - [`l_mid224_predictions.csv`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/l_mid224_predictions.csv)

### 3.3. Phân tích thống kê và kiểm định McNemar
- Kiểm định McNemar hai phía trên dự đoán từng ảnh test (250 ảnh):
  - Model L vs Basic Mid32: L đúng 11 ảnh mà Basic sai, Basic đúng 6 ảnh mà L sai ($p = 0,3323$).
  - Model L vs Basic Mid224: L đúng 14 ảnh mà Basic sai, Basic đúng 6 ảnh mà L sai ($p = 0,1153$).
  - Model L vs Model C Mid32: L đúng 11 ảnh mà C sai, C đúng 10 ảnh mà L sai ($p = 1,0000$).
  - Model L vs Model C Mid224: L đúng 8 ảnh mà C sai, C đúng 5 ảnh mà L sai ($p = 0,5811$).
- Khoảng tin cậy Wilson 95% cho Accuracy của Model L:
  - Mid32: $[87,50\% - 94,44\%]$
  - Mid224: $[92,29\% - 97,53\%]$
- Phân tích per-class:
  - Ở Mid224: L nhận diện xuất sắc cả 5 lớp: Horse đạt 100% (50/50), Frog đạt 98% (49/50), Bird đạt 96% (48/50), Cat 92% (46/50), Dog 92% (46/50).
  - Ở Mid32: L gặp khó khăn nhỏ giữa cặp `dog` và `cat` (nhầm 9/50 ảnh dog thành cat), giải thích rõ sự suy giảm độ phân giải từ 224 xuống 32 làm mất các chi tiết phân biệt giữa hai loài vật này.

---

## 4. Hướng dẫn chạy lại mã kiểm chứng tính đúng đắn

Để tự động kiểm chứng lại toàn bộ các bằng chứng trên máy local, kích hoạt môi trường và thực thi script:

```bash
python /home/intern-tdkhuong/Desktop/TickNets/docs/results/Model_L/method_soundness_and_verification/verify_evidence.py
```

Lệnh trên sẽ tự động:
1. Đọc lại sáu checkpoint và tính toán SHA-256.
2. Kiểm tra tính tương đương của manifest membership.
3. Đo lại FLOPs và tham số.
4. Chạy suy luận trên tập test local và so khớp với báo cáo.
