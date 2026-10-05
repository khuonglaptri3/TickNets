# Kiến trúc Mô hình Đề xuất TickNet-L v1 trên CIFAR-10 & CIFAR-100

Tài liệu này ghi nhận chi tiết thiết kế kiến trúc của mô hình **TickNet-L v1** ([`models/ticknet_l.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/ticknet_l.py)), các cải tiến kỹ thuật so với mô hình gốc **TickNet-Basic** của tác giả ([`models/TickNet.py`](file:///home/intern-tdkhuong/Desktop/TickNets/models/TickNet.py)), và chứng minh toán học về tính tuân thủ ngân sách của đề thi cuối kỳ.

---

## 1. Mục tiêu Thiết kế & Ràng buộc Đề bài Cuối kỳ

Đề thi quy định hai điều kiện tiên quyết cho mô hình L:
1. **Số lượng tham số học được (Learnable Parameters):** Không vượt quá **6.000.000 tham số ($\le 6\text{M}$)**.
2. **Chi phí tính toán (FLOPs forward):** Dưới **1.000.000.000 FLOPs ($< 1\text{G}$)**.
3. **Mục tiêu thực nghiệm:** Huấn luyện và đánh giá trên hai bộ dữ liệu chuẩn quốc tế **CIFAR-10** (10 lớp) và **CIFAR-100** (100 lớp) với độ phân giải tự nhiên $32 \times 32$.

---

## 2. Bảng Đối chiếu Chi tiết: TickNet-L v1 vs TickNet-Basic (Tác giả)

| Thành phần Kiến trúc | TickNet-Basic (Tác giả) | TickNet-L v1 (Đề xuất) | Rationale & Ý nghĩa Kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Stem Conv (Khởi đầu)** | 32 kênh, Stride 1 | **24 kênh, Stride 1** | Giảm chi phí tính toán tại tầng đầu khi độ phân giải không gian còn lớn ($32 \times 32$). |
| **Kênh 5 Stages** | [128, 64, 128, 256, 512] | **[112, 64, 144, 288, 512]** | Tái phân bổ ngân sách: Giảm kênh ở Stage 1, tăng biểu diễn ở Stage 3 & 4. |
| **Số Blocks mỗi Stage** | [1, 1, 1, 1, 1] (5 blocks) | **[1, 1, 2, 2, 1] (7 blocks)** | Tăng chiều sâu mạng ở các tầng có spatial nhỏ ($16 \times 16$ và $8 \times 8$), tăng khả năng trích xuất phi tuyến. |
| **Pointwise Đầu Block** | $C_{in} \rightarrow C_{in}$ (Không nén) | **$C_{in} \rightarrow \text{hidden}$ (Tỷ lệ 0.75)** | Giảm $25\%$ số phép nhân ma trận ở tầng Pointwise tốn kém nhất. |
| **Depthwise Convolution** | Thuần $3 \times 3$ trên mọi kênh | **Mixed DW ($3 \times 3$ và $5 \times 5$)** | Stage 1–2 dùng $3 \times 3$; Stage 3–5 chia đôi kênh chạy $3 \times 3$ và $5 \times 5$ song song, thu nhận đa trường nhìn (Multi-scale receptive field). |
| **Squeeze-and-Excitation** | Có ở mọi block ($r=16$) | **Giữ nguyên SE Attention** | Tái hiệu chỉnh trọng số các kênh đặc trưng. |
| **Tầng Conv trước Pooling**| 1024 kênh | **768 kênh** | Tiết kiệm tham số và FLOPs trước khi vào Classifier. |
| **Phân loại (Head)** | 10 hoặc 100 lớp | **10 hoặc 100 lớp** | Tương thích hoàn hảo cả CIFAR-10 và CIFAR-100. |

---

## 3. Cấu trúc Khối FR-PDP Cải tiến trong TickNet-L

Khối FR-PDP v1 kết hợp cơ chế thắt cổ chai (bottlenecking) và Depthwise phân tách đa trường nhìn:

```text
                     +---------- Shortcut Identity / PW 1x1 ----------+
                     |                                                 |
Input (Cin) -> PW 1x1 Linear (hidden = 0.75*Cin) 
            -> Split channels (50% DW 3x3, 50% DW 5x5) 
            -> Concat -> BN + ReLU 
            -> PW 1x1 (Cout) + BN + ReLU 
            -> SE Attention (ChannelGate) 
            -> Cộng với Shortcut (+) -> Output (Cout)
```

Quy tắc làm tròn số kênh ẩn:
$$\text{hidden} = \max\left(16, \left\lfloor \frac{0.75 \times C_{in} + 4}{8} \right\rfloor \times 8\right)$$
Đảm bảo số kênh luôn chia hết cho 8, tối ưu hóa quá trình vector hóa trên GPU Tensor Cores.

---

## 4. Bằng chứng Định lượng Tuân thủ Ngân sách (Complexity Proof)

Đo lường bằng [`models/model_profile.py`](../../models/model_profile.py), phạm vi Conv2d/Linear (không tính BN, activation, pooling và phép toán phần tử), quy ước $1\text{ MAC} = 2\text{ FLOPs}$, batch size = 1, tensor đầu vào $(1, 3, 32, 32)$:

### 4.1. Kết quả trên CIFAR-10 (10 lớp)
- **Số lượng Tham số học được:** **1.100.105 tham số** ($\approx 1.10\text{M} \le 6.000.000$ $\to$ **ĐẠT, chỉ chiếm 18.3% trần cho phép**).
- **Chi phí Tính toán (FLOPs forward):** **157.828.544 FLOPs** ($\approx 0.1578\text{ GFLOPs} < 1.000.000.000$ $\to$ **ĐẠT, chỉ chiếm 15.8% trần cho phép**).

### 4.2. Kết quả trên CIFAR-100 (100 lớp)
- **Số lượng Tham số học được:** **1.169.315 tham số** ($\approx 1.17\text{M} \le 6.000.000$ $\to$ **ĐẠT, chỉ chiếm 19.5% trần cho phép**).
- **Chi phí Tính toán (FLOPs forward):** **157.966.784 FLOPs** ($\approx 0.1580\text{ GFLOPs} < 1.000.000.000$ $\to$ **ĐẠT, chỉ chiếm 15.8% trần cho phép**).

### 4.3. Bảng Tổng Hợp So Sánh Độ Phức Tạp
| Mô hình | Dataset | Tham số (Params) | Tỷ lệ trần Params (6M) | Chi phí FLOPs | Tỷ lệ trần FLOPs (1G) | Trạng thái Tuân thủ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TickNet-Basic** (Tác giả) | CIFAR-10 | 1.067.348 | 17.8% | 0.1584 GFLOPs | 15.8% |  HỢP LỆ |
| **TickNet-Basic** (Tác giả) | CIFAR-100| 1.159.598 | 19.3% | 0.1586 GFLOPs | 15.9% |  HỢP LỆ |
| **TickNet-L v1** (Đề xuất) | CIFAR-10 | **1.100.105** | **18.3%** | **0.1578 GFLOPs** | **15.8%** |  **HỢP LỆ** |
| **TickNet-L v1** (Đề xuất) | CIFAR-100| **1.169.315** | **19.5%** | **0.1580 GFLOPs** | **15.8%** |  **HỢP LỆ** |

*(Cả hai mô hình đều được kiểm thử và xác nhận 100% qua bộ unit test `tests/test_ticknet_l.py`).*
