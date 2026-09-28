# Quy trình làm sạch dữ liệu (Data Cleaning & Quality Assurance)

Module `cleaning/` cung cấp công cụ kiểm định, phát hiện trùng lặp và làm sạch bộ dữ liệu `Mid224` và `Mid32` theo 3 kỹ thuật cốt lõi:

1. **Deduplication (Khử trùng lặp)** bằng Perceptual Hashing (pHash, dHash)
2. **Quality Filtering (Lọc chất lượng)** đo độ mờ (Laplacian blur variance), phơi sáng và tương phản
3. **Object & Label Verification (Kiểm định đối tượng / nhãn rác)** bằng mạng nhẹ MobileNetV3-Small (pretrained ImageNet-1K) hỗ trợ cả CPU và GPU

---

## 1. Phương pháp & Tiêu chí kỹ thuật

### 1.1 Deduplication (pHash & dHash)

* **pHash (Perceptual Hash)**:
  - Thu nhỏ ảnh về 32x32 grayscale và áp dụng biến đổi DCT 2D (`scipy.fftpack.dct`).
  - Lấy 8x8 hệ số tần số thấp (low frequencies), so sánh với giá trị trung vị (median) để tạo chuỗi băm 64-bit (16 ký tự hex).
  - Kháng nhiễu với các phép co giãn kích thước nhỏ, thay đổi độ sáng nhẹ và nén JPEG.
* **Khoảng cách Hamming (Hamming Distance)**:
  - $\text{dist} = 0$: Trùng lặp pHash hoàn toàn.
  - $0 < \text{dist} \le 4$: Cặp ảnh gần trùng (near-duplicate: ảnh chụp góc hơi lệch, crop nhẹ).

### 1.2 Quality Filtering

* **Độ mờ (Blur Variance)**:
  - Tính phương sai toán tử Laplacian trên ảnh xám: $\sigma^2(\nabla^2 I)$.
  - Ảnh có $\sigma^2 < 80$ (với kích thước 224x224) hoặc $< 15$ (với kích thước 32x32) bị gắn cờ `BLURRY`.
* **Phơi sáng & Tương phản**:
  - Độ sáng trung bình $\mu < 35$: Gắn cờ `UNDEREXPOSED` (quá tối).
  - Độ sáng trung bình $\mu > 220$: Gắn cờ `OVEREXPOSED` (cháy sáng / lóa).
  - Độ lệch chuẩn độ sáng $\sigma < 18$: Gắn cờ `LOW_CONTRAST` (ảnh bệt màu / gần đơn sắc).

### 1.3 Object & Label Filtering

* Sử dụng mô hình nhẹ **MobileNetV3-Small** (2.54M tham số, pretrained trên ImageNet-1K):
  - Ánh xạ 1000 lớp ImageNet về 5 lớp mục tiêu:
    - `bird`: 59 loài chim
    - `dog`: 118 giống chó
    - `cat`: 13 loài mèo và thú họ mèo
    - `frog`: 3 loài ếch/cóc (`bullfrog`, `tree frog`, `tailed frog`)
    - `horse`: ngựa (`sorrel`, `zebra`)
  - Tính tổng xác suất mục tiêu $P(\text{class} = c_{\text{target}})$.
  - Nếu xác suất mục tiêu $< 0.05$ và mô hình dự đoán mạnh sang một lớp vật thể khác ($P_{\text{top1}} > 0.20$), mẫu ảnh bị gắn cờ `SUSPICIOUS_LABEL`.

---

## 2. Hướng dẫn chạy kiểm định (Audit)

### 2.1 Quét nhanh kiểm tra (Audit thử nghiệm 500 ảnh)

```bash
python -m cleaning.clean_dataset --variant Mid224 --split test --max-samples 250
```

### 2.2 Quét toàn bộ tập test hoặc train

```bash
# Quét tập test (250 ảnh mỗi độ phân giải)
python -m cleaning.clean_dataset --variant Mid224 --split test --output-dir cleaning/reports

# Quét toàn bộ tập train (25,000 ảnh)
python -m cleaning.clean_dataset --variant Mid224 --split train --batch-size 64 --device auto --output-dir cleaning/reports
```

### 2.3 Chế độ bỏ qua Object Filter (chỉ lọc pHash và Blur)

Nếu cần tốc độ cực nhanh trên CPU (vài giây cho hàng nghìn ảnh):

```bash
python -m cleaning.clean_dataset --variant Mid224 --split train --skip-object-filter
```

---

## 3. Cấu trúc kết quả đầu ra

Các file báo cáo được lưu trong thư mục `--output-dir` (mặc định `cleaning/reports/`): 

- `cleaning_summary_<variant>_<split>.json`: Thống kê tổng số ảnh quét, số lượng ảnh sạch (Clean), tỷ lệ %, phân tích các cờ lỗi vi phạm, số nhóm trùng lặp.
- `flagged_samples_<variant>_<split>.csv`: Danh sách chi tiết từng ảnh bị nghi ngờ kèm lý do cụ thể (`BLURRY`, `UNDEREXPOSED`, `DUPLICATE_PHASH`, `SUSPICIOUS_LABEL`), điểm số đo đạc và đường dẫn file để phục vụ viết báo cáo Midterm.
