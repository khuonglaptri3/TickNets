# 4. Dataset Splitting: Train / Val / Test & Stratified Split

Tài liệu này tổng hợp chi tiết hiện trạng, phương pháp luận và các bước kỹ thuật đã thực hiện trong đồ án đối với mục **Dataset Splitting**, bao gồm cơ chế phân tầng (Stratified Split), cấu trúc phân chia Train / Val / Test, cơ chế ghép cặp đa độ phân giải (Paired Splitting) và quy trình kiểm thử chống rò rỉ dữ liệu (Data Leakage Prevention).

---

## 1. Phương pháp phân tầng (Stratified Split)

### 1.1. Mục tiêu và thách thức
Bộ dữ liệu gồm 5 lớp đối tượng: `bird`, `cat`, `dog`, `frog`, `horse`.
Nếu áp dụng chia ngẫu nhiên thuần túy (random split) trên toàn bộ 25.250 mẫu, xác suất cao sẽ dẫn đến hiện tượng phân bố lệch giữa các lớp (Class Imbalance) giữa tập Train và Test, làm sai lệch đánh giá khách quan về hiệu năng mô hình.

### 1.2. Thuật toán phân tầng trong đồ án (Per-Class Stratification)
Đồ án áp dụng phương pháp **Stratified Random Hold-Out theo từng lớp**:
- **Cố định hạt giống ngẫu nhiên (Reproducible Seed)**: Sử dụng duy nhất một bộ sinh số ngẫu nhiên `random.Random(seed=42)` cho toàn bộ quá trình, không khởi tạo lại giữa các lớp.
- **Thứ tự duyệt lớp cố định**: Duyệt theo thứ tự bảng chữ cái/chỉ số nhãn chuẩn hóa:
  $$\text{bird (0)} \longrightarrow \text{cat (1)} \longrightarrow \text{dog (2)} \longrightarrow \text{frog (3)} \longrightarrow \text{horse (4)}$$
- **Khử phụ thuộc hệ điều hành (Deterministic Ordering)**:
  Trước khi lấy mẫu, danh sách định danh mẫu (`class_name/filename`) trong từng lớp được sắp xếp theo thứ tự chuỗi (`sorted_ids`).
- **Lấy mẫu không hoàn lại (Sampling without replacement)**:
  Với mỗi lớp, gọi `rng.sample(sorted_ids, 50)` để chọn chính xác **50 mẫu cho tập Test**, toàn bộ **5.000 mẫu còn lại** được gán vào tập Train.

### 1.3. Kết quả phân tầng
- Tỷ lệ mỗi lớp trong tập Train: Đúng $5.000 / 25.000 = \mathbf{20.0\%}$.
- Tỷ lệ mỗi lớp trong tập Test: Đúng $50 / 250 = \mathbf{20.0\%}$.
- **Phân bố cân bằng tuyệt đối** (perfectly balanced), triệt tiêu hoàn toàn độ lệch phân bố nhãn.

---

## 2. Cấu trúc các tập dữ liệu: Train / Val / Test

### 2.1. Thống kê phân bổ

| Tập dữ liệu | Số lượng / lớp | Tổng số lượng | Tỷ lệ dữ liệu | Vai trò & Quy định xử lý trong đồ án |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 5.000 ảnh | **25.000 ảnh** | 99.01% | Dùng huấn luyện mạng TickNet. Tích hợp data augmentation: `RandomCrop` (reflection padding) và `RandomHorizontalFlip`. |
| **Test** | 50 ảnh | **250 ảnh** | 0.99% | **Held-Out Test Set**: Đánh giá độc lập 1 lần duy nhất ở cuối quá trình huấn luyện; không tham gia chọn checkpoint hay tuning siêu tham số. |
| **Validation** | *(Tách từ train)* | — | — | **Chưa đóng gói thư mục `val/` cứng trên ổ đĩa**, tuân thủ nguyên tắc thiết kế chống rò rỉ dữ liệu (xem mục 2.2). |

### 2.2. Giải trình bản chất tập Validation trong đồ án
Trong cấu trúc thư mục đóng gói (`Mid32.tar.xz`, `Mid224.tar.xz` và `data/<variant>/`), đồ án **chủ động không chia sẵn thư mục `val/` vật lý trên ổ đĩa**:
1. **Khớp với yêu cầu đề bài**: Đề tài quy định cụ thể số lượng mẫu nộp là 5.000 train và 50 test mỗi lớp.
2. **Nguyên tắc bảo vệ tập Test (Held-out Integrity)**:
   Tập Test được coi là "hộp đen" chỉ mở một lần cuối cùng. Tuyệt đối không dùng Test để điều chỉnh learning rate, chọn epoch dừng sớm (early stopping) hay chọn seed tốt nhất.
3. **Chiến lược tạo tập Validation khi thực nghiệm**:
   Đồ án quy định rõ trong tài liệu kỹ thuật (`docs/DATASET_SPLIT.md` và `docs/MODEL_C.md`): Khi cần so sánh kiến trúc (Basic vs. L vs. C) hoặc tinh chỉnh siêu tham số (tuning), nhóm nghiên cứu sẽ **tách một phần từ 25.000 ảnh Train** (ví dụ: Stratified K-Fold hoặc tách 10-20% từ tập train) để làm Validation cục bộ, giữ tập Test hoàn toàn biệt lập.

---

## 3. Cơ chế chia đồng bộ đa độ phân giải (Paired Cross-Resolution Splitting)

Một đóng góp kỹ thuật quan trọng của đồ án là xử lý tính nhất quán giữa 2 phiên bản độ phân giải:
- **Mid32** ($32 \times 32$ pixels, phục vụ huấn luyện nhanh và kiểm thử giới hạn tính toán).
- **Mid224** ($224 \times 224$ pixels, độ phân giải tiêu chuẩn thị giác máy tính).

Thay vì chia độc lập 2 bộ dữ liệu:
- Hai bộ dữ liệu được **ghép cặp 1-1 (Paired Mapping)** theo mã định danh `class_name/filename`.
- Thuật toán phân chia chỉ thực thi **một lần duy nhất** trên manifest chung, sau đó ánh xạ đồng thời cho cả Mid32 và Mid224.
- **Cam kết**: Bất kỳ hình ảnh nào ở Mid32 thuộc tập Train (hoặc Test) thì hình ảnh tương ứng tại Mid224 cũng nằm đúng ở tập đó, loại trừ mọi nguy cơ bất tương thích khi đối sánh thực nghiệm liên độ phân giải.

---

## 4. Xác thực tính toàn vẹn & Chống rò rỉ dữ liệu (Data Leakage Verification)

Đồ án đã thực thi 4 tầng kiểm định độc lập để chứng minh tính hợp lệ của phân chia:

1. **Kiểm tra giao tập định danh (Sample ID Disjointness)**:
   $$\text{Train} \cap \text{Test} = \emptyset$$
   Đã kiểm tra bằng unit test tự động trong pipeline (`models/mid_data.py`), nếu phát hiện bất kỳ file nào trùng tên giữa train và test sẽ báo lỗi dừng chương trình.
2. **Kiểm tra trùng pixel thô (Raw Pixel SHA-256 Check)**:
   Giải mã pixel thô của toàn bộ ảnh và tính hash SHA-256. Ghi nhận **0 nhóm ảnh trùng pixel** giữa Train và Test.
3. **Kiểm tra trùng lặp nhận thức (Perceptual Hashing Audit - pHash)**:
   Được ghi nhận tại `docs/DATASET_CLEANING.md`: Sử dụng biến đổi DCT tần số thấp để quét toàn bộ 25.000 ảnh train và 250 ảnh test. Kết quả xác nhận **không có bất kỳ cặp ảnh trùng lặp nhận thức nào xuyên giữa tập Train và Test**.
4. **Truy vết và tái lập (Provenance & Reproducibility)**:
   Toàn bộ kết quả phân chia được khóa cứng trong file `data/split_manifest.csv` với chữ ký SHA-256:
   `9939a6ee404c6fbbdbe1b07a50763dc6d606497709385708512e1e98d71f2780`.
   Trong `train_mid.py`, mã nguồn tích hợp bộ kiểm tra mã băm manifest tự động trước khi cho phép huấn luyện mô hình.

---

## 5. Tài liệu tham chiếu liên quan
- [DATASET_SPLIT.md](../DATASET_SPLIT.md): Báo cáo chi tiết về thông số kích thước, dung lượng TAR.XZ và lệnh tạo lại dataset.
- [DATASET_CLEANING.md](../DATASET_CLEANING.md): Báo cáo kiểm định chất lượng dữ liệu, phân tích quang học và giải mã nhãn cảnh báo.
- `prepare_mid_dataset.py`: Mã nguồn thực thi thuật toán phân tầng và đóng gói.
- `models/mid_data.py`: Pipeline nạp dữ liệu PyTorch với seeded sampler và data augmentation.
