# 2. Dữ liệu và tiền xử lý

| Nội dung cần viết | Nguồn |
|---|---|
| Năm lớp, số ảnh, paired split và dung lượng | [DATASET_SPLIT.md](DATASET_SPLIT.md) |
| Giải thích chia phân tầng | [DATASET_SPLITTING.md](DATASET_SPLITTING.md) |
| Kiểm định chất lượng, ảnh trùng và nhãn nghi ngờ | [DATASET_CLEANING.md](DATASET_CLEANING.md) |
| ToTensor, data_bn, crop và flip | [PREPROCESSING.md](PREPROCESSING.md) |
| Seed, số lượng và byte chính xác | [data/split_config.json](../../data/split_config.json) |
| Danh sách mẫu và checksum ảnh | [data/split_manifest.csv](../../data/split_manifest.csv) |
| Kết quả kiểm tra dataset | [data/validation_report.json](../../data/validation_report.json) |
| Số ảnh theo lớp để dựng bảng | [dataset_properties_by_class.csv](../05_results/Model_L/report_assets/dataset_properties_by_class.csv) |

Mỗi độ phân giải có 25.000 train + 250 test. Hai bản có cùng membership theo class/filename. Khi tuning, validation được tách từ train theo [protocol](../04_training/MIDTERM_PROTOCOL.md).

## Điểm cần trình bày đúng

- Mid224: tổng JPEG **31,803510 MB**, gói TAR.XZ **24,610144 MB**. Mid32: tổng JPEG **19,057221 MB**, gói TAR.XZ **12,306416 MB**. MB = 1.000.000 byte; ghi rõ giới hạn đang được đối chiếu với gói nén.
- Pipeline chuẩn bị dữ liệu sao chép ảnh đã đúng kích thước. Cần bổ sung nguồn thu thập và phương pháp resize/nén ban đầu vào báo cáo.
- Cảnh báo từ bộ lọc nhãn không đồng nghĩa ảnh chắc chắn sai nhãn; không biến tỷ lệ cảnh báo thành tỷ lệ lỗi ground truth đã xác nhận.

Mã liên quan: [prepare_mid_dataset.py](../../prepare_mid_dataset.py), [mid_data.py](../../models/mid_data.py), [cleaning](../../cleaning/README.md).
