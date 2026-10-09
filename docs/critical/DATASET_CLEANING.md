# Kiểm tra tính toàn vẹn dữ liệu CIFAR

Final dùng CIFAR-10 và CIFAR-100 qua [downloader](../../scripts/download_cifar.py)
và [loader](../../models/cifar_data.py), không dùng bộ lọc dữ liệu giữa kỳ.

- Downloader kiểm tra checksum archive và các batch đã giải nén; cache thiếu/hỏng không được coi là hợp lệ.
- 50.000 ảnh train chính thức được chia phân tầng thành 45.000 train và 5.000 validation, seed 42.
- 10.000 ảnh test chính thức được giữ riêng; không dùng để chọn epoch hoặc learning rate.
- Crop, flip và Cutout chỉ áp dụng cho train. Validation/test chỉ chuyển tensor và chuẩn hóa.
- `config.json` lưu SHA-256 của dữ liệu và membership từng split. Resume kiểm tra lại bằng chứng này.
- Đây là kiểm tra tính toàn vẹn và phân tách index, không phải bằng chứng đã loại mọi ảnh gần trùng về nội dung trong benchmark gốc.

Kết quả đã lưu nằm trong [báo cáo final](../experiments/README.md).
Validator đối chiếu log, dự đoán, confusion matrix, metric và checksum artifact;
việc đối chiếu artifact không thay thế chạy inference lại trên dữ liệu CIFAR gốc.
