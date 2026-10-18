# Dữ liệu CIFAR

Chạy `python checkpoint10.py` hoặc `python checkpoint100.py` để tải archive CIFAR và chuẩn bị tập test bằng torchvision. Nếu dữ liệu gốc đã có, dùng `--no-download`; dùng `--data-root` để chọn thư mục lưu.

Ảnh và archive tải xuống được bỏ qua bởi Git. Hai file checkpoint kiểm tra SHA-256 của pixel, nhãn và thứ tự trước inference. Mã train và bước chuẩn bị dữ liệu trên Kaggle nằm trong nguồn nhúng của từng notebook.

[Split và preprocessing](../docs/02_dataset/README.md) · [Kaggle](../docs/04_training/README.md) · [Chạy checkpoint](../docs/08_appendix/README.md).
