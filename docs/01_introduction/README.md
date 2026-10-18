# 01. Giới thiệu và yêu cầu

Phát triển mô hình phân loại ảnh nhẹ từ TickNet-Basic, đánh giá độc lập trên CIFAR-10 và CIFAR-100 (RGB 32×32). Mô hình final là TickNet-L Large v2, 14 khối PDP.

[Đề cuối kỳ](<../../.doc/Final exam.docx>) giới hạn tối đa **6.000.000 tham số học được** và dưới **1.000.000.000 FLOPs** cho một ảnh. Repo lưu thí nghiệm SGD và Adam trên mỗi dataset, tổng bốn run 200 epochs.

Yêu cầu bổ sung: hai file `checkpoint10`, `checkpoint100` phải chạy checkpoint tương ứng, kiểm tra và hiển thị kết quả tốt nhất trong báo cáo. Bản nộp có [checkpoint10.py](../../checkpoint10.py), [checkpoint100.py](../../checkpoint100.py), tương tự `checkpoint224.py`, `checkpoint32.py` ở nhánh giữa kỳ.
