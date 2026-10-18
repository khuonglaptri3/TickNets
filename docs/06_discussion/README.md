# 06. So sánh và giới hạn

SGD có validation Top-1 cao hơn Adam trên cả hai dataset, nên hai checkpoint SGD được chọn. Test CIFAR-10 đạt 96.04% so với 95.01% (+1.03 điểm phần trăm); CIFAR-100 đạt 81.23% so với 78.04% (+3.19 điểm).

Hai optimizer dùng LR khác nhau (0.175 và 0.001), nên kết quả so sánh hai cấu hình train đã chạy, chưa tách riêng tác động optimizer. Mỗi cấu hình một seed, chưa đo độ lệch chuẩn qua nhiều lần train. Lý do thiết kế kiến trúc chưa thay thế ablation độc lập từng thành phần.

CIFAR-10 có nhầm lẫn cat→dog (53 ảnh), dog→cat (46 ảnh). CIFAR-100 có các lớp khó class_046, class_035, class_072 theo nhãn số của artifact. [Phân tích và hình](../05_results/README.md).

Xác thực dự đoán/metrics đã lưu và chạy inference mới là hai bước riêng. Script checkpoint hiển thị số đo thực tế; loss có thể chênh nhỏ theo PyTorch, batch size và thiết bị. [Phụ lục xác thực](../08_appendix/README.md).
