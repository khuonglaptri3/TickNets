# 3. Kiến trúc và độ phức tạp

Mô hình dùng cho hai checkpoint báo cáo là **TickNet-L v1, 7 block**.

| Nội dung | Nguồn |
|---|---|
| Thiết kế, cấu trúc block, lịch kênh và kích thước tensor | [MODEL_L.md](MODEL_L.md) |
| So sánh Basic–L và điểm nghẽn stage 2 | [TRAINING_AND_ARCHITECTURE.md](TRAINING_AND_ARCHITECTURE.md) |
| Mã L v1 | [models/ticknet_l.py](../../models/ticknet_l.py) |
| Mã Basic gốc | [models/TickNet.py](../../models/TickNet.py) |
| Tham số và FLOPs theo layer/stage | [model_profiles.json](model_profiles.json) |
| Hình chèn báo cáo | [figures](figures/README.md) |
| Sơ đồ HTML tương tác | [architectures](architectures/README.md) |
| Kiểm chứng block, gradient và phép đếm | [METHOD_SOUNDNESS.md](../05_results/Model_L/method_soundness_and_verification/METHOD_SOUNDNESS.md) |

L v1 có **1.096.260 tham số**, **0,157821 GFLOPs trên Mid32**, **0,796760 GFLOPs trên Mid224**. Quy ước: một ảnh eval, 1 MAC = 2 FLOPs, đếm Conv2d và Linear.

## Các ứng viên Midterm khác

- [ticknet_l_stage4.py](../../models/ticknet_l_stage4.py): thêm một block ở stage 4.
- [ticknet_l_midterm10.py](../../models/ticknet_l_midterm10.py): ứng viên 10 block cho Mid32/Mid224, CLI `--model l_midterm10`; 1.280.827 tham số. Chưa gán các kết quả 99,2%/97,2% của L v1 cho ứng viên này.

`checkmodel.py` mặc định kiểm tra L v1. Chạy `python checkmodel.py --model l_midterm10` để kiểm tra riêng ứng viên 10 block.
