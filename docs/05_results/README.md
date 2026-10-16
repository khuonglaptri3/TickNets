# 5. Kết quả thực nghiệm

Xem [báo cáo tổng hợp](../MIDTERM_REPORT.md) để đọc đủ phương pháp, kết quả hai checkpoint được chọn, hình và phụ lục trong một file Markdown. Hình/bảng của báo cáo nằm trong [selected_checkpoints](selected_checkpoints).

Các thư mục bằng chứng được giữ cùng nhau để config, log, checkpoint, predictions và biểu đồ của mỗi run không bị tách rời.

| Nhóm kết quả | Nguồn chính |
|---|---|
| Basic và L lịch sử, train đủ 25.000 ảnh | [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md) và [model_comparison](model_comparison/README.md) |
| Smoothing và learning rate, có validation | [experiment_midterm_smoothing_lr](experiment_midterm_smoothing_lr/README.md) |
| Tăng độ sâu stage 4 | [experiment_midterm_stage4_depth](experiment_midterm_stage4_depth/EXPERIMENT_REPORT.md) |
| Định nghĩa các độ đo | [VALIDATION_AND_METRICS.md](VALIDATION_AND_METRICS.md) |

## Kết quả dùng để trình diễn checkpoint

| Dataset | Cấu hình L v1 | Test Top-1 đã lưu | Hồ sơ chi tiết |
|---|---|---:|---|
| Mid224 | Smoothing 0,10, LR 0,10, epoch 176 | 99,20% (248/250) | [l_mid224_smoothing010](experiment_midterm_smoothing_lr/models/l_mid224_smoothing010/README.md) |
| Mid32 | LR 0,15, epoch 186 | 97,20% (243/250) | [l_mid32_lr015](experiment_midterm_smoothing_lr/models/l_mid32_lr015/README.md) |

Trong mỗi hồ sơ, `test_metrics.json` chứa số liệu tổng thể; `classification_report.csv` chứa precision/recall/F1 từng lớp; `confusion_matrix.csv`/`.png` chứa ma trận; `learning_curves.png` chứa đường học; `predictions.csv` phục vụ phân tích ảnh sai.

Chạy [checkpoint224.py](../../checkpoint224.py) và [checkpoint32.py](../../checkpoint32.py) để đo lại trên ảnh test. Hai script dùng bản trọng số được ghim ở [checkpoints](../../checkpoints/README.md).

## Bảng tổng hợp ưu tiên

- [Test metrics của 10 run smoothing/LR](experiment_midterm_smoothing_lr/comparisons/test_metrics.csv).
- [Validation và best epoch](experiment_midterm_smoothing_lr/selection/validation_summary.csv).
- [Khóa lựa chọn theo validation](experiment_midterm_smoothing_lr/selection/selection_lock.json).
- [So sánh từng lớp](experiment_midterm_smoothing_lr/comparisons/class_metrics.csv).

Đi tiếp đến [so sánh và hạn chế](../06_discussion/README.md) trước khi viết kết luận về mô hình tốt nhất.
