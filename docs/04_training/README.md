# 4. Thiết lập thực nghiệm

| Nội dung | Nguồn |
|---|---|
| Train/val/test, chọn checkpoint, resume và đánh giá | [MIDTERM_PROTOCOL.md](MIDTERM_PROTOCOL.md) |
| Khởi tạo từ đầu | [MODEL_INITIALIZATION.md](MODEL_INITIALIZATION.md) |
| Batch, shuffle, workers và pin memory | [DATALOADER.md](DATALOADER.md) |
| Cơ sở tuning và các đề xuất ban đầu | [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md) |
| Cách chạy từng hướng thí nghiệm | [BRANCH_GUIDE.md](BRANCH_GUIDE.md) |
| Notebook Kaggle | [notebooks](notebooks/README.md) |
| Trainer có validation | [train_mid_experiment.py](../../train_mid_experiment.py) |
| DataLoader có validation | [mid_experiment_data.py](../../models/mid_experiment_data.py) |

Các tài liệu thiết kế có thể ghi trạng thái ở thời điểm trước khi train; dùng config và log thực tế khi lập bảng thông số.

## Config và log cho các kết quả báo cáo

| Run | Config | Log 200 epoch |
|---|---|---|
| Mid224 smoothing 0,10 | [config](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_smoothing010/config.json) | [epochs.csv](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_smoothing010/epochs.csv) |
| Mid32 LR 0,15 | [config](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid32_lr015/config.json) | [epochs.csv](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid32_lr015/epochs.csv) |
| Mid224 baseline được chọn trong nhóm smoothing/LR theo validation | [config](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_baseline/config.json) | [epochs.csv](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_baseline/epochs.csv) |

Các run trên có 22.500 train, 2.500 validation, 250 test; SGD momentum 0,9, weight decay 1e-4, batch 64, cosine 200 epoch, seed 42 và split seed 123.

Preset ở [configs/midterm](../../configs/midterm) là đầu vào cho lượt chạy mới. Khi mô tả kết quả đã có, ưu tiên config lưu cùng checkpoint của lượt chạy đó.
