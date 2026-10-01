**TickNet-L v1 + Mixup — thí nghiệm giữa kỳ**

Nhánh `experiment/lv1-midterm-mixup` giữ nguyên L v1, 1.096.260 tham số; chỉ thay augmentation theo batch trong train.

Mixup trộn ảnh và nhãn theo lambda lấy từ Beta(alpha,alpha). Mốc alpha=0,2, áp dụng mọi batch; loss là lambda×CE(nhãn A)+(1−lambda)×CE(nhãn B).

Baseline chạy với `mixing=none`, ứng viên dùng `configs/midterm/experiment.json`. Validation/test không trộn ảnh hay nhãn.

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --output-dir runs/mid32_mixup_seed42
```

Đổi variant thành Mid224 và dùng tên output riêng. Khi dùng worktree local, `--data-root` trỏ tới thư mục data của checkout chính. JSON chỉ cung cấp defaults; CLI ghi đè được. Giữ cùng batch/workers/LR/epoch giữa đối chứng và ứng viên.

Sau mốc đầu, có thể thử riêng alpha 0,1 hoặc 0,4 bằng `--mixing-alpha`; không thêm smoothing hay đổi kiến trúc trong cùng lượt đối chứng đầu tiên.

Chọn cấu hình theo validation Top-1/loss; dùng `best_val.pt` để đánh giá cuối bằng `--evaluate`. Train Top-1 là metric có trọng số của hai nhãn, không so trực tiếp với accuracy train nhãn cứng. Giả thuyết: regularization giúp khái quát hóa tốt hơn; chưa có kết quả 200 epoch chứng minh mức tăng.

Xem [protocol chung](MIDTERM_PROTOCOL.md) để resume, huấn luyện đủ 25.000 ảnh và báo kết quả nhiều seed. Notebook `docs/kaggle/experiments/Kaggle_Midterm_Experiment.ipynb` chuẩn bị dữ liệu và chạy baseline/ứng viên trên nhánh này.
