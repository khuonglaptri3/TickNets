**TickNet-L v1 + CutMix — thí nghiệm giữa kỳ**

Nhánh `experiment/lv1-midterm-cutmix` giữ nguyên L v1, 1.096.260 tham số; chỉ thay augmentation theo batch trong train.

CutMix thay một vùng chữ nhật bằng patch của ảnh khác. Mốc alpha=1,0, áp dụng xác suất 0,5; lambda được tính lại theo diện tích patch sau khi cắt theo biên ảnh, không dùng nguyên lambda đã lấy mẫu.

Baseline chạy với `mixing=none`, ứng viên dùng `configs/midterm/experiment.json`. Validation/test không trộn ảnh hay nhãn.

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --output-dir runs/mid32_cutmix_seed42
```

Đổi variant thành Mid224 và dùng tên output riêng. Khi dùng worktree local, `--data-root` trỏ tới thư mục data của checkout chính. JSON chỉ cung cấp defaults; CLI ghi đè được. Giữ cùng batch/workers/LR/epoch giữa đối chứng và ứng viên.

Sau mốc đầu, có thể thử riêng xác suất 0,25 hoặc 1,0 bằng `--mixing-probability`; không thêm smoothing hay đổi kiến trúc trong cùng lượt đối chứng đầu tiên.

Chọn cấu hình theo validation Top-1/loss; dùng `best_val.pt` để đánh giá cuối bằng `--evaluate`. Train Top-1 là metric có trọng số của hai nhãn, không so trực tiếp với accuracy train nhãn cứng. Giả thuyết: regularization giúp khái quát hóa tốt hơn; chưa có kết quả 200 epoch chứng minh mức tăng.

Xem [protocol chung](MIDTERM_PROTOCOL.md) để resume, huấn luyện đủ 25.000 ảnh và báo kết quả nhiều seed. Notebook `docs/kaggle/experiments/Kaggle_Midterm_Experiment.ipynb` chuẩn bị dữ liệu và chạy baseline/ứng viên trên nhánh này.
