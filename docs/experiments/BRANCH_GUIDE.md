# Thí nghiệm tăng độ sâu Stage 4 của TickNet-L

Nhánh `experiment/lv1-midterm-stage4-depth` xuất phát từ mã dùng chung
`2256a1d3`. Mục tiêu là đo tác động của một block bổ sung tại Stage 4 khi giữ
cùng công thức huấn luyện. Đây là giả thuyết kiến trúc chưa có kết quả accuracy
trên bộ dữ liệu thật; chạy smoke test không chứng minh chất lượng phân loại.

## Baseline và biến thể

Baseline dùng `configs/midterm/baseline.json`, model `l`, revision
`ticknet-l-v1`. Biến thể dùng `configs/midterm/experiment.json`, model
`l_stage4`, revision `ticknet-l-v1-stage4-depth3`. Hai config chỉ khác `model`.
Model riêng kế thừa L v1 và thêm `backbone.stage4.unit3`: CompressedPDPBlock
288 → 216 → 288 kênh, depthwise chia đều kernel 3/5, stride 1, shortcut identity.
Độ sâu các stage đổi từ `(1, 1, 2, 2, 1)` thành `(1, 1, 2, 3, 1)`.
Block mới dùng cùng khởi tạo Conv2d như L v1; các trọng số đã khởi tạo của
model cha được giữ nguyên. API là `build_ticknet_l_stage4(num_classes=5, *, cifar=False)`.

| Model | Số block | Tham số học được (5 lớp) | FLOPs Mid32 | FLOPs Mid224 |
| --- | ---: | ---: | ---: | ---: |
| L v1 | 7 | 1.096.260 | 157.820.864 | 796.760.240 |
| L Stage4-depth3 | 8 | 1.236.030 | 174.236.864 | 846.991.472 |
| Phần tăng | 1 | 139.770 | 16.416.000 | 50.231.232 |

Ngân sách bài thi: ≤ 6.000.000 tham số và **< 1.000.000.000 FLOPs**.
Cả hai kích thước đạt ngân sách này. Mid224 của biến thể vượt mục tiêu thiết kế
800 triệu FLOPs của L v1, nên cần báo rõ chi phí tăng khi so sánh.
Quy ước: batch 1, eval forward, 1 MAC = 2 FLOPs, chỉ Conv2d và Linear
(gồm Linear trong SE); không tính BN, activation, pooling, cộng residual,
nhân SE, cộng bias hay di chuyển dữ liệu. Bộ đếm theo module đã được đối chiếu
với `torch.utils.flop_counter.FlopCounterMode`. FLOPs không phải độ trễ thiết bị.

Giả thuyết: thêm một block residual ở Stage 4 có thể cải thiện biểu diễn đặc trưng
với chi phí tính toán tăng vừa phải. Thí nghiệm phải đo validation Top-1/loss,
thời gian huấn luyện và suy luận thực tế; cũng có thể gặp overfitting hoặc không
cải thiện. Không kết luận accuracy tăng chỉ từ số block hay ngân sách.

## Chạy so sánh có kiểm soát

Chạy từ thư mục gốc của worktree, với dữ liệu ImageFolder đã chuẩn bị tại
`data/Mid32` và `data/Mid224` cùng `split_manifest.csv`.
Ví dụ PowerShell trên máy hiện tại:

```powershell
$python = 'python'
$env:TORCH_HOME = 'runs/torch-cache'
& $python train_mid_experiment.py --config configs/midterm/baseline.json --data-root data --variant Mid32 --output-dir runs/stage4/baseline-Mid32
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid32 --output-dir runs/stage4/depth3-Mid32
& $python train_mid_experiment.py --config configs/midterm/baseline.json --data-root data --variant Mid224 --output-dir runs/stage4/baseline-Mid224
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid224 --output-dir runs/stage4/depth3-Mid224
```

Trong Kaggle/Linux, dùng `python` thay đường dẫn `$python` và truyền
`--data-root /kaggle/input/<dataset>` cùng thư mục output dưới `/kaggle/working`.
Mỗi lần chạy mới cần thư mục output mới. Config tự chọn model, không cần
`--model` override. CLI có thể ghi đè các giá trị khác để kiểm tra nhanh:

```powershell
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid32 --output-dir runs/stage4/smoke-Mid32 --epochs 1 --batch-size 5 --device cpu --threads 2
```

Baseline và biến thể đều dùng seed huấn luyện 42, split seed 123,
validation phân tầng 10%, batch 64, 200 epoch, SGD LR 0,1/momentum 0,9/
weight decay 1e-4 và CosineAnnealingLR. Không dùng mixing hoặc label smoothing.
Cùng class/filename vào validation ở Mid32 và Mid224; validation không augmentation.
Với 25.000 ảnh train, mặc định dùng 22.500 train và 2.500 validation.
Đối chiếu `validation_sha256` và `membership_sha256` trong `config.json`
giữa các lần chạy để xác nhận cùng dữ liệu. Muốn so sánh nhiều seed, đổi
`--seed` giống nhau cho hai model và giữ `--split-seed 123`.

## Checkpoint và đánh giá

`epochs.csv` chỉ ghi train/validation; `best_val.pt` chọn theo validation Top-1,
phá hòa bằng loss thấp hơn. `last.pt` lưu trạng thái optimizer/scheduler/RNG
để tiếp tục đúng lịch tại ranh giới epoch:

```powershell
& $python train_mid_experiment.py --config configs/midterm/experiment.json --data-root data --variant Mid32 --output-dir runs/stage4/depth3-Mid32 --resume runs/stage4/depth3-Mid32/last.pt
```

Nếu muốn chạy từng phần, dùng `--stop-after-epoch` với tổng `--epochs 200`
ngay từ đầu; không đổi tổng epoch khi resume. Giữ nguyên công thức và dữ liệu.
Revision thiếu hoặc khác bị từ chối trước khi tải trọng số và tạo output;
checkpoint L v1 không dùng cho Stage4-depth3.

Chỉ mở test sau khi cố định cấu hình dựa trên validation:

```powershell
& $python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/stage4/baseline-Mid32/best_val.pt --output-dir runs/stage4/test-baseline-Mid32
& $python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/stage4/depth3-Mid32/best_val.pt --output-dir runs/stage4/test-depth3-Mid32
```

Model/revision được lấy từ checkpoint khi evaluate. Thay Mid32 bằng Mid224
và đường dẫn tương ứng để đánh giá độ phân giải lớn. Đầu ra gồm
`test_metrics.json` (Top-1, loss, macro-F1), `test_predictions.csv` và
`confusion_matrix.csv`. Không dùng test để chọn model, seed, epoch hay LR.
Sau khi khóa thiết lập, có thể dùng `--full-train` với output mới để học đủ
25.000 ảnh; chế độ này chỉ có `last.pt`, không chọn checkpoint bằng validation.

## Kiểm chứng và bàn giao

```powershell
& $python -m pytest -p no:cacheprovider tests/test_ticknet_l_stage4.py -q
```

Dùng tên `--basetemp` riêng cho mỗi lượt chạy song song. Test kiểm tra L gốc,
8 block của biến thể, khởi tạo block mới, số tham số/FLOPs với đối chiếu độc lập,
gradient hữu hạn ở 32/224, từ chối revision sai khi resume/evaluate trước output,
và CLI chạy một epoch rồi đánh giá tường minh trên ảnh fixture ở cả hai kích thước.
Các chỉ số trên fixture chỉ kiểm chứng đường chạy, không dùng báo cáo accuracy.

Phạm vi bàn giao chỉ gồm `models/ticknet_l_stage4.py`, đăng ký trong
`train_mid_experiment.py`, `configs/midterm/experiment.json`,
`tests/test_ticknet_l_stage4.py` và guide này. Không commit/push tại worker;
parent tích hợp kết quả.
Notebook Kaggle ghim commit v? ch?y baseline/?ng vi?n n?m t?i `docs/kaggle/experiments/Kaggle_Midterm_Experiment.ipynb`. Xem [protocol chung](MIDTERM_PROTOCOL.md) ?? d?ng d? li?u ? checkout ch?nh, resume v? retrain ??y ??.
