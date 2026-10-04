# Nhánh L: label smoothing và learning rate

Nhánh này dùng nguyên kiến trúc L (`ticknet-l-v1`, 5 lớp). Chỉ thay đổi loss huấn luyện hoặc learning rate ban đầu; validation và test luôn dùng CrossEntropyLoss với nhãn cứng, smoothing bằng 0. Tham số `--label-smoothing` có mặc định 0 và chỉ nhận số hữu hạn trong khoảng `0 <= value < 1`. Giá trị từ CLI ghi đè giá trị hợp lệ trong JSON.

## Các công thức thí nghiệm

Tất cả công thức dùng batch size 64, 200 epoch, SGD, momentum 0.9, weight decay 1e-4, seed huấn luyện 42, split seed 123, validation fraction 0.1, augmentation mặc định và không Mixup/CutMix. Scheduler là CosineAnnealingLR với `T_max=200`; LR trong bảng là LR ban đầu.

| Config | Label smoothing | LR ban đầu | So sánh với baseline |
| --- | ---: | ---: | --- |
| `configs/midterm/baseline.json` | 0 (mặc định CLI) | 0.1 | Mốc đối chứng |
| `configs/midterm/experiment.json` | 0.05 | 0.1 | Công thức mặc định của nhánh |
| `configs/midterm/smoothing_005.json` | 0.05 | 0.1 | Chỉ thay smoothing |
| `configs/midterm/smoothing_010.json` | 0.1 | 0.1 | Chỉ thay smoothing |
| `configs/midterm/lr_005.json` | 0 | 0.05 | Chỉ thay LR |
| `configs/midterm/lr_015.json` | 0 | 0.15 | Chỉ thay LR |

`experiment.json` và `smoothing_005.json` là cùng một công thức. Hai config LR đặt smoothing bằng 0 để đo riêng tác động của LR. Không dùng kết quả của một tổ hợp smoothing/LR để kết luận về tác động riêng của từng yếu tố.

## Chạy trên máy Windows

Mở PowerShell tại thư mục gốc của worktree này. Dữ liệu đã chuẩn bị cần có cấu trúc `Mid32/train`, `Mid32/test`, `Mid224/train`, `Mid224/test` với 5 lớp bird, cat, dog, frog, horse; giữ cùng manifest và danh sách mẫu cho mọi công thức. Sửa `$dataRoot` theo nơi lưu dữ liệu. Mỗi lần chạy cần một thư mục output mới; trainer tự chọn CUDA nếu có GPU.

```powershell
$pythonExe = 'python'
$dataRoot = 'data'
```

Các lệnh cho Mid32:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/baseline.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_baseline
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoothing005
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_010.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoothing010
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_005.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_lr005
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_015.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_lr015
```

Các lệnh cho Mid224:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/baseline.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_baseline
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_smoothing005
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_010.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_smoothing010
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_005.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_lr005
& $pythonExe train_mid_experiment.py --config configs/midterm/lr_015.json --data-root $dataRoot --variant Mid224 --output-dir runs/l_mid224_lr015
```

Dùng công thức mặc định của nhánh thay cho `smoothing_005.json`:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_experiment
```

Muốn chạy baseline bằng config mặc định của nhánh, thêm `--label-smoothing 0`. Nếu muốn khảo sát LR riêng, luôn dùng `lr_005.json` hoặc `lr_015.json`; không chỉ ghi đè LR trên `experiment.json` vì config này vẫn giữ smoothing 0.05.

## Chạy trên Kaggle

Trong thư mục chứa mã nguồn của nhánh, dùng `python` của môi trường Kaggle và thay đường dẫn dữ liệu/output. Ví dụ trong một ô notebook:

```python
!python train_mid_experiment.py --config configs/midterm/experiment.json --data-root /kaggle/input/midterm-prepared --variant Mid32 --output-dir /kaggle/working/l_mid32_smoothing005
```

Áp dụng lần lượt từng config trong bảng và đặt output khác nhau. Dùng cùng `--variant`, manifest, seed, split seed và tổng số epoch khi so sánh.

## Kiểm tra một epoch và tiếp tục

`--stop-after-epoch 1` dừng tại ranh giới epoch nhưng giữ lịch cosine 200 epoch, phù hợp kiểm tra trước khi chạy đủ. Ví dụ:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoke --stop-after-epoch 1
& $pythonExe train_mid_experiment.py --config configs/midterm/experiment.json --data-root $dataRoot --variant Mid32 --output-dir runs/l_mid32_smoke --resume runs/l_mid32_smoke/last.pt
```

Resume phải dùng `last.pt`, thư mục chạy cũ cùng `epochs.csv`, và giữ nguyên công thức, kể cả smoothing, tổng epoch, augmentation và split. Đổi smoothing từ 0.05 sang 0 hoặc 0.1 sẽ bị từ chối trước khi ghi kết quả. Checkpoint lưu cả model, optimizer, scheduler, RNG và generator của loader; cùng công thức cho phép tiếp tục chính xác tại ranh giới epoch. `--epochs 1` cũng chạy được một epoch nhưng tạo lịch cosine khác, chỉ nên dùng cho kiểm tra độc lập.

## Chọn mô hình và ghi nhận kết quả

Validation lấy riêng từ tập train bằng split seed 123; tập test không tham gia chọn công thức hoặc checkpoint. Kiểm tra `validation_sha256` và `membership_sha256` trong `config.json` để bảo đảm các lần chạy dùng cùng mẫu. Trainer chọn `best_val.pt` theo validation top-1 cao nhất; nếu top-1 bằng nhau thì chọn hard-label validation loss thấp hơn. `last.pt` dùng để resume. Không dùng `--full-train` trong vòng so sánh vì chế độ đó bỏ validation.

So sánh từng công thức với baseline riêng cho Mid32 và Mid224. Ghi validation top-1, validation loss, epoch được chọn, công thức, seed và hash split. Khi thay seed để kiểm tra độ ổn định, chạy lại cả baseline lẫn các công thức với cùng seed đó, vẫn giữ split seed 123. Không so sánh trực tiếp train loss giữa các mức smoothing vì hàm mục tiêu khác nhau; hard-label validation loss vẫn so sánh được.

Sau khi cố định công thức bằng validation, đánh giá test một lần bằng checkpoint đã chọn, ví dụ nếu smoothing 0.05 thắng trên validation:

```powershell
& $pythonExe train_mid_experiment.py --config configs/midterm/smoothing_005.json --data-root $dataRoot --variant Mid32 --evaluate runs/l_mid32_smoothing005/best_val.pt --output-dir runs/l_mid32_smoothing005_test
```

Evaluation dùng hard-label loss dù config có smoothing. Kết quả gồm `test_metrics.json`, `test_predictions.csv` và `confusion_matrix.csv`. Không quay lại điều chỉnh tham số hoặc chọn checkpoint theo test.

## Giả thuyết cần kiểm chứng

Smoothing 0.05 hoặc 0.1 có thể giảm dự đoán quá tự tin và cải thiện validation; mức quá lớn cũng có thể làm giảm khả năng phân biệt lớp. Với 5 lớp, phân phối mục tiêu là `(1 - epsilon) * one_hot + epsilon / 5`, nên lớp đúng nhận `1 - epsilon + epsilon / 5` và mỗi lớp còn lại nhận `epsilon / 5`.

LR 0.05 có thể giúp cập nhật ổn định hơn nhưng hội tụ chậm; LR 0.15 có thể tiến nhanh hơn nhưng dao động mạnh hơn ở đầu lịch. Đây là giả thuyết, chưa phải kết luận về độ chính xác. Các kiểm tra một epoch chỉ xác nhận pipeline, loss, checkpoint và resume; cần hoàn tất các lần chạy 200 epoch để đánh giá giả thuyết.

## Kiểm thử

```powershell
$env:TORCH_HOME = 'runs/torch-cache'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $pythonExe -m pytest tests/test_mid_smoothing.py -p no:cacheprovider
```

Các test kiểm tra loss/gradient với công thức tính tay và PyTorch, mặc định baseline bằng 0, miền giá trị CLI/JSON, CLI ghi đè config, một epoch thực cho cả 5 config mới trên dữ liệu 5 lớp, hard-label validation/test loss, từ chối resume khi đổi smoothing và resume khớp chính xác khi giữ nguyên công thức.

Notebook `docs/kaggle/experiments/Kaggle_Midterm_Experiment.ipynb` ch?y baseline v? smoothing 0,05; th?m c?c config LR/smoothing kh?c v?o CONFIGS ?? kh?o s?t ti?p. Xem [protocol chung](MIDTERM_PROTOCOL.md) cho quy tr?nh ??y ??.
