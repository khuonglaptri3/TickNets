**Bốn hướng thử nghiệm L v1 cho giữa kỳ**

Các nhánh cùng phát triển từ `feature/model-l-midterm-final`, commit `2bcbdb4105c5c8576829af568d6c31bb89a52c8f`:

| Nhánh | Thay đổi cần kiểm chứng |
|---|---|
| `experiment/lv1-midterm-mixup` | Mixup; giữ nguyên kiến trúc L v1 |
| `experiment/lv1-midterm-cutmix` | CutMix; giữ nguyên kiến trúc L v1 |
| `experiment/lv1-midterm-smoothing-lr` | Label smoothing 0,05/0,1 và LR 0,05/0,15 bằng các lượt riêng |
| `experiment/lv1-midterm-stage4-depth` | Stage 4 từ 2→3 block; recipe giữ như baseline |

Đọc thêm `docs/experiments/BRANCH_GUIDE.md` trên nhánh mình nhận. `configs/midterm/experiment.json` là cấu hình chính của nhánh; `baseline.json` là đối chứng chung. Không mặc định phối hợp nhiều kỹ thuật.

**Cài đặt và dữ liệu**

Dùng Python 3.10 trở lên, PyTorch/torchvision tương thích (profiler cần torch ≥2.5):

```bash
python -m pip install torch torchvision numpy pillow pytest scipy
```

Chuẩn bị `DATA_ROOT/Mid32/train|test/{bird,cat,dog,frog,horse}` và tương tự `Mid224`. Hai độ phân giải phải có cùng tên file/class/split. Ảnh giữ đúng kích thước gốc 32 hoặc 224. Nếu dùng local worktree, trỏ `--data-root` về thư mục `data` ở checkout chính. Không copy hoặc chia lại tập test.

Với dữ liệu chuẩn: mỗi lớp 5.000 ảnh train và 50 ảnh test. Trainer mới tách 500 ảnh/lớp làm validation: 22.500 train, 2.500 validation, 250 test. `split_seed=123` độc lập `seed=42`; validation membership được chọn theo class/filename, nhất quán ở hai độ phân giải. Hash trong `validation_split.json` xác nhận membership, không xác nhận byte ảnh. `source_manifest_sha256` ghi thêm manifest nguồn nếu có.

**Chạy đối chứng và ứng viên**

```bash
python train_mid_experiment.py --config configs/midterm/baseline.json --variant Mid32 --data-root data --output-dir runs/mid32_baseline_seed42
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --output-dir runs/mid32_experiment_seed42
```

Lặp lại với `--variant Mid224` và output khác. JSON cung cấp defaults; CLI ghi đè defaults. Cùng batch 64, SGD/momentum 0,9/WD 1e-4, cosine 200 epoch, augmentation crop/flip; đổi duy nhất kỹ thuật của nhánh. Có thể tăng workers cho Kaggle, nhưng phải đổi giống nhau trong cả baseline và ứng viên. So sánh dùng cùng split/seed; kết quả baseline mới có validation không so trực tiếp như ablation với checkpoint lịch sử train đủ 25.000 ảnh.

Mỗi run ghi `config.json` (bao gồm Git revision, môi trường, complexity), `validation_split.json`, `epochs.csv`, `last.pt`, `best_val.pt`. Chọn best theo validation Top-1, tie-break bằng loss thấp hơn. Validation dùng ảnh không augmentation và cross-entropy nhãn cứng. Train Top-1 với Mixup/CutMix là accuracy có trọng số lambda với hai nhãn; không so trực tiếp với train Top-1 nhãn cứng.

**Tiếp tục phiên Kaggle**

Định trước tổng 200 epoch, phiên đầu có thể dừng ở epoch 100:

```bash
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid224 --data-root data --output-dir runs/mid224_experiment_seed42 --stop-after-epoch 100
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid224 --data-root data --output-dir runs/mid224_experiment_seed42 --resume runs/mid224_experiment_seed42/last.pt
```

Khôi phục nguyên thư mục run từ artifact ZIP ở phiên mới, rồi chạy resume trong thư mục đó. Giữ nguyên recipe, tổng epoch và phiên bản môi trường. Lưu model/optimizer/scheduler và RNG Python/NumPy/torch/CUDA/DataLoader tại cuối epoch; không tuyên bố resume giữa batch. Không khởi động lại cosine. `best_val.pt` dùng để chọn mô hình, `last.pt` dùng để resume. Kiểm thử CPU đã đối chiếu run liên tục và chia phiên; khả năng tái lập GPU còn phụ thuộc môi trường/hardware.

**Đánh giá cuối và huấn luyện đủ dữ liệu**

Trainer phát triển không tự chạy test. Sau khi chốt cấu hình bằng validation:

```bash
python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/mid32_experiment_seed42/best_val.pt --output-dir runs/mid32_selected_test
```

Lệnh này xuất Top-1/loss/Macro F1, `test_predictions.csv`, `confusion_matrix.csv`; kiểm tra mapping/membership/validation/architecture trước khi nạp trọng số. Dữ liệu test đã được quan sát ở giữa kỳ nên phải ghi rõ đây vẫn là split lịch sử, không gọi là một test độc lập mới.

Để báo cáo theo quy mô 5.000 train/class của đề, có thể retrain toàn bộ 25.000 ảnh sau khi chọn recipe và số epoch bằng validation:

```bash
python train_mid_experiment.py --config configs/midterm/experiment.json --variant Mid32 --data-root data --full-train --epochs 200 --output-dir runs/mid32_final_full_seed42
python train_mid_experiment.py --data-root data --variant Mid32 --evaluate runs/mid32_final_full_seed42/last.pt --output-dir runs/mid32_final_full_test
```

`200` ở ví dụ phải được thay bằng số epoch đã chốt; `--full-train` không có validation/best checkpoint và vẫn không tự đánh giá test. Tuning trên validation và retrain toàn train là hai bước khác nhau, có thể cho kết quả khác nhau.

**Ghi nhận để nhóm so sánh**

Ghi branch, commit, dataset, seed, config, train/val/test count, validation Top-1/loss tại best epoch, epoch được chọn, params/GFLOPs, thời gian epoch và GPU. Xác nhận lựa chọn bằng seeds 42/43/44 nếu đủ tài nguyên, giữ split seed 123; báo mean±std và không chọn seed bằng test. FLOPs: 1 MAC=2 FLOPs, Conv2d/Linear, một ảnh eval; FLOPs không phải latency.

Chạy test trước khi train GPU: `python -m pytest tests -q`. Nếu hệ điều hành hạn chế thư mục tạm, dùng `--basetemp` trỏ tới một thư mục thử nghiệm mới trong workspace.

Chưa có kết quả train 200 epoch cho bốn nhánh này. Kiểm thử/smoke xác minh kỹ thuật hoạt động, không chứng minh tăng accuracy.
