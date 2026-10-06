# Final CIFAR run archive

Thư mục này chứa **toàn bộ output của 8 run thuộc 4 phase**, copy nguyên byte từ `runs/`:
`best_val.pt`, `last.pt`, `config.json`, `epochs.csv`, `progress.json`,
`completion.json`, `test_metrics.json`, `confusion_matrix.csv`, `test_predictions.csv`.
12 file summary/manifest của từng phase cũng được giữ nguyên ở đây. `runs/` gốc không bị thay đổi.

## Checkpoint được chọn

Chọn bằng validation Top-1, giải hòa bằng validation loss; không chọn theo test.

| Dataset | Run | Best epoch | Val Top-1 | Test Top-1 |
| --- | --- | ---: | ---: | ---: |
| CIFAR-10 | [cifar10_sgd_lr015](cifar10_sgd_lr015/) | 188 | 95.70% | 94.97% |
| CIFAR-100 | [cifar100_sgd_lr015](cifar100_sgd_lr015/) | 190 | 74.34% | 75.28% |

Dùng `best_val.pt` cho inference. `last.pt` là trạng thái cuối epoch 200,
bao gồm optimizer, scheduler, RNG, history và bản sao best checkpoint, không phải trọng số được chọn để báo cáo test.

Xem [toàn bộ so sánh và hình](../docs/experiments/README.md),
[chỉ mục lựa chọn](../docs/experiments/selected_checkpoints.json) và
[SHA-256 của 84 file gốc](../docs/experiments/archive_manifest.json).

## Kiểm tra và đánh giá

Đối chiếu artifact đã lưu, không train/evaluate lại:

```bash
python scripts/aggregate_grid_search.py --runs-dir checkpoints
```

SE đã được gộp vào `ticknet_l.py` và hàm seed đã chuyển module, nên hash mã nguồn
hiện tại khác snapshot Kaggle gốc. Không sửa metadata checkpoint để che khác biệt này.
Muốn chạy inference lại sau refactor, dùng lựa chọn tường minh và một thư mục **chưa tồn tại**:

```bash
python train_cifar.py --evaluate checkpoints/cifar10_sgd_lr015/best_val.pt --allow-eval-source-change --output-dir runs/reeval_cifar10 --data-root data
python train_cifar.py --evaluate checkpoints/cifar100_sgd_lr015/best_val.pt --allow-eval-source-change --output-dir runs/reeval_cifar100 --data-root data
```

Dataset/model được suy ra từ checkpoint. Trainer vẫn kiểm tra architecture/trainer revision,
state_dict và hash dữ liệu test; kết quả mới ghi cả hash nguồn train và nguồn evaluate.
Tùy chọn này **không cho phép resume** với mã nguồn khác. Muốn resume đúng trạng thái cũ,
dùng snapshot trong [notebook Kaggle đã chạy](../docs/experiments/notebooks/) và cùng runtime/GPU.
Chỉ load checkpoint đáng tin cậy: định dạng PyTorch này dùng pickle (`weights_only=False`).
