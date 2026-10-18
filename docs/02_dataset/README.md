# 02. Dữ liệu và preprocessing

| Dataset | Lớp | Train gốc | Train thực tế | Validation | Test |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 10 | 50.000 | 45.000 | 5.000 | 10.000 |
| CIFAR-100 | 100 fine labels | 50.000 | 45.000 | 5.000 | 10.000 |

Ảnh RGB 32×32; chia train/validation phân tầng, `val_fraction=0.1`, seed 42. Test chính thức tách biệt. SHA-256 pixel, nhãn và thứ tự mẫu được ghi tại `data_evidence` của từng `config.json` trong [bốn run](../05_results/checkpoints_4_phase/).

Validation/test chỉ chuyển tensor và chuẩn hóa:

| Dataset | Mean (R,G,B) | Std (R,G,B) |
|---|---|---|
| CIFAR-10 | (0.4914,0.4822,0.4465) | (0.2470,0.2435,0.2616) |
| CIFAR-100 | (0.5071,0.4867,0.4408) | (0.2675,0.2565,0.2761) |

Train CIFAR-10 dùng random crop (reflect padding 4), horizontal flip, normalization và Cutout 16. CIFAR-100 dùng crop/flip, RandAugment (2 phép, magnitude 7), normalization và CutMix (α=1, xác suất 0.5). Không áp dụng augmentation train lên test.

Hai script checkpoint tự tải dữ liệu và kiểm tra fingerprint test trước inference. Dữ liệu ảnh bỏ qua bởi Git. [Hướng dẫn dữ liệu](../../data/README.md) · [Tiền xử lý test](../../models/checkpoint_utils.py).
