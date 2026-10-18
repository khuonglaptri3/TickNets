# 04. Huấn luyện và tái lập trên Kaggle

Tái lập bằng bốn notebook dưới đây. Bật GPU và Internet trên Kaggle, mở notebook rồi Run All. Mỗi notebook đã nhúng ZIP mã nguồn gốc, cấu hình và launcher; notebook tự giải nén vào `/kaggle/working/` và kiểm tra SHA-256 trước khi chạy. Không cần các file `train_cifar*`, thư mục `scripts/` hay `configs/` ở root repo.

| Dataset | Optimizer | LR | Weight decay | Notebook chạy lại | Notebook đã chạy |
|---|---|---:|---:|---|---|
| CIFAR-10 | SGD + Nesterov | 0.175 | 0.0001 | [Phase14](notebooks/kaggle/Phase14_CIFAR10_TickNetL_LargeV2_14Blocks.ipynb) | [Output](notebooks/executed/phase-14-cifar-10-v-i-ticknet-l-large-v2-14-kh-i.ipynb) |
| CIFAR-10 | Adam | 0.001 | 0.0001 | [Phase17](notebooks/kaggle/Phase17_CIFAR10_TickNetL_LargeV2_14Blocks_Adam.ipynb) | [Output](notebooks/executed/ticknet-l-large-v2-14-kh-i-adam.ipynb) |
| CIFAR-100 | SGD + Nesterov | 0.175 | 0.0005 | [Phase13](notebooks/kaggle/Phase13_CIFAR100_TickNetL_LargeV2_CutMix_A10.ipynb) | [Output](notebooks/executed/phase13-cifar100-ticknetl-largev2-cutmix-a10.ipynb) |
| CIFAR-100 | Adam | 0.001 | 0.0005 | [Phase18](notebooks/kaggle/Phase18_CIFAR100_TickNetL_LargeV2_14Blocks_Adam.ipynb) | [Output](notebooks/executed/phase18-cifar100-ticknetl-largev2-14blocks-adam.ipynb) |

Thiết lập chung: kiến trúc 14 khối, seed 42, batch 128, 200 epochs, validation 10%, cross-entropy và cosine LR đến 0. SGD dùng momentum 0.9, Nesterov; Adam dùng β=(0.9,0.999), ε=1e-8. CIFAR-10 dùng crop/flip/Cutout 16; CIFAR-100 dùng crop/flip/RandAugment/CutMix.

Chọn epoch và cấu hình theo validation Top-1, hòa thì chọn loss thấp hơn. Test dùng để báo cáo sau lựa chọn. Mã nhúng được đối chiếu với `source_sha256` trong config của từng run. Notebook đã chạy được giữ nguyên; notebook Phase13 dùng để chạy lại được bổ sung nguồn nhúng từ snapshot đã xác thực, thay cho yêu cầu gắn ZIP ngoài.

Các biến `EPOCHS_PER_SESSION` và `RESUME_ROOT` trong cell đầu hỗ trợ chia phiên và tiếp tục train. Resume phải dùng đúng nguồn, môi trường và trạng thái đã lưu. Thay đổi GPU hoặc phiên bản thư viện có thể làm kết quả train khác; các số báo cáo là kết quả Kaggle đã lưu.

[Phân tích quá trình train và mô hình](TRAINING_AND_MODEL_REPORT.md) · [Kết quả bốn run](../05_results/README.md) · [Đánh giá hai checkpoint đã chọn](../../README.md).
