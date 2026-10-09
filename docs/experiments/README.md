# Kết quả final — TickNet-L trên CIFAR-10 và CIFAR-100

Đã tổng hợp **4 phase / 8 run**, mỗi run 200 epochs. Các số liệu dưới đây được
đối chiếu từ output Kaggle trong `runs/`, không phải kết quả train mới trong lần dọn repo.
Bản sao đầy đủ nằm tại [checkpoints](../../checkpoints/README.md).

## Kết quả và lựa chọn

| Phase | Dataset | Optimizer | LR | Best epoch | Val Top-1 (%) | Test Top-1 (%) | Test loss | Macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | CIFAR-10 | SGD | 0.10 | 193 | 95.56 | 94.52 | 0.226801 | 0.945148 |
| 1 | CIFAR-10 | **SGD** | **0.15** | **188** | **95.70** | **94.97** | **0.197835** | **0.949716** |
| 2 | CIFAR-10 | Adam | 0.001 | 191 | 94.58 | 93.38 | 0.259598 | 0.933801 |
| 2 | CIFAR-10 | Adam | 0.0003 | 152 | 91.52 | 89.73 | 0.358073 | 0.897082 |
| 3 | CIFAR-100 | SGD | 0.10 | 186 | 73.72 | 74.07 | 1.132403 | 0.739847 |
| 3 | CIFAR-100 | **SGD** | **0.15** | **190** | **74.34** | **75.28** | **1.080919** | **0.751867** |
| 4 | CIFAR-100 | Adam | 0.001 | 177 | 71.02 | 70.79 | 1.303312 | 0.707075 |
| 4 | CIFAR-100 | Adam | 0.0003 | 164 | 63.46 | 63.81 | 1.652653 | 0.638118 |

**Chọn SGD, learning rate 0.15 cho cả hai dataset.** Cả hai đều dẫn đầu validation;
test chỉ dùng để báo cáo sau lựa chọn. CIFAR-10 dùng `cifar10_sgd_lr015/best_val.pt`
(epoch 188); CIFAR-100 dùng `cifar100_sgd_lr015/best_val.pt` (epoch 190).

- CIFAR-10: SGD 0.15 cao hơn SGD 0.10 **0.14 điểm % validation / 0.45 điểm % test**;
  cao hơn Adam 0.001 **1.12 / 1.59 điểm %**.
- CIFAR-100: SGD 0.15 cao hơn SGD 0.10 **0.62 / 1.21 điểm %**;
  cao hơn Adam 0.001 **3.32 / 4.49 điểm %**.
- Adam 0.0003 thấp nhất trong grid ở cả hai dataset. Đây là kết luận cho recipe,
  seed và ngân sách đang xét, không chứng minh Adam nói chung kém SGD.
- Chỉ có **một seed (42)** mỗi cấu hình: chưa đủ để khẳng định chênh lệch có ý nghĩa
  qua nhiều lần train. Không dùng kết quả midterm để so sánh với final.
- Chưa có run TickNet-Basic final trong bộ này; chưa thể kết luận L hơn Basic về accuracy CIFAR.

![CIFAR-10 optimizer comparison](report_assets/cifar10_optimizer_comparison.png)
![CIFAR-100 optimizer comparison](report_assets/cifar100_optimizer_comparison.png)

## Bộ figure và CSV

[report_assets/](report_assets/) dùng phong cách bộ báo cáo Basic cũ: learning curves
hai ô Loss/Accuracy, confusion matrix xanh, bảng classification report và bảng tổng hợp.

Mỗi run có:

- `<run>_learning_curves.png`: train/validation loss và Top-1, đánh dấu epoch được chọn.
- `<run>_epochs.csv`: log 200 epochs giữ nguyên byte từ run.
- `<run>_confusion_matrix.png` và `.csv`: hình trực quan và ma trận đếm gốc.
- `<run>_classification_report.csv`: precision, recall, F1, support từng lớp;
  accuracy, macro average và weighted average. Tỷ lệ không xác định được đặt bằng 0.

CIFAR-10 dùng tên 10 lớp và hiển thị số đếm. CIFAR-100 dùng **fine-label ID 0–99**
(`class_000`…`class_099`), không suy đoán tên lớp từ output không có metadata tên.
Hình CIFAR-100 chuẩn hóa theo hàng thành %, CSV vẫn là số đếm.
[class_mapping.csv](report_assets/class_mapping.csv) ghi thứ tự hàng/cột.

[final_required_summary.csv](report_assets/final_required_summary.csv) chứa đủ 8 run,
phase, best validation loss, số tham số/FLOPs và cờ `Selected`.
[grid_search_summary.csv](grid_search_summary.csv) giữ precision số đầy đủ.

## Protocol và tài nguyên

- Official train 50.000 ảnh → phân tầng 45.000 train / 5.000 validation; test 10.000 ảnh.
- 200 epochs, batch size 128, seed 42, weight decay 0.0001, CosineAnnealingLR đến 0.
- SGD: momentum 0.9, Nesterov. Adam: betas 0.9/0.999, eps 1e-8.
- Train có crop, flip, Cutout 16; validation/test không augmentation.
- Mỗi run chọn epoch bằng validation Top-1, hòa thì validation loss thấp hơn;
  nếu vẫn hòa giữ epoch sớm hơn. Chọn run cùng dataset theo cùng tiêu chí validation.
- Runtime được ghi trong config: Tesla T4, torch 2.11.0+cu128, torchvision 0.26.0+cu128;
  FP32, deterministic algorithms, không TF32.
- CIFAR-10: **1.100.105 parameters / 157.828.544 FLOPs**.
- CIFAR-100: **1.169.315 parameters / 157.966.784 FLOPs**.
- FLOPs tính Conv2d/Linear, batch 1, `1 MAC = 2 FLOPs`; không bao gồm BN,
  activation, pooling, residual add, bias add, SE scaling và di chuyển dữ liệu.
  Cả hai đạt giới hạn ≤6M parameters và <1G FLOPs theo quy ước này.

Train loss/accuracy có augmentation và model ở train mode, nên không so trực tiếp
với validation để kết luận bất thường chỉ vì validation tốt hơn train.

## Bảo toàn bằng chứng

- [notebooks/](notebooks/) giữ nguyên bốn notebook đã chạy, bao gồm output và snapshot nguồn cũ.
- [checkpoints/](../../checkpoints/) chứa **84 file gốc**: 9 file × 8 run và 12 file phase summary/manifest.
- [archive_manifest.json](archive_manifest.json) ghi SHA-256 toàn bộ 84 file đã copy;
  mỗi run giữ nguyên `completion.json` và hash mã nguồn/data trong `config.json`.
- [selected_checkpoints.json](selected_checkpoints.json) ghi đường dẫn, epoch, metric và hash best checkpoint.
- Validator đối chiếu checksum, 200 dòng epoch, 45.000/5.000 mẫu mỗi epoch,
  10.000 dự đoán test, ma trận nhầm lẫn, Top-1, Macro-F1 và trung bình NLL.
  Đây là xác thực tính nhất quán output; lần cleanup này không chạy lại full inference CIFAR.

## Tái tạo

```bash
pip install torch torchvision numpy pillow pandas scipy matplotlib pytest
# Từ bản lưu đã có (không train lại):
python scripts/build_final_report.py --runs-dir checkpoints
# Hoặc copy toàn bộ output gốc rồi tạo báo cáo; file khác nội dung sẽ bị từ chối ghi đè:
python scripts/build_final_report.py --runs-dir runs --archive-dir checkpoints
```

Hướng dẫn evaluate checkpoint sau refactor: [checkpoints/README.md](../../checkpoints/README.md).
Notebook mới để train từ đầu: [docs/kaggle](../kaggle/README.md).
