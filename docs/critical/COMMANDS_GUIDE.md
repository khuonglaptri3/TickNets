# Sổ tay các Lệnh Thực thi Đồ án TickNets - Cuối kỳ (Command-Line Reference Guide)

Tài liệu này tổng hợp toàn bộ các câu lệnh CLI đã được chuẩn hóa và kiểm thử thành công trong đồ án cuối kỳ, phục vụ từ khâu tải và kiểm định dữ liệu CIFAR-10/100, đo lường FLOPs/Params, huấn luyện mô hình TickNet-L & Baseline, khảo sát lưới siêu tham số (Grid Search), đến đánh giá và gộp báo cáo.

---

## 1. Kiểm tra Môi trường & Bộ Kiểm thử Tự động (PyTest)

Chạy toàn bộ 47 unit tests kiểm tra tính toàn vẹn của mô hình `TickNet-L`, mô hình tác giả `TickNet-Basic`, DataLoaders, Data Augmentation Cutout, Nesterov Momentum và pipeline huấn luyện:
```bash
PYTHONPATH=. pytest -v
```
*(Yêu cầu: 100% tests PASSED trước khi triển khai huấn luyện).*

---

## 2. Tải Tự động & Xác thực Toàn vẹn Dữ liệu CIFAR (Automated Downloader)

Hệ thống tải trực tiếp từ server Đại học Toronto (`cave.cs.toronto.edu`) kèm fallback mirror (`www.cs.toronto.edu`) và cơ chế chống lỗi SSL trên Linux:

```bash
# Tải cả 2 bộ dữ liệu CIFAR-10 và CIFAR-100 vào thư mục data/
python download_cifar.py --data-root data --dataset all

# Tải riêng từng bộ dữ liệu:
python download_cifar.py --data-root data --dataset cifar10
python download_cifar.py --data-root data --dataset cifar100

# Bắt buộc tải lại (bỏ qua cache) và kiểm tra mã băm MD5
python download_cifar.py --data-root data --force
```

---

## 3. Đo lường Hồ sơ Mô hình (Model Profiling - Params & FLOPs)

Kiểm tra ràng buộc của đề bài (Tham số $\le 6.000.000$, FLOPs $< 1.000.000.000$):

```bash
# Kiểm tra nhanh độ phức tạp mô hình TickNet-L trên CIFAR-10 và CIFAR-100
python -c '
from models.ticknet_l import build_ticknet_l
from models.model_profile import profile_model
print("TickNet-L CIFAR-10:", profile_model(build_ticknet_l(10, cifar=True), 32))
print("TickNet-L CIFAR-100:", profile_model(build_ticknet_l(100, cifar=True), 32))
'

# Kiểm tra mô hình gốc tác giả TickNet-Basic trên CIFAR-10 và CIFAR-100
python -c '
from models.TickNet import build_TickNet
from models.model_profile import profile_model
print("TickNet-Basic CIFAR-10:", profile_model(build_TickNet(10, typesize="basic", cifar=True), 32))
print("TickNet-Basic CIFAR-100:", profile_model(build_TickNet(100, typesize="basic", cifar=True), 32))
'
```

*Số liệu kiểm chứng:*
- **TickNet-L (CIFAR-10):** 1.100.105 tham số ($\le 6M$) | 0.1578 GFLOPs ($< 1G$).
- **TickNet-L (CIFAR-100):** 1.169.315 tham số ($\le 6M$) | 0.1580 GFLOPs ($< 1G$).
- **TickNet-Basic (CIFAR-10):** 1.067.348 tham số ($\le 6M$) | 0.1584 GFLOPs ($< 1G$).
- **TickNet-Basic (CIFAR-100):** 1.159.598 tham số ($\le 6M$) | 0.1586 GFLOPs ($< 1G$).

---

## 4. Huấn luyện Mô hình Cuối kỳ (`train_cifar.py`)

Huấn luyện chạy 200 epochs với Cosine Annealing LR về 0, hỗ trợ cả `sgd` (kèm Nesterov) và `adam`, tăng cường dữ liệu `Cutout` $16 \times 16$:

### 4.1. Huấn luyện qua file cấu hình JSON (`configs/final/`)
```bash
# CIFAR-10 với SGD lr=0.10 (Nesterov + Cutout)
python train_cifar.py --config configs/final/cifar10_sgd_lr010.json --data-root data --output-dir runs/cifar10_sgd_lr010

# CIFAR-10 với Adam lr=0.001 (Cutout)
python train_cifar.py --config configs/final/cifar10_adam_lr0001.json --data-root data --output-dir runs/cifar10_adam_lr0001

# CIFAR-100 với SGD lr=0.10 (Nesterov + Cutout)
python train_cifar.py --config configs/final/cifar100_sgd_lr010.json --data-root data --output-dir runs/cifar100_sgd_lr010

# Baseline Tác giả TickNet-Basic trên CIFAR-10
python train_cifar.py --config configs/final/baseline_cifar10_sgd_lr010.json --data-root data --output-dir runs/baseline_cifar10_sgd_lr010
```

### 4.2. Huấn luyện trực tiếp qua CLI Flags
```bash
# Chạy TickNet-L trên CIFAR-100 với SGD lr=0.15
python train_cifar.py --model l --dataset cifar100 --optimizer sgd --learning-rate 0.15 --momentum 0.9 --nesterov --cutout --cutout-length 16 --epochs 200 --batch-size 128 --output-dir runs/cifar100_sgd_lr015

# Chạy TickNet-Basic của tác giả trên CIFAR-100
python train_cifar.py --model basic --dataset cifar100 --optimizer sgd --learning-rate 0.10 --momentum 0.9 --nesterov --cutout --cutout-length 16 --epochs 200 --batch-size 128 --output-dir runs/baseline_cifar100_sgd_lr010
```

### 4.3. Phục hồi huấn luyện khi bị ngắt quãng (`--resume`)
```bash
python train_cifar.py --config configs/final/cifar10_sgd_lr010.json --resume runs/cifar10_sgd_lr010/last.pt --output-dir runs/cifar10_sgd_lr010
```

### 4.4. Đánh giá Checkpoint độc lập trên tập Test (`--evaluate`)
```bash
python train_cifar.py --dataset cifar10 --evaluate runs/cifar10_sgd_lr010/best_val.pt --output-dir runs/cifar10_sgd_lr010/eval_test
```

---

## 5. Thực thi Thực nghiệm trên Kaggle GPU (Tesla T4 / P100)

Đồ án đã được module hóa thành 5 Notebook độc lập trong [`docs/kaggle/`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle):

1. **Baseline Tác giả:** Upload [`docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Kaggle_Author_TickNet_Baseline.ipynb) $\to$ Chạy 2 thực nghiệm trên CIFAR-10 & CIFAR-100.
2. **Phase 1 (CIFAR-10 SGD):** Upload [`docs/kaggle/Phase1_CIFAR10_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase1_CIFAR10_SGD.ipynb) $\to$ Chạy SGD lr=0.10 & lr=0.15.
3. **Phase 2 (CIFAR-10 Adam):** Upload [`docs/kaggle/Phase2_CIFAR10_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase2_CIFAR10_Adam.ipynb) $\to$ Chạy Adam lr=0.001 & lr=0.0003.
4. **Phase 3 (CIFAR-100 SGD):** Upload [`docs/kaggle/Phase3_CIFAR100_SGD.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase3_CIFAR100_SGD.ipynb) $\to$ Chạy SGD lr=0.10 & lr=0.15.
5. **Phase 4 (CIFAR-100 Adam):** Upload [`docs/kaggle/Phase4_CIFAR100_Adam.ipynb`](file:///home/intern-tdkhuong/Desktop/TickNets/docs/kaggle/Phase4_CIFAR100_Adam.ipynb) $\to$ Chạy Adam lr=0.001 & lr=0.0003.

*(Mỗi notebook tự động tạo file `.zip` kết quả ở `/kaggle/working/` để tải về).*

---

## 6. Tổng hợp Kết Quả & Xuất Báo cáo So Sánh (Master Report Aggregation)

Sau khi giải nén các thư mục thực nghiệm vào thư mục `runs/`, chạy script tổng hợp:
```bash
python scripts/aggregate_grid_search.py --runs-dir runs --output-csv docs/results/grid_search_summary.csv --output-md docs/results/grid_search_summary.md
```
Script sẽ tự động:
- Đọc tất cả 8 folder grid search + 2 folder baseline.
- Tìm điểm `Best Val Epoch`, `Best Val Acc`, `Test Top-1 Accuracy`, `Test Loss`, `Macro F1`.
- Xuất bảng Markdown và CSV để đưa thẳng vào báo cáo cuối kỳ.
