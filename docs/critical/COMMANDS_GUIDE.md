# Sổ tay các Lệnh Thực thi Đồ án TickNets (Command-Line Reference Guide)

Tài liệu này tổng hợp toàn bộ các câu lệnh CLI đã được chuẩn hóa và kiểm thử thành công trong đồ án, phục vụ từ khâu kiểm định dữ liệu, đo lường FLOPs/Params, huấn luyện mô hình, tinh chỉnh siêu tham số (tuning) đến đánh giá checkpoint.

---

## 1. Kiểm tra Môi trường, Unit Tests & Kiểm tra Dữ liệu

### 1.1. Chạy toàn bộ Unit Tests kiểm tra tính toàn vẹn (PyTest)
Chạy bộ test bao gồm kiểm tra loader, split manifest, shape ảnh, và kiểm thử huấn luyện 1 epoch mẫu:
```bash
pytest -v
```

### 1.2. Kiểm tra nhanh DataLoader (`--check-data`)
Kiểm tra khả năng đọc dữ liệu thô, shape batch `(64, 3, H, W)` và bảng ánh xạ 5 lớp mà **không tốn thời gian train**:
```bash
# Kiểm tra bộ Mid32 (32x32)
python train_mid.py --data-root data --variant Mid32 --seed 42 --check-data

# Kiểm tra bộ Mid224 (224x224)
python train_mid.py --data-root data --variant Mid224 --seed 42 --check-data
```

---

## 2. Kiểm định & Làm sạch Dữ liệu (Data Cleaning & Quality Audit)

Module `cleaning/` cung cấp công cụ kiểm định 3 lớp (pHash deduplication, blur/exposure filter, và MobileNetV3 object verification):

```bash
# Quét kiểm tra nhanh 250 ảnh tập test
python -m cleaning.clean_dataset --variant Mid224 --split test --output-dir cleaning/reports

# Quét toàn diện toàn bộ 25.000 ảnh tập train (có GPU)
python -m cleaning.clean_dataset --variant Mid224 --split train --batch-size 64 --device auto --output-dir cleaning/reports

# Quét siêu nhanh trên CPU (Bỏ qua MobileNetV3, chỉ đo độ mờ Laplacian, tương phản và trùng lặp pHash)
python -m cleaning.clean_dataset --variant Mid224 --split train --skip-object-filter --output-dir cleaning/reports
```

---

## 3. Đo lường Hồ sơ Mô hình (Model Profiling - Params & FLOPs)

Đo lường chính xác số lượng tham số học được và chi phí FLOPs cho cả 3 kiến trúc (**Basic, L, C**) trên 2 độ phân giải:
```bash
python profile_mid.py --models basic l c --output docs/model_profiles.json
```
*(Kết quả ghi nhận: Basic: 1.06M / 0.988 GFLOPs; L: 1.10M / 0.796 GFLOPs; C: 5.16M / 0.821 GFLOPs).*

---

## 4. Huấn luyện Mô hình (Model Training Pipelines)

> [!NOTE]
> Mặc định lệnh huấn luyện chạy 200 epochs với Cosine Annealing LR. Kết quả mỗi run được tự động lưu vào thư mục `--output-dir` gồm: `config.json`, `epochs.csv`, `last.pt` (checkpoint) và `test_metrics.json`.

### 4.1. Huấn luyện Baseline: TickNet-Basic (Bản gốc)
```bash
# Trên Mid32 (32x32)
python train_mid.py --data-root data --variant Mid32 --model basic --seed 42 --output-dir runs/basic_mid32_seed42

# Trên Mid224 (224x224)
python train_mid.py --data-root data --variant Mid224 --model basic --seed 42 --output-dir runs/basic_mid224_seed42
```

### 4.2. Huấn luyện Ứng viên 1: TickNet-L v1 (Tối ưu FLOPs ~0.79G, Mixed DW 3x3/5x5)
```bash
# Trên Mid32 (32x32)
python train_mid.py --data-root data --variant Mid32 --model l --seed 42 --output-dir runs/l_mid32_seed42

# Trên Mid224 (224x224)
python train_mid.py --data-root data --variant Mid224 --model l --seed 42 --output-dir runs/l_mid224_seed42
```

### 4.3. Huấn luyện Ứng viên 2: TickNet-C v1 (Mở rộng 5.16M params, 9 blocks)
```bash
# Trên Mid32 (32x32)
python train_mid.py --data-root data --variant Mid32 --model c --seed 42 --output-dir runs/c_mid32_seed42

# Trên Mid224 (224x224)
python train_mid.py --data-root data --variant Mid224 --model c --seed 42 --output-dir runs/c_mid224_seed42
```

---

## 5. Tinh chỉnh Siêu tham số & Thí nghiệm Bóc tách (Ablation Studies)

### 5.1. Thử nghiệm nhanh số epoch nhỏ (Sanity Check / Fast Run)
```bash
# Thử nghiệm nhanh 5 epoch trên GPU
python train_mid.py --data-root data --variant Mid224 --model l --epochs 5 --batch-size 64 --output-dir runs/l_mid224_epochs5
```

### 5.2. Thử nghiệm quét Learning Rate & Batch Size
```bash
# Tốc độ học nhỏ hơn (lr=0.01)
python train_mid.py --data-root data --variant Mid224 --model l --learning-rate 0.01 --output-dir runs/l_mid224_lr001

# Batch size lớn hơn (batch-size=128)
python train_mid.py --data-root data --variant Mid224 --model l --batch-size 128 --learning-rate 0.1 --output-dir runs/l_mid224_bs128
```

### 5.3. Thí nghiệm bóc tách: Tắt Data Augmentation (`--no-augment`)
Đo lường mức độ đóng góp của `RandomCrop (reflect)` và `RandomHorizontalFlip`:
```bash
python train_mid.py --data-root data --variant Mid224 --model l --no-augment --output-dir runs/l_mid224_no_augment
```

---

## 6. Đánh giá lại Checkpoint đã lưu (Checkpoint Evaluation)

Khi đã có checkpoint `last.pt`, bạn có thể đánh giá lại độ chính xác trên tập test mà không cần huấn luyện lại:
```bash
python train_mid.py --data-root data --variant Mid224 --evaluate runs/l_mid224_seed42/last.pt
```
*(Lệnh này tự động kiểm tra đối chiếu mã SHA-256 của `split_manifest.csv` và `architecture_revision` trước khi load trọng số).*

---

## 7. Tái tạo Dataset từ Thư mục Gốc (Dataset Preparation)

Nếu cần tạo lại toàn bộ cấu trúc thư mục dữ liệu, manifest và file nén từ nguồn thô ban đầu:
```bash
python prepare_mid_dataset.py --source-root "<path_to_source_folder>" --output-root data_recreated --seed 42 --archive-format tar.xz
```
