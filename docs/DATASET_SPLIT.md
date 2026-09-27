# Mid32 / Mid224 — mô tả dữ liệu và cách chia

## 1. Mục tiêu và nguồn dữ liệu

Dataset phân loại 5 lớp: bird, cat, dog, frog, horse. Hai phiên bản có cùng
25,250 mã mẫu; mỗi mã mẫu có một ảnh 32x32 và một ảnh 224x224.
Nguồn khi tạo: `C:\Users\lanph\Downloads\Final_Dataset_`.
Đầu ra: `C:\Users\lanph\OneDrive\Desktop\Deep Learning\TickNets\data`.

Ảnh được sao chép nguyên byte JPEG, không resize hoặc nén lại. Tất cả ảnh đã
được giải mã để kiểm tra JPEG, RGB, kích thước và kiểm tra SHA-256 sau sao chép.

## 2. Phương pháp chia: stratified random hold-out theo lớp

- **Seed: 42**; chia lại toàn bộ 5 lớp, gộp các thư mục train/test cũ
  trước khi chọn mẫu. Split cũ chỉ được giữ trong manifest để truy vết.
- Mỗi lớp chọn ngẫu nhiên không hoàn lại đúng **50 ảnh test**; các ảnh
  còn lại là **5000 ảnh train**. Tỷ lệ là 99.009901% train
  và 0.990099% test, xuất phát từ số lượng đề bài quy định.
- Duyệt lớp theo thứ tự `bird, cat, dog, frog, horse` và sắp xếp mã mẫu trong
  từng lớp theo thứ tự chuỗi Python trước khi lấy mẫu.
- Khởi tạo **một** `random.Random(42)`; gọi `rng.sample(sorted_ids,
  50)` lần lượt cho từng lớp. Không khởi tạo lại RNG giữa các lớp.
- Mã mẫu là `class_name/filename`; tên giống nhau ở hai độ phân giải là một cặp.
  Chọn split một lần rồi áp dụng cho cả hai phiên bản. Không chia độc lập từng bản.
- Đây là cách chia phân tầng theo nhãn với số mẫu test cố định mỗi lớp, không
  phải lời khẳng định PyTorch quy định một tỷ lệ chia train/test bắt buộc.

## 3. Số lượng trong MỖI phiên bản

| Lớp | Class index | Train | Test | Tổng |
|---|---:|---:|---:|---:|
| bird | 0 | 5,000 | 50 | 5,050 |
| cat | 1 | 5,000 | 50 | 5,050 |
| dog | 2 | 5,000 | 50 | 5,050 |
| frog | 3 | 5,000 | 50 | 5,050 |
| horse | 4 | 5,000 | 50 | 5,050 |
| **Tổng** | | **25,000** | **250** | **25,250** |

Không tạo validation trong bản đóng gói. Nếu cần tuning, tách validation từ
train và giữ test độc lập; không dùng test để chọn seed, siêu tham số hay checkpoint.

## 4. Cấu trúc tương thích torchvision.datasets.ImageFolder

```text
data/
  Mid32/
    train/bird/*.jpeg    (tương tự cat, dog, frog, horse)
    test/bird/*.jpeg
  Mid224/
    train/bird/*.jpeg
    test/bird/*.jpeg
  split_manifest.csv
  split_config.json
  DATASET_SPLIT.md
  Mid32.tar.xz
  Mid224.tar.xz
```

Truyền `data/Mid32/train` hoặc `data/Mid224/train` vào ImageFolder khi huấn luyện;
truyền thư mục test tương ứng khi đánh giá. Ánh xạ lớp giống nhau cho mọi split.

## 5. Manifest, kiểm tra và tái lập

`split_manifest.csv` có một dòng cho mỗi cặp ảnh: mã mẫu, nhãn, chỉ số nhãn,
split mới, split cũ, đường dẫn nguồn/đích, SHA-256 tệp, SHA-256 pixel và số byte
của từng phiên bản. **Manifest là danh sách phân chia chính thức** để dùng lại.

- Cặp ảnh Mid32/Mid224 có cùng split và tên lớp: đã kiểm tra.
- Train/test không giao nhau theo mã mẫu: đã kiểm tra.
- Ảnh trùng pixel chính xác nằm ở cả train và test: **0 nhóm** ở từng phiên bản.
- Kiểm tra trùng pixel không phát hiện được mọi ảnh gần trùng, crop hay cùng chủ thể;
  dữ liệu nguồn không cung cấp group ID để thực hiện group-aware split.
- Cùng tập nguồn, tên tệp, thuật toán và seed sẽ tạo lại cùng danh sách. Lưu
  manifest để tránh phụ thuộc vào thay đổi phiên bản thư viện trong tương lai.
- Python: 3.11.16; Pillow: 12.3.0.
- SHA-256 của manifest: `9939a6ee404c6fbbdbe1b07a50763dc6d606497709385708512e1e98d71f2780`.

Tái tạo vào một thư mục đích chưa tồn tại:

```powershell
python prepare_mid_dataset.py --source-root "C:\Users\lanph\Downloads\Final_Dataset_" --output-root data_recreated --seed 42 --archive-format tar.xz
```

## 6. Dung lượng

MB trong bảng = 1.000.000 byte, không phải dung lượng cấp phát trên ổ đĩa.

| Bản | Kích thước | Tổng JPEG (MB) | Gói nén (MB) |
|---|---|---:|---:|
| Mid32 | 32 x 32 | 19.0571 | 12.3062 |
| Mid224 | 224 x 224 | 31.8035 | 24.6096 |

Gói TAR.XZ chỉ chứa cây thư mục ảnh của từng phiên bản; metadata nằm ngoài gói nén.
Giới hạn đề bài: Mid224 không quá 25M, Mid32 không quá 20M. Đối chiếu dung lượng
gói nén trong bảng khi nộp theo dạng nén; đề không định nghĩa rõ M/MB hay MiB.
TAR.XZ nén toàn bộ luồng TAR chung (solid), giúp loại bỏ phần lặp giữa nhiều
tệp nhỏ. Giải nén khôi phục nguyên byte JPEG. ZIP nén từng tệp riêng và có thêm
header cho từng ảnh, nên bản ZIP của bộ này lớn hơn giới hạn; dùng TAR.XZ để đóng gói.

Giải nén bằng 7-Zip/WinRAR hoặc lệnh `tar -xf Mid32.tar.xz` / `tar -xf Mid224.tar.xz`.

## 7. Bộ đọc và pipeline PyTorch

```powershell
conda activate fresher
python train_mid.py --data-root data --variant Mid32 --seed 42 --output-dir runs/mid32_seed42
python train_mid.py --data-root data --variant Mid224 --seed 42 --output-dir runs/mid224_seed42
```

Đây là lệnh huấn luyện để chạy sau khi chuẩn bị dữ liệu, không phải kết quả đã
huấn luyện. Mặc định dùng TickNet-basic với đầu ra 5 lớp; Mid32 sử dụng cấu hình
stride dành cho 32x32. Pipeline lưu cấu hình, log train từng epoch, checkpoint
cuối và kết quả test cuối quá trình; không chọn checkpoint theo test.

Train áp dụng random crop có padding và horizontal flip; test không augment.
ToTensor chuyển RGB về [0, 1]; không thêm Normalize vì TickNet hiện có data_bn.
Seed điều khiển Python, NumPy, Torch, sampler và worker. Kết quả huấn luyện vẫn
phụ thuộc môi trường/phần cứng; seed cố định không bảo đảm giống bit giữa mọi máy.

## 8. Tài liệu phương pháp

- [ImageFolder — PyTorch](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
- [Stratification — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)
- [Reproducibility — PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html)
