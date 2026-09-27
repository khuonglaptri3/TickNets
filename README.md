# TickNets: Efficient tick-shape networks of full-residual point-depth-point blocks for image classification

## Dataset giữa kỳ: Mid32 / Mid224

Dataset chuẩn hóa nằm trong `data/`, theo cấu trúc
`data/<Mid32|Mid224>/<train|test>/<class>/*.jpeg`.
Mỗi lớp có 5.000 ảnh train và 50 ảnh test; chia lại cả 5 lớp bằng seed 42.
Hai độ phân giải dùng chung một manifest. Xem
[mô tả cách chia và kết quả kiểm tra](docs/DATASET_SPLIT.md).

```powershell
conda activate fresher
python train_mid.py --data-root data --variant Mid32 --seed 42 --check-data
python train_mid.py --data-root data --variant Mid224 --seed 42 --check-data
```

Huấn luyện baseline TickNet-basic, với đầu ra 5 lớp:

```powershell
python train_mid.py --data-root data --variant Mid32 --seed 42 --output-dir runs/mid32_seed42
python train_mid.py --data-root data --variant Mid224 --seed 42 --output-dir runs/mid224_seed42
```

Các lệnh train mặc định chạy 200 epoch; có thể cấu hình bằng `--help`.
Mỗi run tạo `config.json`, `epochs.csv`, `last.pt`, `test_metrics.json`.
Test được đánh giá ở cuối, không dùng để chọn checkpoint. Dùng thư mục output
mới cho mỗi run. Có thể đánh giá lại checkpoint bằng `--evaluate <last.pt>`.
Đây là pipeline cho backbone gốc, chưa phải kiến trúc L cải tiến của bài nộp.

Tạo lại dataset từ nguồn vào một thư mục chưa tồn tại:

```powershell
python prepare_mid_dataset.py --source-root "C:\Users\lanph\Downloads\Final_Dataset_" --output-root data_recreated --seed 42 --archive-format tar.xz
```

Toàn bộ `data/` được lưu trong Git, gồm ảnh của cả hai độ phân giải, gói TAR.XZ,
manifest, cấu hình và báo cáo kiểm tra. Clone repository sẽ tải kèm dữ liệu.
Git giữ nguyên byte của các tệp dữ liệu để bảo toàn checksum đã ghi nhận.
Kết quả huấn luyện trong `runs/` và cache được bỏ qua bởi Git.
Kiểm thử: `python -m pytest -q`.

**Abstract:**

* Light-weight convolutional neural networks (CNNs) are crucial for deploying computer vision applications in mobile devices.
However, such models ordinarily have steady-increased channels in their backbone leading to a sharp increase in the model size; while the deficiency of identity mappings in their residual mechanism can lead to modest performance in feature extraction.
To mitigate those issues, we propose light-weight networks based on three novel concepts as follows.
Firstly,  an efficient perceptron is presented to encapsulate point-depth-point (PDP) features extracted by light-weight convolutions along with a full-residual (FR) mechanism through the architecture of a network.
This full-residual connection is proposed to deal with the shortcoming of the existing light-weight models whose architecture has incompletely exploited identity mappings. It is due to the variability of spatial dimension caused by several strides of their convolutional operations.
Secondly, a tick-shape backbone is then introduced by designing its structure in accordance with the channel elasticity of the FR-PDP perceptron subject to the shape of a check mark.
Thirdly, taking advantage of the channel elasticity concept, three tick-shape networks (TickNets) are constructed in a light-weight architecture by hooking one or more tick-shape backbones.
Experimental results for image classification on benchmark datasets have clearly corroborated the prominence of the proposed methods.

<u>**Training TickNets on Datasets:**</u>

For Stanford Dogs. Note that it will automatically run for all TickNets, i.e., TickNet-basic, TickNet-small and TickNet-large
```
$ python TickNet_Dogs.py
```
For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small
```
$ python TickNet_ImageNet.py -a small
$ python TickNet_Places365.py -a small 
```
Note: Subject to your system, modify these training files (*.py) to have the right path to datasets

**Validating the trained models of TickNets:**
* For Stanford Dogs (TickNet-small)
```
$ python TickNet_Dogs.py --evaluate
```
* For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small.
```
$ python TickNet_ImageNet.py -a small --evaluate
$ python TickNet_Places365.py -a small --evaluate
```

Note: For instances of validation of TickNet-small, download the trained model of TickNet-small on Datasets: [Click here for Places365](https://drive.google.com/drive/folders/1EdlA3tuOutBJMR23B-fcSOKKB69hAQ5R?usp=sharing); [Click here for ImageNet-1k](https://drive.google.com/drive/folders/1t1M_QJwCmcaTgKBsJBmzrU-kabQeOPDT?usp=sharing); [Click here for Stanford Dogs](https://drive.google.com/drive/folders/1RGglukdrd5xDrGSo6ONmHTCZNZ-YwpZb?usp=sharing). And then locate the downloaded file at ./checkpoints/[name_dataset]/small

**Related citations:**

If you use any materials, please cite the following relevant works.

```
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  volume       = {596},
  pages        = {127942},
  year         = {2024},
  url          = {https://doi.org/10.1016/j.neucom.2024.127942}
}
```
