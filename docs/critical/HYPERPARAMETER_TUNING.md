# 9. Hyperparameter Tuning: Thiết kế, Không gian tìm kiếm & Nguyên tắc thực thi

Tài liệu này ghi nhận chi tiết hiện trạng thiết kế, không gian siêu tham số (Hyperparameter Search Space), nguyên tắc chống rò rỉ dữ liệu khi tinh chỉnh (Tuning Protocol) và lộ trình thực nghiệm trong đồ án TickNets.

---

## 1. Bảng tổng hợp Siêu tham số Mặc định (Baseline Hyperparameters)

Các siêu tham số mặc định trong pipeline huấn luyện [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py#L44-L72) được kế thừa theo chuẩn kinh điển của các nghiên cứu thị giác máy tính hiện đại (ResNet, MobileNet):

| Siêu tham số (Hyperparameter) | Giá trị mặc định | Cờ dòng lệnh (CLI flag) | Ý nghĩa kỹ thuật & Căn cứ lựa chọn |
| :--- | :---: | :--- | :--- |
| **Thuật toán tối ưu (Optimizer)** | **SGD** | Hardcoded trong pipeline | SGD với momentum giúp vượt qua các điểm yên ngựa (saddle points) và tìm cực tiểu phẳng tốt hơn Adam trên CNN |
| **Tốc độ học (Learning Rate - $\eta$)** | **0.1** | `--learning-rate 0.1` | Mức chuẩn khởi đầu cho SGD khi train from scratch trên ảnh tự nhiên |
| **Động lượng (Momentum - $\mu$)** | **0.9** | `--momentum 0.9` | Tích lũy vận tốc gradient, làm mượt các dao động dao động ziczac |
| **Suy giảm trọng số (Weight Decay - $\lambda$)** | **$10^{-4}$** ($0.0001$) | `--weight-decay 1e-4` | Chuẩn hóa $L_2$ chống overfitting, phạt các trọng số quá lớn |
| **Lập lịch học (LR Scheduler)** | **Cosine Annealing** | `CosineAnnealingLR` | Hạ dần LR theo hàm cosine từ $0.1 \rightarrow 0$ trong suốt số epoch, giúp hội tụ mượt mà |
| **Số lượng Epochs** | **200** | `--epochs 200` | Đủ dài để các khối PDP học được các tầng biểu diễn đặc trưng phân cấp |
| **Kích thước Batch (Batch Size)** | **64** | `--batch-size 64` | Cân bằng giữa tốc độ cập nhật gradient và dung lượng bộ nhớ VRAM |
| **Tăng cường dữ liệu (Augmentation)** | Bật (Crop + Flip) | `--no-augment` (để tắt) | Giảm overfit, tăng tính bất biến với dịch chuyển và lật ảnh |
| **Hạt giống ngẫu nhiên (Seed)** | **42** | `--seed 42` | Cố định toàn bộ luồng số ngẫu nhiên cho Python, NumPy, PyTorch CPU/CUDA |

---

## 2. Nguyên tắc vàng: Chống rò rỉ dữ liệu khi Tuning (Tuning Integrity Protocol)

Trong tài liệu kỹ thuật [docs/DATASET_SPLIT.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/DATASET_SPLIT.md#L40-L42), [docs/MODEL_L.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_L.md#L131-L133) và [docs/MODEL_C.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_C.md#L141-L144), đồ án thiết lập nguyên tắc nghiêm ngặt:

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG SỬ DỤNG TẬP TEST ĐỂ TUNING SIÊU THAM SỐ!**
> - Tập **Test (250 ảnh)** là **Held-Out Test Set**, chỉ được dùng duy nhất một lần ở cuối cùng để báo cáo năng lực khái quát hóa khách quan.
> - Nếu dùng tập Test để chọn Learning Rate, chọn Batch Size, chọn Epoch dừng sớm (Early Stopping) hay chọn Checkpoint tốt nhất, mô hình sẽ bị hiện tượng **Overfitting to Test Set** (Rò rỉ dữ liệu qua siêu tham số - Data Leakage via Hyperparameters), khiến kết quả báo cáo khoa học mất giá trị tin cậy.

### Quy trình Tuning chuẩn mực được quy định trong đồ án:
1. **Tách tập Validation từ Train**:
   Từ 25.000 ảnh Train, thực hiện tách phân tầng 10% - 20% (hoặc K-Fold Cross Validation trên train):
   - $\text{Train}_{\text{sub}}$: 20.000 - 22.500 ảnh.
   - $\text{Val}$: 2.500 - 5.000 ảnh.
2. **Thực hiện thử nghiệm Grid / Random Search trên tập Val**:
   So sánh Validation Accuracy và Validation Loss giữa các bộ siêu tham số trên tập $\text{Val}$.
3. **Chốt bộ tham số tối ưu (Best Hyperparameters)**:
   Sau khi chọn được bộ siêu tham số tốt nhất trên $\text{Val}$, huấn luyện lại mô hình trên toàn bộ tập Train (25.000 ảnh).
4. **Đánh giá cuối cùng trên Held-Out Test**:
   Chỉ chạy kiểm thử trên tập Test đúng 1 lần bằng cờ `--evaluate <checkpoint.pt>`.

---

## 3. Không gian Siêu tham số đề xuất cho Thực nghiệm (Search Space)

Nhờ thiết kế linh hoạt của `train_mid.py`, người thực nghiệm có thể chạy kịch bản tự động quét qua các không gian tham số:

### 3.1. Grid Search đề xuất
1. **Learning Rate ($\eta$)**:
   $$\eta \in \{0.01, 0.05, 0.1, 0.2\}$$
   *(Kiểm tra độ nhạy của tốc độ hội tụ khi huấn luyện from scratch trên dải chuẩn hóa `data_bn`)*.
2. **Weight Decay ($\lambda$)**:
   $$\lambda \in \{10^{-5}, 10^{-4}, 5 \times 10^{-4}\}$$
   *(Kiểm tra mức độ phạt trọng số để kiểm soát hiện tượng overfit trên tập 5.000 ảnh/lớp)*.
3. **Batch Size ($B$)**:
   $$B \in \{32, 64, 128\}$$
   *(Ảnh hưởng đến độ ồn của gradient trong SGD và độ ổn định của thống kê running mean/var trong tầng `data_bn`)*.
4. **Ablation Study (Thực nghiệm bóc tách)**:
   - Thử nghiệm tắt Data Augmentation (`--no-augment`) so với bật Augmentation để đo mức độ đóng góp của `RandomCrop (reflect)` và `RandomHorizontalFlip`.
   - Thử nghiệm so sánh hiệu quả giữa 3 kiến trúc: `--model basic` vs. `--model l` vs. `--model c` trên cùng bộ siêu tham số cố định.

---

## 4. Hiện trạng thực tế trong đồ án (Current Implementation Status)

Cần làm rõ hiện trạng trong báo cáo:

1. **Những gì ĐÃ HOÀN THÀNH**:
   - Xây dựng hoàn chỉnh toàn bộ hạ tầng điều khiển siêu tham số qua giao diện CLI chuẩn POSIX trong `train_mid.py`.
   - Cơ chế tự động ghi nhận toàn bộ siêu tham số vào file `config.json` và log chi tiết từng epoch vào `epochs.csv` trong thư mục đầu ra `--output-dir`.
   - Cơ chế cố định hạt giống ngẫu nhiên (`seed_everything` kết hợp `seed_worker`) đảm bảo việc so sánh giữa các bộ siêu tham số diễn ra trên cùng điều kiện ban đầu hoàn toàn công bằng.
   - Cơ chế bảo vệ nguồn gốc manifest (`split_manifest_sha256`) và phiên bản kiến trúc (`architecture_revision`).

2. **Những gì CHƯA THỰC HIỆN**:
   - Các thí nghiệm chạy quét toàn diện (full sweep) nhiều bộ siêu tham số trong 200 epochs **chưa được thực hiện** (thư mục `runs/` hiện chưa lưu kết quả huấn luyện dài hạn).
   - Trong tài liệu [docs/MODEL_L.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_L.md#L134-L135) và [docs/MODEL_C.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_C.md#L141-L144), nhóm nghiên cứu ghi nhận rõ: việc huấn luyện đầy đủ 200 epoch để tinh chỉnh siêu tham số và so sánh độ chính xác (accuracy) thực tế là **bước thực nghiệm tiếp theo** sau khi hoàn thành thiết kế kiến trúc và chuẩn bị dữ liệu.

---

## 5. Liên kết tham chiếu trong dự án
- [train_mid.py](file:///home/intern-tdkhuong/Desktop/TickNets/train_mid.py): Định nghĩa các parser argument và triển khai CosineAnnealingLR.
- [docs/DATASET_SPLIT.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/DATASET_SPLIT.md): Quy định về việc bảo vệ tập Test và nguyên tắc tách Validation khi tuning.
- [docs/MODEL_L.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_L.md) & [docs/MODEL_C.md](file:///home/intern-tdkhuong/Desktop/TickNets/docs/MODEL_C.md): Kế hoạch thực nghiệm và đề xuất ablation study.
