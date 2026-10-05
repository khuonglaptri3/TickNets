# CIFAR trainer v2: kiểm thử và độ tin cậy Phase 1–4

Đây là đặc tả hiện hành cho các thay đổi về recovery, kiểm chứng kết quả và triển khai
Kaggle. Những spec/plan v1 cùng ngày là hồ sơ lịch sử; checkbox hoàn thành triển khai
không chứng minh hoàn thành bài thi hay thực nghiệm 200 epochs.

## Các điều kiện nghiệm thu

- SGD/Adam, CIFAR-10/100, TickNet-L/Basic có test train, chọn checkpoint và evaluate.
- Resume phải khớp chạy liên tục về weights, optimizer, scheduler, lịch sử và best epoch.
  Có trường hợp worker augmentation và CUDA; test CUDA skip nếu máy không có GPU.
- Lưu RNG Python/NumPy/PyTorch/CUDA và generator train/validation/test ở ranh giới epoch.
- `last.pt` commit trước dependent artifacts, chứa cả best weights để khôi phục
  sau lỗi ghi log hoặc khi chuyển sang thư mục output mới.
- Từ chối thay recipe, runtime/GPU, source, dataset pixels/labels hoặc membership khi resume.
- Dừng giữa chặng không evaluate test; run đầy đủ có completion evidence được tái sử dụng.
- Cache CIFAR chỉ hợp lệ khi tất cả file train/test/meta khớp checksum chính thức.
- Metric được kiểm tra bằng trường hợp có đáp án biết trước và đối chiếu dự đoán từng ảnh.
- Bốn notebook chứa mã nguồn/test đóng gói với SHA-256, không clone nhánh thay đổi.
- Mọi subprocess phải dừng khi lỗi; partial artifact phải được gọi recovery.
- Xuất results phải xác thực đủ 200 epochs, 45.000/5.000 train/val, 10.000 test,
  checkpoint được chọn bởi validation và tất cả artifact còn nguyên vẹn.

## Thông số thống nhất

- CIFAR-100 mean `(0.5071, 0.4867, 0.4408)`, std `(0.2675, 0.2565, 0.2761)`.
- CosineAnnealingLR: `T_max=200`, `eta_min=0` cho cả SGD và Adam.
- TickNet-L CIFAR-10: 1.100.105 learnable parameters, 157.828.544 FLOPs.
- TickNet-L CIFAR-100: 1.169.315 learnable parameters, 157.966.784 FLOPs.
- FLOPs là Conv2d/Linear, batch 1, eval forward, 1 MAC = 2 FLOPs;
  chưa tính BN, activation, pooling và phép toán phần tử.
- From-scratch là lựa chọn thực nghiệm của nhóm; DOCX không ghi một lệnh cấm pretrained.
  Backbone Conv dùng Kaiming Uniform, classifier weight dùng Xavier Normal;
  SE Linear và classifier bias giữ quy tắc khởi tạo của constructor PyTorch hiện có.

## Bằng chứng và giới hạn

Kết quả kiểm thử cuối cùng, các trường hợp skip và hướng dẫn upload được ghi tại
[`docs/kaggle/README.md`](../../kaggle/README.md) và
[`docs/kaggle/VALIDATION.md`](../../kaggle/VALIDATION.md).
Test tích hợp CIFAR thật chỉ chạy khi đã có file checksum hợp lệ, không tự tải qua mạng
trong pytest. Test GPU chỉ chạy khi CUDA có sẵn. Cần ghi rõ skip thay vì coi là pass.

Mã nguồn đã kiểm tra không phải kết quả thi: vẫn cần chạy đủ 8 thực nghiệm CIFAR,
hoàn thiện báo cáo PDF, giải thích kiến trúc khác mạng gốc hoặc kế thừa kiến trúc nhóm
đã đề xuất ở giữa kỳ, bảo đảm không trùng kiến trúc nhóm khác, chuẩn bị vấn đáp và nộp đúng hạn.
