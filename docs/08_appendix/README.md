# 8. Phụ lục và kiểm chứng

| Nội dung | Nguồn |
|---|---|
| Các lệnh chạy dự án | [COMMANDS_GUIDE.md](COMMANDS_GUIDE.md) |
| Quy trình train/evaluate hiện hành | [MIDTERM_PROTOCOL.md](../04_training/MIDTERM_PROTOCOL.md) |
| Kiểm thử hạ tầng ở thời điểm triển khai | [IMPLEMENTATION_VERIFICATION.json](IMPLEMENTATION_VERIFICATION.json) |
| Hồ sơ kiểm định kết quả smoothing/LR | [provenance](../05_results/experiment_midterm_smoothing_lr/provenance) |
| Lịch sử thiết kế, kế hoạch triển khai Midterm | [development](development) |
| Đường dẫn và checksum tại thời điểm sắp xếp | [restructure_manifest.json](restructure_manifest.json) |
| Biên bản xóa các thử nghiệm đã loại khỏi phạm vi | [removed_experiments.json](removed_experiments.json) |
| Biên bản dọn ứng viên kiến trúc và bảng so sánh | [model_cleanup.json](model_cleanup.json) |
| Kiểm thử sau khi dọn kiến trúc | [JSON](validation/model_cleanup_tests.json), [JUnit](validation/model_cleanup_tests.xml) |
| Chạy lại hai checkpoint sau khi dọn kiến trúc | [Kết quả](validation/model_cleanup_checkpoint_results.json) |
| Tự giải nén từ thư mục chỉ có hai gói dataset | [Checkpoint](validation/archive_only_checkpoint_results.json), [file được Git theo dõi](validation/archive_only_git_results.json) |
| Kết quả suy luận lại hai checkpoint Midterm | [checkpoint_results.json](validation/checkpoint_results.json) |
| Log chạy checkpoint32 / checkpoint224 | [Mid32](validation/checkpoint32.txt), [Mid224](validation/checkpoint224.txt) |
| Kiểm thử sau khi dọn nhánh và chuyển thư mục | [test_results.json](validation/test_results.json) |
| Kiểm thử các thử nghiệm còn lại sau khi loại hai nhóm augmentation | [JSON](validation/remaining_experiments_tests.json), [JUnit](validation/remaining_experiments_tests.xml) |

## Log từng epoch

- [Mid224 smoothing 0,10](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_smoothing010/epochs.csv).
- [Mid32 LR 0,15](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid32_lr015/epochs.csv).
- [Mid224 baseline](../05_results/experiment_midterm_smoothing_lr/checkpoints/l_mid224_baseline/epochs.csv).

## Kiểm tra checkpoint báo cáo

```powershell
python checkpoint224.py
python checkpoint32.py
```

Có thể đặt `--checkpoint`, `--data-root` và `--batch-size`. Hai script nằm ở thư mục gốc và được giữ nguyên trong lần sắp xếp này.

Các manifest và hash nguồn huấn luyện lịch sử ghi nhận checkout tại thời điểm tạo run; chúng có thể chứa tên nguồn đã bị xóa sau đó. Biên bản xóa ghi rõ các artifact được loại khỏi nhánh. Không sửa metadata của checkpoint hoặc hash lịch sử để che khác biệt với checkout hiện tại. Đổi tên ứng viên 10 block làm thay đổi tên nguồn/đăng ký model; các checkpoint L v1 vẫn nạp qua implementation L v1 riêng.
