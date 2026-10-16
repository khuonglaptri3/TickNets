# Báo cáo giữa kỳ — hướng dẫn đọc theo flow

**Bản tổng hợp đầy đủ:** [MIDTERM_REPORT.md](MIDTERM_REPORT.md), có learning curves, confusion matrix, chỉ số từng lớp và phụ lục 200 epoch của hai checkpoint được chọn.

Nhánh này phát triển bài giữa kỳ trên **Mid224 và Mid32**, với năm lớp bird, cat, dog, frog, horse.

| Thứ tự | Phần báo cáo | Bắt đầu đọc |
|---|---|---|
| 1 | Bài toán và yêu cầu | [01_introduction](01_introduction/README.md) |
| 2 | Dữ liệu và tiền xử lý | [02_dataset](02_dataset/README.md) |
| 3 | Kiến trúc và ngân sách mô hình | [03_model](03_model/README.md) |
| 4 | Thiết lập và quy trình thực nghiệm | [04_training](04_training/README.md) |
| 5 | Kết quả, biểu đồ và log | [05_results](05_results/README.md) |
| 6 | So sánh hai hướng thử nghiệm và thảo luận | [06_discussion](06_discussion/README.md) |
| 7 | Tài liệu tham khảo | [07_references](07_references/README.md) |
| 8 | Lệnh chạy, kiểm định và lịch sử phát triển | [08_appendix](08_appendix/README.md) |

## Chạy checkpoint báo cáo

Chạy tại thư mục gốc repo:

```powershell
python checkpoint224.py
python checkpoint32.py
```

Hai file giữ nguyên ở root và dùng trọng số trong [checkpoints](../checkpoints/README.md). Chúng thực hiện suy luận trên ảnh test và in kết quả đo được.

## Quy tắc dùng số liệu

- Lấy thông số từ `config.json` của đúng run, kết quả từ `test_metrics.json`/CSV, và chi tiết theo epoch từ `epochs.csv`.
- Phân biệt kết quả lịch sử train đủ 25.000 ảnh với thí nghiệm có 22.500 train + 2.500 validation.
- Trong nhóm smoothing/LR, khóa lựa chọn theo validation chọn baseline cho Mid224; checkpoint smoothing 0,10 được ghim để trình diễn mức test 99,2%. Xem [thảo luận](06_discussion/README.md).
- Các config, checkpoint, CSV kết quả và manifest nguồn huấn luyện của các run còn lại được giữ nguyên byte. Cấu hình dùng để chạy lượt mới nằm trong `configs/midterm`. [Biên bản sắp xếp](08_appendix/restructure_manifest.json) ghi đường dẫn trước/sau và checksum.
