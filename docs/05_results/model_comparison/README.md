# TickNet Basic/L — so sánh kết quả lịch sử

Thư mục này đối chiếu các lượt train seed 42, 200 epoch của Basic và L trên Mid32/Mid224. Recipe lịch sử khác nhau nên chưa thể quy toàn bộ mức tăng accuracy cho riêng một thành phần của L.

| Mô hình | Dataset | Test Top-1 | Macro-F1 | Tham số | GFLOPs | Batch |
|---|---|---:|---:|---:|---:|---:|
| Basic | Mid32 | 89,60% | 0,895598 | 1.062.223 | 0,158428 | 64 |
| L | Mid32 | 91,60% | 0,915748 | 1.096.260 | 0,157821 | 128 |
| Basic | Mid224 | 92,40% | 0,923955 | 1.062.223 | 0,988343 | 64 |
| L | Mid224 | 95,60% | 0,955877 | 1.096.260 | 0,796760 | 64 |

Số liệu đầy đủ nằm trong [reported_metrics.csv](reported_metrics.csv). FLOPs tính Conv2d/Linear trên một ảnh eval, với 1 MAC = 2 FLOPs.

## Kết quả và giới hạn

L cao hơn Basic 2,0 điểm phần trăm trên Mid32 và 3,2 điểm trên Mid224. FLOPs Mid224 giảm khoảng 19,4%, trong khi tham số tăng khoảng 3,2%.

Basic dùng batch 64 và 0 worker; L dùng batch 128 trên Mid32, batch 64 trên Mid224 và 2 worker. Kênh, block và kernel cũng thay đồng thời. Cùng số epoch chưa bảo đảm cùng số bước cập nhật hoặc chuỗi augmentation.

Audit lịch sử đối chiếu filename/class/split membership và chạy lại các checkpoint trên tập test local. Hash manifest L khác Basic vì schema CSV khác nhau; tái dựng schema L từ membership canonical khớp hash đã lưu. Byte ảnh Kaggle lịch sử chưa được kiểm chứng độc lập. Xem [split_provenance.csv](split_provenance.csv) và [biên bản audit](../AUDIT_TICKNET_2026-10-01.md).

McNemar hai phía cho L so với Basic có p=0,3323 trên Mid32 và p=0,1153 trên Mid224. Một seed và 250 ảnh test chưa đủ để khẳng định lợi ích ổn định hoặc ưu thế có ý nghĩa thống kê.

## Hình và nguồn

- [Test Top-1](accuracy_by_dataset.png).
- [Macro-F1](macro_f1_by_dataset.png).
- [Accuracy và compute trên Mid224](accuracy_vs_compute_mid224.png).
- [Hồ sơ Basic](../Model_Basic/README.md).
- [Hồ sơ L](../Model_L/README.md).
- [Dự đoán và kiểm định lịch sử](../audit_20261001).

Các hình được dựng lại từ CSV bằng `python scripts/build_midterm_report.py`, chạy từ gốc repo. Kết quả có validation của smoothing/LR và stage 4 depth được trình bày riêng trong [báo cáo tổng hợp](../../MIDTERM_REPORT.md#so-sanh).
