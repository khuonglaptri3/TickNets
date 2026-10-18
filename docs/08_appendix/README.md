# 08. Phụ lục và kiểm tra bản nộp

## Đánh giá trên GPU

```bash
python -m pip install -r requirements.txt
python checkpoint10.py --batch-size 16
python checkpoint100.py --batch-size 16
```

Hai file mặc định dùng `cuda:0`, nạp đúng trọng số từ [manifest lựa chọn](../05_results/selected_checkpoints.json), tính tham số/FLOPs, xác thực fingerprint dữ liệu và chạy inference mới trên 10.000 ảnh mỗi dataset. Thiếu CUDA thì báo lỗi; không tự chuyển sang CPU. Dùng `--help` xem tùy chọn, `--no-download` khi đã có dữ liệu gốc.

JSON, CSV dự đoán, confusion matrix và chỉ số từng lớp được ghi vào `runs/`, không ghi đè artifact gốc. [Bằng chứng GPU](validation/README.md).

## Tái lập train

Dùng [bốn notebook Kaggle](../04_training/README.md). Mỗi notebook tự giải nén mã train, cấu hình và launcher đã nhúng sẵn. Bản nộp không cần thư mục test, script train hoặc công cụ phát triển ở root. Notebook đã chạy và toàn bộ artifact Kaggle gốc được giữ nguyên.

## Mirror test khi máy chủ archive gốc tải chậm

[reproduce_test_mirror.py](validation/reproduce_test_mirror.py) chạy cùng hàm đánh giá bằng mirror UofT. Script chỉ chấp nhận pixel, nhãn và thứ tự có SHA-256 trùng tập test gốc; hai file checkpoint chính vẫn tải archive gốc qua torchvision.

```bash
python -m pip install pyarrow
python docs/08_appendix/validation/reproduce_test_mirror.py --device cuda:0 --batch-size 16
```

## Mã tác giả

[author/](../../author/README.md) giữ nguyên bộ mã từ commit nhập nguồn ban đầu `cfbd3a3fe663158918e30934271e80f1949efe94`. Mô hình L dùng các lớp dùng chung trong `author/models/common.py`; phần điều chỉnh của nhóm ở `models/ticknet_l.py`. Tham khảo và trích dẫn ở [07_references](../07_references/README.md).
