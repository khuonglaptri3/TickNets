# Bằng chứng kiểm thử Phase 1–4

Ngày hoàn tất review: 2026-10-06. Môi trường local: Python 3.11.16,
PyTorch 2.13.0+cu126, TorchVision 0.28.0+cu126, pytest 9.1.1.

## Kết quả

| Kiểm tra | Kết quả |
| --- | --- |
| `python -m pytest tests -q -p no:cacheprovider` | **193 passed, 2 skipped**, 186,93 giây |
| Mã nguồn được trích trực tiếp từ source bundle của notebook, chạy bộ test CIFAR nhúng | **155 passed**, 156,47 giây; CUDA có sẵn ở lần kiểm tra này |
| Recovery và kết quả sau vòng review cuối | **54 passed** |
| Notebook: snapshot, bootstrap, cấu hình bốn phase, biên dịch cell, fail-fast, tái tạo xác định | **9 passed** |
| CUDA bị ẩn bằng `CUDA_VISIBLE_DEVICES=-1` cho preflight | Đã xác nhận `torch.cuda.is_available() == False` |
| `git diff --check` | Không có lỗi whitespace |

Bộ test hiện có **195 trường hợp**. Hai test skip là tích hợp original CIFAR-10
và CIFAR-100: file dữ liệu checksum hợp lệ chưa có ở máy local. Download đã thử nhưng
tốc độ quá chậm để hoàn tất trong vòng kiểm tra; partial download đã được dọn.
Các test downloader dùng fixture nhỏ để xác nhận cache lỗi, checksum, fallback,
post-extraction và từ chối archive không an toàn; những fixture đó không thay thế
hai test tích hợp dữ liệu thật. CI tải dữ liệu trước pytest để chạy hai test này.

Preflight notebook chủ động ẩn GPU và chạy kiểm tra CPU; sáu trường hợp CUDA được
skip trong chế độ đó. Lần kiểm tra snapshot ở local có CUDA đã chạy đủ cả sáu.
Kết quả test không phải accuracy benchmark sau 200 epochs.

## Phạm vi từng file

| File test | Số trường hợp | Những hành vi được kiểm tra |
| --- | ---: | --- |
| `test_cifar_contracts.py` | 69 | Config không hợp lệ; Nesterov; LR; đáp án metric biết trước; batch lẻ; empty/NaN/Inf; Cutout sát biên và kích thước lẻ; normalization; stratified counts; loader thật với dữ liệu mock; budget/gradient; train/eval L/Basic × CIFAR-10/100 × SGD/Adam; TickNet-L CUDA |
| `test_cifar_recovery.py` | 30 | Resume khớp weights/optimizer/scheduler/history/best; augmentation Python/NumPy/Torch; nhiều worker; CUDA Dropout; từ chối recipe/source/revision/runtime/data/history/scheduler sai; recover log/best mất; không test lúc pause; tái sử dụng run hoàn tất |
| `test_cifar_results.py` | 24 | Đọc val_top1; giữ precision; thiếu run; artifact bị sửa; metric sai dù rehash; chống kết quả synthetic mang nhãn official; đóng gói bốn phase; checkpoint ghi lỗi vẫn giữ bản commit trước |
| `test_cifar_download_integrity.py` | 14 | Missing/unrelated/corrupt/valid cache trên cả CIFAR-10/100; post-extraction checksum; path traversal/link; fallback và dọn partial file |
| `test_kaggle_phases.py` | 9 | Matrix cố định; source bundle đúng checkout; snapshot xác định; mọi cell compile được; bootstrap reject source bị sửa; subprocess fail-fast; recovery copy |
| `test_cifar_official_data.py` | 2 | Original CIFAR files, 45.000/5.000/10.000 mẫu, phân tầng từng lớp, membership không giao nhau, real-image forward/backward; skip khi thiếu dữ liệu |
| Các test có sẵn | 47 | CIFAR transforms/trainer, downloader, midterm dataset/pipeline, profiler và cleaning; test ObjectVerifier đã thay download pretrained bằng classifier có đáp án biết trước |

## Snapshot để upload

Bốn notebook có cùng `ticknets_source_sha256`:

```text
13bd640469d1ea1b8c0583c59afe4f327ca8bcbce4b96880a698014190f9f909
```

SHA-256 này mô tả source bundle, không phải accuracy hay hash checkpoint đã train.
Bootstrap và cell chạy phase dùng `check=True`; một kết quả thiếu/sai sẽ không được
đóng gói như results. Hướng dẫn chạy và resume nằm ở [README](README.md).

Chưa chạy tám thực nghiệm 200 epochs trên Kaggle trong phiên review này. Chỉ số test
cuối cùng phải lấy từ các run hoàn tất và qua validator. Đề thi vẫn cần báo cáo PDF,
thuyết minh kiến trúc, vấn đáp và nộp đúng hạn.
