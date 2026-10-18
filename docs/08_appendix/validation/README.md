# Xác thực GPU

Hai checkpoint được đánh giá lại trên NVIDIA GeForce RTX 2050 (`cuda:0`), batch 16, đủ 10.000 ảnh mỗi dataset.

| Dataset | Ảnh đúng / tổng | Test Top-1 | Macro-F1 | Loss |
|---|---:|---:|---:|---:|
| CIFAR-10 | 9604/10000 | 96.04% | 0.960355 | 0.172538 |
| CIFAR-100 | 8123/10000 | 81.23% | 0.812142 | 0.700414 |

[GPU_VERIFICATION.json](GPU_VERIFICATION.json) ghi thiết bị, batch size, trọng số, fingerprint dữ liệu, nguồn đang dùng và đối chiếu từng dự đoán với artifact Kaggle. Lượt kiểm tra dùng mirror UofT có pixel/nhãn/thứ tự trùng SHA-256 trong checkpoint; không kiểm tra hoàn tất đường tải archive gốc của torchvision.

[reproduce_test_mirror.py](reproduce_test_mirror.py) tái lập đánh giá này. Cài `pyarrow`, rồi chạy từ repo root:

```bash
python docs/08_appendix/validation/reproduce_test_mirror.py --device cuda:0 --batch-size 16
```

Bốn notebook dùng để chạy lại cũng được kiểm tra nguồn nhúng, manifest và SHA-256 so với config đã train. Notebook đã chạy cùng 36 artifact gốc được giữ nguyên.
