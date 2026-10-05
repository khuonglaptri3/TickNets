# Chạy Phase 1–4 với mã nguồn đã được kiểm tra

Bốn notebook `Phase1_CIFAR10_SGD.ipynb`, `Phase2_CIFAR10_Adam.ipynb`,
`Phase3_CIFAR100_SGD.ipynb`, `Phase4_CIFAR100_Adam.ipynb` chứa cùng một snapshot
mã nguồn và test, có SHA-256. Upload trực tiếp notebook lên Kaggle; không cần push
thay đổi lên GitHub để notebook nhận được mã mới. Bản baseline và notebook grid-search
cũ vẫn dùng cách clone GitHub; hướng dẫn snapshot này chỉ áp dụng cho Phase 1–4.

## Chạy lần đầu

1. Import notebook tương ứng, chọn GPU và bật Internet.
2. Chạy cell khởi tạo. Cell kiểm tra mã nguồn, GPU và khả năng import profiler;
   chỉ cài dependency kiểm thử/bảng kết quả còn thiếu, không thay bản torch/CUDA có sẵn.
3. Chạy preflight tests. Bất kỳ test fail nào sẽ dừng notebook trước khi train.
   Test CUDA được kiểm tra riêng ở local/CI có GPU; preflight notebook chạy trên CPU.
4. Cell cấu hình có `EPOCHS_PER_SESSION = None` để chạy toàn bộ số epoch còn lại.
   Nếu cần chia phiên, đặt một số nguyên như `50`; ngân sách này áp dụng cho từng run.
   Không có cam kết hai run 200 epoch sẽ vừa thời gian của một phiên Kaggle.
5. Chạy cell cuối. Chương trình dùng `subprocess.run(check=True)` nên lệnh lỗi không
   bị bỏ qua. Download phải khớp MD5 của archive và từng file CIFAR đã giải nén.

Pipeline dùng FP32, tắt TF32 và bật deterministic algorithms. Mỗi run giữ đúng
seed 42, batch size 128, 200 epochs, chia stratified 45.000 train / 5.000 validation.
Validation/test không có crop, flip hay Cutout. Checkpoint tốt nhất được chọn bằng
validation Top-1, giải hòa bằng validation loss thấp hơn; tie hoàn toàn giữ epoch trước.
Learning rate thay đổi theo epoch với CosineAnnealingLR, `eta_min=0`.
Chọn learning rate bằng validation; không chọn dựa vào kết quả test.

## Chạy tiếp ở phiên mới

- Download `phaseN_<dataset>_<optimizer>_recovery.zip` trước khi phiên kết thúc.
  Nếu phiên bị timeout trước khi đóng zip, lưu các thư mục run có `last.pt` từ output.
- Giải nén file recovery, tạo Kaggle Dataset chứa các thư mục run và attach vào notebook.
- Đặt `RESUME_ROOT` thành đường dẫn đến thư mục chứa các run,
  ví dụ `/kaggle/input/my-phase1-recovery` (điều chỉnh theo cấu trúc dataset thực tế).
- Dùng lại cùng notebook, phiên bản torch/torchvision/NumPy/Pillow và loại GPU.
  Không đổi learning rate, optimizer, seed, số worker hay tổng epochs khi resume.
- Checkpoint `last.pt` chứa trạng thái ngẫu nhiên, optimizer, scheduler, lịch sử và
  bản sao checkpoint tốt nhất. Nếu chỉ còn `last.pt`, trainer vẫn dựng lại được
  `epochs.csv` và `best_val.pt`. Trainer v1 không có đủ trạng thái và bị từ chối resume.

Checkpoint được commit bằng atomic replace sau mỗi epoch. Log/checkpoint tốt nhất
được dựng từ checkpoint đã commit, nên lỗi trong lúc xuất log không làm trùng epoch.
Không đánh giá official test khi dừng trước epoch 200. Run hoàn thành được xác thực
và tái sử dụng khi chạy lại notebook, không tự đánh giá test lần nữa.

## Đọc kết quả

- `*_recovery.zip`: còn run chưa hoàn thành; dùng để chạy tiếp, không dùng như kết quả cuối.
- `*_results.zip`: cả hai run đủ 200 epochs và được xác thực. Có summary CSV/Markdown
  và `phase_manifest.json` chứa SHA-256 của artifact.
- Mỗi run có cấu hình, phiên bản thư viện/GPU, SHA-256 mã nguồn và dữ liệu,
  log từng epoch, checkpoint, `test_metrics.json`, confusion matrix, dự đoán từng ảnh
  kèm negative log-likelihood, và `completion.json`.
- Validator kiểm tra 45.000/5.000 mẫu ở mọi epoch, 10.000 mẫu test, checkpoint được
  chọn bằng validation, giới hạn tài nguyên, SHA-256, Top-1, Macro-F1 và loss tính
  lại từ artifact. Một run thiếu/sai artifact sẽ không được xuất như kết quả hoàn tất.

Sau khi tải đủ bốn phase, giải nén các thư mục run vào `runs/` rồi chạy:

```bash
python scripts/aggregate_grid_search.py --runs-dir runs
```

Mặc định cần đủ 8 run đã xác thực. `--allow-partial` chỉ dùng để xem tiến độ;
bảng ghi rõ số run đã hoàn tất. CSV giữ số liệu dạng số, không làm tròn sớm.

## Tái tạo notebook sau khi sửa code

```bash
python scripts/generate_kaggle_phase_notebooks.py
python -m pytest tests -q
```

Snapshot được tạo xác định và chuẩn hóa newline. CI kiểm tra notebook được commit
có cùng nội dung với snapshot tạo từ checkout. Sau khi sửa mã nguồn/test/config,
phải tái tạo notebook trước khi upload.

Các kiểm tra này bảo vệ tính đúng và khả năng kiểm chứng của thực nghiệm. Độ chính xác
phân loại cuối cùng phải được đo sau huấn luyện thực tế; test không chứng minh trước
được accuracy sau 200 epochs hay thứ hạng của nhóm.
