# Tổng kết kết quả và kiểm định TickNet Basic / L

Tài liệu này giữ phần bằng chứng Basic và L từ audit ngày 2026-10-01. Việc cập nhật phạm vi tài liệu không huấn luyện hoặc suy luận lại các lượt lịch sử.

## 1. Kết quả lịch sử

| Mô hình | Dataset | Test Top-1 | Đúng / 250 | Macro-F1 | Tham số | GFLOPs |
|---|---|---:|---:|---:|---:|---:|
| Basic | Mid32 | 89,60% | 224 | 0,895598 | 1.062.223 | 0,158428 |
| L | Mid32 | 91,60% | 229 | 0,915748 | 1.096.260 | 0,157821 |
| Basic | Mid224 | 92,40% | 231 | 0,923955 | 1.062.223 | 0,988343 |
| L | Mid224 | 95,60% | 239 | 0,955877 | 1.096.260 | 0,796760 |

Nguồn: [reported_metrics.csv](model_comparison/reported_metrics.csv), [verification.json](audit_20261001/verification.json) và [checkpoint manifest](checkpoints/checkpoint_manifest.csv). Quy ước: một ảnh eval, chỉ Conv2d/Linear, 1 MAC = 2 FLOPs.

L hơn Basic 5 ảnh đúng trên Mid32 và 8 ảnh đúng trên Mid224. FLOPs Mid224 giảm 19,4%, tham số tăng khoảng 3,2%. Kết quả này chưa xác lập đóng góp riêng của từng thay đổi kiến trúc.

## 2. Kiến trúc và tính đúng đắn

Basic dùng trực tiếp factory `build_TickNet(..., typesize='basic')` trong [mid_models.py](../../models/mid_models.py). Audit lịch sử đối chiếu các nguồn [TickNet.py](../../models/TickNet.py), [common.py](../../models/common.py) và [SE_Attention.py](../../models/SE_Attention.py) với nguồn tác giả tại commit `e71679978f861a599192ece7613dc7ad3768e55b`.

Block gốc có luồng PW1 tuyến tính → DW 3×3 + BN + ReLU → PW2 + BN + ReLU → SE → cộng shortcut. Basic có stem 32, năm block, stage 128/64/128/256/512 và head 1024.

L trong [ticknet_l.py](../../models/ticknet_l.py) giữ cấu trúc PW–DW–PW, SE và shortcut, đồng thời đổi phân bổ tài nguyên:

| Thành phần | Basic | L |
|---|---|---|
| Stem | 32 | 24 |
| Block | 5 | 7 |
| Stage output | 128/64/128/256/512 | 112/64/144/288/512 |
| Hidden PW1 | Cin | Khoảng 0,75 Cin từ stage 2, làm tròn bội 8 |
| Depthwise | 3×3 | 3×3 ở stage 1–2; chia kênh 3×3/5×5 ở stage 3–5 |
| Head | 1024 | 768 |

Stage 2 của Basic chiếm 521.028.608 FLOPs, khoảng 52,7% tổng chi phí Mid224. L giảm phần này bằng điều chỉnh kênh và hidden, rồi bổ sung block ở stage 3/4.

Phép kiểm tra block L với hidden=Cin và kernel=(3,) khớp block gốc khi sao chép trọng số. Xem [block_equivalence.json](audit_20261001/block_equivalence.json), [phân tích phương pháp](Model_L/method_soundness_and_verification/METHOD_SOUNDNESS.md) và [model profiles](../03_model/model_profiles.json).

## 3. Dữ liệu và khả năng so sánh

Hai độ phân giải dùng cùng filename/class/split membership, với 25.000 train và 250 test. Audit kiểm tra checksum 50.500 ảnh local theo manifest canonical.

Basic ghi manifest canonical SHA-256 `a71d026f7f2c910da1b1b7c534f72158dc82c62953972e27f6789debaf8db591`. L ghi schema `variant,split,class_name,filename` với SHA-256 `35bcc76fde54e97d54f7825cdb34376a3732fd79aa829624ea4fc982b4590bdf`.

Tái dựng schema L từ membership canonical khớp hash đã ghi. Điều này xác minh membership, nhưng không xác minh byte ảnh Kaggle lịch sử vì manifest L không lưu checksum nội dung ảnh. Nguồn: [split_provenance.csv](model_comparison/split_provenance.csv).

Các khác biệt recipe cần ghi rõ:

- Basic dùng batch 64, workers 0; L Mid32 dùng batch 128 và L dùng workers 2.
- Cùng 200 epoch, Mid32 Basic có 391 bước/epoch, còn L có 196 bước/epoch.
- Kênh, độ sâu và kernel thay đồng thời; đây chưa phải ablation một yếu tố.
- Chỉ có một seed và 250 ảnh test; mỗi ảnh tương ứng 0,4 điểm phần trăm.

## 4. Kiểm định dựa trên dự đoán từng ảnh

Audit lịch sử đối chiếu confusion matrix và Top-1 sau khi nạp checkpoint bằng `state_dict` nghiêm ngặt. Các dự đoán Basic/L còn trong [audit_20261001](audit_20261001).

| Dataset | Cặp A → B | A đúng, B sai | A sai, B đúng | McNemar p hai phía |
|---|---|---:|---:|---:|
| Mid32 | Basic → L | 6 | 11 | 0,3323 |
| Mid224 | Basic → L | 6 | 14 | 0,1153 |

Các p-value chưa đạt ngưỡng 0,05. Chưa thể viết rằng đã chứng minh L vượt trội có ý nghĩa thống kê; điều này cũng không chứng minh hai mô hình tương đương.

## 5. Tái lập và kết luận

Notebook Basic Mid224 chạy hai phiên 100+100 và yêu cầu các cờ resume không thuộc trainer lịch sử hiện hành. Cần lưu đúng phiên bản trainer và môi trường nếu muốn tái tạo toàn bộ quy trình đó. Checkpoint và log cuối không tự chứng minh khả năng khôi phục RNG chính xác.

Metadata checkpoint và hash nguồn lịch sử được giữ nguyên. Sau khi dọn mã và đổi đăng ký model, việc đối chiếu hash với checkout hiện tại có thể báo khác biệt; không sửa hash gốc để che thay đổi.

Kết quả hỗ trợ tính khả thi của L trong ngân sách đề bài. Để đánh giá độ ổn định, cần giữ recipe/split cố định và chạy nhiều seed đã định trước. Các thực nghiệm có validation, cùng confusion matrix và learning curves của hai checkpoint được ghim, nằm trong [báo cáo tổng hợp](../MIDTERM_REPORT.md).
