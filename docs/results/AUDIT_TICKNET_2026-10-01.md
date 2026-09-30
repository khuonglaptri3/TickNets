# Tổng kết kết quả và kiểm định TickNet Basic / C / L

Ngày kiểm định: 2026-10-01. Mã nguồn được kiểm tra tại commit
`9cc8126942ba1a59899c348d20a34365eec7f137`.

## 1. Kết luận chính

**Basic giữ nguyên kiến trúc TickNet-Basic gốc. C kế thừa trực tiếp lớp TickNet
và block FR-PDP gốc. L là biến thể phát triển có cơ sở từ TickNet-Basic, nhưng
đã sửa cấu trúc bên trong block; không nên gọi L là bản FR-PDP nguyên bản.**

Chưa phát hiện lỗi luồng tính toán, nhánh residual, kích thước tensor hoặc
gradient của cấu hình L hiện tại. Thiết kế giảm chi phí tầng sớm để bổ sung
block tầng sau có lý do tính toán rõ ràng. Kết quả L là bằng chứng thực nghiệm
ban đầu đáng quan tâm; chưa đủ để khẳng định từng thay đổi kiến trúc gây ra
mức tăng accuracy hoặc L luôn tốt hơn Basic/C.

**Đã đánh giá lại cả sáu checkpoint trên test local: Top-1, Macro F1 và toàn
bộ confusion matrix khớp báo cáo.** Kiểm định McNemar hai phía L–Basic cho
p = 0,3323 ở Mid32 và p = 0,1153 ở Mid224; chưa đạt ngưỡng 0,05. Đây không
phải bằng chứng hai mô hình tương đương, mà là chưa đủ bằng chứng cho khác
biệt trên tập test nhỏ này.

Một kết luận cũ cần sửa: **danh sách train/test của L đã xác minh khớp
Basic/C**, bằng cách tái tạo manifest theo đúng định dạng notebook L và đối
chiếu SHA-256. Khác biệt hash giữa hai định dạng CSV không phải khác split.

Vấn đề tái lập đáng chú ý nằm ở notebook Basic/C Mid224: chúng yêu cầu
`--resume` và `--stop-after-epoch`, trong khi `train_mid.py` đang lưu trong
repo không có hai tùy chọn đó. Do vậy chưa thể tái lập quy trình 100+100 epoch
chỉ với mã nguồn hiện tại. Đây là thiếu sót đóng gói mã huấn luyện, không tự
động chứng minh checkpoint hay mô hình bị sai.

## 2. Kết quả đã đạt được

Mỗi lượt huấn luyện dùng 25.000 ảnh train, 250 ảnh test, 5 lớp cân bằng,
200 epoch, seed 42, SGD, learning rate 0,1, momentum 0,9, weight decay 1e-4,
CosineAnnealingLR. Số liệu gốc nằm ở `training_logs/` và báo cáo từng mô hình.

| Mô hình | Dataset | Top-1 | Đúng / 250 | Macro F1 | Tham số | GFLOPs |
|---|---|---:|---:|---:|---:|---:|
| Basic | Mid32 | 89,6% | 224 | 0,89560 | 1.062.223 | 0,158428 |
| C | Mid32 | 91,2% | 228 | 0,91171 | 5.155.467 | 0,256830 |
| L | Mid32 | 91,6% | 229 | 0,91575 | 1.096.260 | 0,157821 |
| Basic | Mid224 | 92,4% | 231 | 0,92396 | 1.062.223 | 0,988343 |
| C | Mid224 | 94,4% | 236 | 0,94395 | 5.155.467 | 0,821054 |
| L | Mid224 | 95,6% | 239 | 0,95588 | 1.096.260 | 0,796760 |

L hơn Basic 5 ảnh đúng ở Mid32 và 8 ảnh đúng ở Mid224; hơn C lần lượt 1 và
3 ảnh. Mỗi ảnh tương ứng 0,4 điểm phần trăm. L tăng khoảng 3,2% tham số so
với Basic, giảm 19,4% FLOPs ở Mid224 và chỉ giảm 0,38% ở Mid32.

C tăng số tham số lên khoảng 4,85 lần Basic. Việc C vẫn dùng ít FLOPs hơn
Basic ở Mid224 không mâu thuẫn: C giảm spatial sớm hơn, đưa phần lớn dung
lượng vào các tầng có bản đồ đặc trưng nhỏ. Ở Mid32, FLOPs của C lại cao hơn.

FLOPs tính Conv2d và Linear với **1 MAC = 2 FLOPs**, một ảnh, chế độ eval;
không tính BN, activation, pooling, phép cộng residual và nhân scale SE.
Đây không phải phép đo latency, VRAM hay tổng mọi phép toán.

L có Top-1/F1 tốt nhất trong sáu lượt này, nhưng không tốt nhất theo mọi
metric: test loss Mid224 của C là **0,16260**, thấp hơn L **0,16667**. Top-1
và cross-entropy đo các khía cạnh khác nhau; điều này không phải lỗi tính toán.

## 3. Code Kaggle có giữ logic mô hình gốc không?

### 3.1. Basic: giữ nguyên kiến trúc, điều chỉnh bài toán và quy trình train

Đối chiếu Git xác nhận ba file `models/TickNet.py`, `models/common.py` và
`models/SE_Attention.py` giống nguyên bản tại `upstream/main`, commit
`e71679978f861a599192ece7613dc7ad3768e55b`.
Nguồn tham chiếu: [repository TickNets của tác giả](https://github.com/nttbdrk25/TickNets).

`models/mid_models.py` gọi `build_TickNet(..., typesize='basic', cifar=...)`.
Luồng block gốc ở `models/TickNet.py:33` là:

```text
x → PW1 tuyến tính → DW 3×3 + BN + ReLU → PW2 + BN + ReLU → SE
  → cộng shortcut identity hoặc PW projection
```

Stem 32 kênh, lịch stage 128 → 64 → 128 → 256 → 512, head 1024,
global average pooling và classifier được giữ. Đầu ra đổi thành 5 lớp;
Mid32 chọn đúng lịch stride CIFAR có sẵn trong factory gốc.

Tuy nhiên, đây là **tái sử dụng kiến trúc gốc cho bài toán Mid**, không phải
tái lập toàn bộ thí nghiệm của bài báo. Ví dụ `TickNet_Dogs.py` dùng
MultiStepLR và lịch giảm learning rate [100, 150, 180], còn `train_mid.py`
dùng cosine; dataset, augmentation và chính sách đánh giá cũng đã đổi.
Không nên trình bày rằng recipe huấn luyện bài báo được giữ nguyên.

### 3.2. C: kế thừa trực tiếp, thay cấu hình backbone

`models/ticknet_c.py:16` dựng trực tiếp `TickNet`. C có 9 `FR_PDP_block`
gốc; giữ SE, stem 32, head 1024 và thứ tự phép toán. Thay đổi nằm ở:

- Lịch kênh: 80 / 48 / (96,128) / (192,224) / (640,768,896).
- Số block: 1 / 1 / 2 / 2 / 3.
- Mid224 dùng lịch stride (2,1,2,2,2) đã có ở TickNet-small/large.

C thực sự thuộc họ TickNet. So sánh C–Basic là so sánh hai cấu hình kiến
trúc với recipe ghi nhận tương ứng, nhưng cũng chưa phải ablation một yếu tố:
C đổi đồng thời kênh, độ sâu và phân bổ stride.

### 3.3. L: biến thể PDP có nén kênh và mixed depthwise

`models/ticknet_l.py:47` triển khai block mới, vẫn giữ PW–DW–PW, SE và
shortcut qua mọi block. Lịch output có đỉnh sớm rồi đáy trước khi tăng dần:
24 → 112 → 64 → 144 → 144 → 288 → 288 → 512 → 768.

| Thành phần | Basic | L |
|---|---|---|
| Stem | 32 | 24 |
| Số block | 5 | 7 |
| PW1 | Cin → Cin, tuyến tính | Cin → hidden, tuyến tính |
| hidden | Cin | Khoảng 0,75 Cin từ stage 2, làm tròn bội 8 |
| DW | 3×3 | Stage 1–2: 3×3; stage 3–5: chia kênh cho 3×3 và 5×5 |
| SE | Có sẵn | Kế thừa |
| Shortcut | Identity / projection | Identity / projection |
| Head | 1024 | 768 |

Hai nhánh DW của L xử lý **hai nhóm kênh khác nhau**, rồi concatenate;
không phải mỗi nhánh nhận toàn bộ tensor. Mỗi convolution có groups bằng
số kênh của nhánh, giữ tính depthwise. PW2 trộn lại thông tin giữa các nhóm.
Stride và padding tạo cùng spatial để nối tensor và cộng shortcut.

L không phải TickNet-**large** gốc: `--model l` và `--model large` là hai
lựa chọn khác nhau. Nên dùng tên **TickNet-L v1, biến thể đề xuất từ
TickNet-Basic**, tránh nhập nhằng chữ L.

## 4. TickNet-L có sai về triển khai hoặc lập luận không?

### 4.1. Chưa thấy lỗi trong cấu hình đang dùng

Đã chạy 32 kiểm thử mô hình/pipeline/dataset: **32 passed**. Các kiểm thử
bao gồm forward/backward ở 32 và 224, gradient hữu hạn cho mọi tham số L,
đếm FLOPs độc lập, lưu/nạp checkpoint và kiểm tra phiên bản kiến trúc.

Kiểm tra bổ sung: đặt `hidden=Cin`, `kernels=(3,)`, copy trọng số từ block
gốc sang block L. Với (Cin,Cout,stride) = (32,32,1), (32,64,1), (32,64,2),
(32,32,2), sai số tuyệt đối lớn nhất đều **0**. Kết quả lưu trong
[`block_equivalence.json`](audit_20261001/block_equivalence.json).
Đây là kiểm tra trường hợp thu về block gốc, không có nghĩa cấu hình L đầy
đủ cho đầu ra giống Basic.

Shortcut của L vẫn nhận tensor đầu vào đầy đủ, không nhận tensor đã nén.
Các block cùng shape dùng identity. SE nằm trước phép cộng như code gốc.
Không có softmax thừa trước CrossEntropyLoss. Test dùng `eval()` và không
augment. Chưa thấy lỗi làm mất residual hoặc làm tăng accuracy giả từ những
đường tính toán này.

### 4.2. Phương pháp thiết kế có cơ sở, nhưng chưa tách được đóng góp

Ở Basic Mid224, stage 2 dùng 0,52103 GFLOPs, khoảng 52,7% toàn mạng.
PW1 diễn ra trước DW giảm kích thước, nên rất tốn chi phí trên 112×112.
L giảm kênh đầu vào stage này từ 128 xuống 112 và hidden xuống 88, đưa chi
phí stage 2 còn 0,33252 GFLOPs. Điều này giải thích hợp lý ngân sách để
thêm block tại stage 3–4 và tăng kernel cho một phần kênh.

Mixed depthwise dựa trên ý tưởng đã có trong
[MixConv, Tan và Le (2019)](https://arxiv.org/abs/1907.09595).
SE cũng đã có trong TickNet gốc. Đóng góp có thể bảo vệ là **cách cấu hình
và phân bổ tài nguyên của biến thể L**, không phải phát minh mới SE, residual
hay mixed depthwise. Kết quả MixConv trên mạng/dataset khác không tự chứng
minh hiệu quả của thành phần đó trong L.

Nén kênh có thể làm mất thông tin; thêm độ sâu và kernel lớn có thể cải
thiện biểu diễn nhưng không bảo đảm tăng accuracy. Tỷ lệ 0,75 và các số kênh
cụ thể mới được hỗ trợ bởi lý do ngân sách tính toán, chưa được chứng minh
là tối ưu. Sau làm tròn, hidden/Cin không luôn đúng 0,75, ví dụ 88/112;
đây là hành vi có chủ đích của công thức, không phải lỗi.

### 4.3. Hạn chế của bằng chứng hiện tại

- Chỉ có một seed. Chưa đo mean/std giữa các lượt train.
- L Mid32 dùng batch 128; Basic/C dùng 64. Với 25.000 ảnh và `drop_last=False`,
  L có 196 bước/epoch, Basic/C có 391; tổng 39.200 so với 78.200 cập nhật.
  Cùng 200 epoch không phải cùng số cập nhật SGD; thống kê BN cũng khác.
- L dùng 2 workers, Basic/C dùng 0. Điều này không tự làm phép thử vô hiệu,
  nhưng có thể thay đổi chuỗi augmentation ngay cả với cùng seed.
- Cùng seed không bảo đảm cùng khởi tạo hoặc cùng augmentation giữa các
  kiến trúc, đặc biệt khi RNG dùng chung với khởi tạo và augmentation.
- Không có validation riêng trong các log hiện tại. Các đường học là train
  loss/top1, không phải validation curve. Train gần 100% trong cả ba mô
  hình không chứng minh mô hình sai, nhưng không đủ đánh giá overfitting
  theo epoch hoặc chọn điểm dừng tối ưu.
- Chưa có ablation: nén kênh, lịch kênh, số block, mixed kernel và head đều
  thay đổi cùng lúc. Không thể nói riêng kernel 5×5 gây ra +3,2 điểm.
- Chỉ 250 ảnh test. Chênh lệch L–C Mid32 bằng đúng một ảnh; cần đặc biệt
  tránh kết luận L vượt trội ổn định từ mức chênh này.

### 4.4. Kiểm định dựa trên dự đoán từng ảnh

Sau khi xác minh membership, audit đã chạy lại mỗi checkpoint trên cùng
danh sách test và lưu sáu file `*_predictions.csv`. McNemar exact hai phía
dùng số ảnh mà hai mô hình bất đồng về đúng/sai, theo định nghĩa
[kiểm định McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).
Đây là phân tích hậu nghiệm cho các checkpoint đã có, không phải kết quả
từ một thí nghiệm nhiều seed được thiết kế trước.

| Dataset | Cặp A → B | A đúng, B sai | A sai, B đúng | p hai phía |
|---|---|---:|---:|---:|
| Mid32 | Basic → L | 6 | 11 | 0,3323 |
| Mid224 | Basic → L | 6 | 14 | 0,1153 |
| Mid32 | C → L | 10 | 11 | 1,0000 |
| Mid224 | C → L | 5 | 8 | 0,5811 |
| Mid32 | Basic → C | 12 | 16 | 0,5716 |
| Mid224 | Basic → C | 8 | 13 | 0,3833 |

Không cặp nào đạt p < 0,05 ngay cả trước hiệu chỉnh nhiều so sánh. Do đó
không nên viết “đã chứng minh L vượt trội có ý nghĩa thống kê”. Kiểm định
này cũng không loại trừ khả năng L có lợi ích thực: số mẫu nhỏ làm hạn chế
khả năng phát hiện khác biệt; p-value không đo xác suất mô hình tốt hơn.
Giả định độc lập giữa mẫu và việc chỉ có một lượt train vẫn là giới hạn.

Khoảng Wilson 95% tham khảo cho accuracy L: **87,50–94,44%** ở Mid32 và
**92,29–97,53%** ở Mid224. Đây là xấp xỉ theo ảnh cho một checkpoint, không
bao gồm biến thiên do seed và không thay thế kiểm định ghép cặp; mẫu test
còn được lấy cân bằng cố định theo lớp.

Ở Mid32, L vẫn nhầm 9/50 ảnh dog thành cat; recall dog chỉ 80%, trong khi
cat đạt 98%. Vì vậy nên bàn thêm chất lượng phân biệt dog–cat thay vì chỉ
báo accuracy tổng. Ở Mid224, L có recall dog 92%, cat 92%, bird 96%, frog
98% và horse 100%; horse 100% ở đây chỉ tương ứng 50 mẫu test.

## 5. Tái lập Kaggle: các khoảng trống cần khắc phục

**Mức ưu tiên cao — thiếu phiên bản trainer cho Basic/C Mid224.** Các notebook
`*-epoch001-100-notebook.ipynb` truyền `--stop-after-epoch 100`; các notebook
`*-epoch101-200-notebook.ipynb` truyền `--resume`. Parser hiện tại không hỗ trợ,
và checkpoint do trainer hiện tại tạo không lưu RNG/DataLoader state phục
vụ resume chính xác. Các checkpoint nhập về có thể đến từ bản trainer trong
Kaggle Input; cần lưu đúng bản đó cùng commit hoặc checksum. Không suy ra
resume đúng hoàn toàn chỉ từ việc đủ 200 dòng log và lịch learning rate đúng.

**Mức ưu tiên vừa — L clone tên nhánh thay vì commit cố định.** Cell 1 của
`Kaggle_TickNet_L_Midterm_Final.ipynb` dùng `--branch feature/giua-ki-model-l`
và bỏ qua clone nếu thư mục `.git` đã tồn tại. Do đó lần chạy mới có thể dùng
code khác, hoặc code cũ đang nằm sẵn trong working directory. Cần pin SHA
và lưu revision nguồn thực tế vào artifact.

**Mức ưu tiên vừa — schema manifest làm CLI đánh giá lại L bị từ chối.**
`train_mid.py:103` so sánh hash CSV thô. Vì vậy `--evaluate` checkpoint L
với `--data-root data` sẽ bị từ chối dù membership đã xác minh tương đương.
Nên chuẩn hóa schema hoặc lưu thêm fingerprint membership chung. Không sửa
hash trong checkpoint để che khác biệt; cần giữ nguyên provenance gốc.

**Mức ưu tiên vừa — artifact L không đóng gói manifest.** Cell cuối chỉ ZIP
`runs` và `midterm_report_assets`; manifest ở `midterm_data` không được đưa
vào ZIP. Tái dựng được membership trong lần audit này không thay thế nhu cầu
lưu manifest, dataset version và checksum nội dung ảnh trong các lần sau.

**Giới hạn kiểm chứng dữ liệu:** manifest L chỉ gồm tên file, không chứa hash
ảnh. Hash khớp xác minh membership; không chứng minh byte ảnh Kaggle lịch sử
giống byte ảnh local. Notebook L cũng chỉ kiểm tra số lượng trước khi train,
không xác minh đầy đủ checksum hay cặp Mid32/Mid224 theo nội dung.

**Tài liệu còn lệch thời điểm:** README gốc và `docs/MODEL_L.md`,
`docs/MODEL_C.md` vẫn có các câu “chưa huấn luyện đầy đủ”. Đó là trạng thái
lúc thiết kế, đã lỗi thời so với sáu checkpoint epoch 200 hiện có. Báo cáo L
cũ còn nói không lưu checkpoint; đường dẫn này đã được cập nhật trong lần
audit. Khi viết báo cáo nộp, dùng số liệu và giới hạn tại tài liệu này.

## 6. Nên trình bày kết quả và bổ sung thí nghiệm như thế nào?

Cách phát biểu phù hợp:

> TickNet-L v1 phát triển từ TickNet-Basic bằng nén kênh trung gian, phân bổ
> lại độ sâu/kênh và mixed depthwise 3×3–5×5. Với seed 42 trên split đã ghi
> nhận, L đạt 91,6% Mid32 và 95,6% Mid224; chi phí Conv/Linear Mid224 giảm
> 19,4% so với Basic. Kết quả hỗ trợ tính khả thi và triển vọng của cấu hình;
> đóng góp riêng của từng thay đổi và độ ổn định giữa các seed chưa được xác lập.

Để củng cố kết luận, thứ tự công việc nên là:

1. Đóng gói đúng trainer chạy 100+100, pin commit và lưu manifest/checksum.
2. Tách validation cố định từ train; chốt lựa chọn và protocol trước đợt
   xác nhận mới. Các kết quả test hiện tại nên ghi là kết quả đã quan sát,
   tránh tiếp tục dùng chúng để chọn cấu hình rồi coi như test chưa từng xem.
3. Chạy Basic và L với cùng batch 64, workers, augmentation, SGD/cosine và
   ít nhất 3 seed định trước, tốt hơn là 5 nếu đủ tài nguyên; báo mean/std.
4. Ablation giữ mọi yếu tố khác cố định: L với toàn DW 3×3 so với L mixed;
   L với hidden=Cin so với nén; L giảm block lặp so với đủ block. Ghi cả
   accuracy, params, FLOPs vì ablation cũng có thể thay đổi ngân sách.
5. Nếu muốn tuyên bố chạy nhanh hơn, đo latency và bộ nhớ trên cùng thiết
   bị/batch, có warm-up. FLOPs thấp hơn chưa đủ cho kết luận tốc độ.

Theo `.doc/Midterm exam.docx`, L đáp ứng các ngưỡng số tham số và FLOPs theo
quy ước đã nêu, có thay đổi kiến trúc cụ thể và đã có kết quả trên hai tập.
Không có cơ sở từ repo để bảo đảm khác mọi nhóm khác hoặc khẳng định mức độ
mới tuyệt đối. Việc chấp nhận cách đếm FLOPs và đánh giá đóng góp thuộc
tiêu chí chấm của giảng viên.

## 7. Bằng chứng và giới hạn kiểm định

- [`verification.json`](audit_20261001/verification.json): sáu checkpoint
  đã xác minh SHA-256, nạp state_dict nghiêm ngặt vào code hiện tại, epoch
  cuối 200, scheduler T_max/last_epoch = 200, đủ 200 dòng epoch liên tục.
  Sai lệch learning rate so với cosine lý thuyết tối đa khoảng 4,86e-17.
  Top-1 và confusion matrix tính lại khớp hoàn toàn; Macro F1 khớp trong
  sai số float, chênh test loss tuyệt đối lớn nhất dưới 1,2e-7.
- Đã đối chiếu SHA-256 của **50.500 file ảnh local** với manifest chuẩn;
  tất cả khớp. Tái tạo CSV schema L từ membership chuẩn thu được đúng
  `35bcc76fde54e97d54f7825cdb34376a3732fd79aa829624ea4fc982b4590bdf`.
- Sáu phép đếm Conv/Linear khớp cả số đã báo cáo lẫn
  `torch.utils.flop_counter.FlopCounterMode`. Suy luận kiểm định dùng CPU,
  batch 16, PyTorch/torchvision ghi cụ thể trong `verification.json`.
- [`source_checks.json`](audit_20261001/source_checks.json): nguồn gốc các
  file gốc, tùy chọn CLI và kết quả kiểm thử liên quan.
- [`verify_evidence.py`](audit_20261001/verify_evidence.py): mã tính lại
  membership hash, kiểm tra checkpoint và suy luận trên test local.
- [`block_equivalence.json`](audit_20261001/block_equivalence.json): kiểm
  tra block L thu về cấu hình block gốc với cùng trọng số.

Lệnh chạy nhóm kiểm thử phù hợp trong môi trường `fresher`:

```powershell
python -m pytest -q tests/test_ticknet_l.py tests/test_ticknet_c.py tests/test_mid_pipeline.py tests/test_mid_dataset.py
python docs/results/audit_20261001/verify_evidence.py
```

Toàn bộ `pytest -q` chưa chạy đạt vì môi trường thiếu `scipy`, gây lỗi collect
`tests/test_cleaning.py`; không quy lỗi môi trường này thành lỗi kiến trúc L.
Audit này không huấn luyện lại 200 epoch và không sửa trọng số, mô hình hay
recipe huấn luyện.
