**Phân tích kết quả và hướng phát triển TickNet cho cuối kỳ — 2026-10-01**

**Kết luận:** Tiếp tục phát triển họ TickNet-L, dùng L v1 làm mốc trên CIFAR, ưu tiên quy trình huấn luyện và khả năng khái quát hóa, sau đó thử tăng chiều sâu hoặc giảm mức nén kênh theo từng yếu tố. Chưa có cơ sở chọn ngay Coordinate Attention, dự báo accuracy CIFAR hoặc khẳng định L nhỏ sẽ luôn tốt hơn mạng lớn.

**1. Phạm vi và mức độ kiểm chứng**

Đã đọc các báo cáo, cấu hình, log 200 epoch, bảng so sánh, profile từng tầng, dự đoán từng ảnh, biên bản audit, mã nguồn L/trainer/dataloader và hai đề thi trong `.doc`.

Lần phân tích này tính lại Top-1, Macro F1, confusion matrix và McNemar Basic–L từ các CSV hiện có; kiểm tra SHA-256 của bốn checkpoint Basic/L cùng các bản sao trong thư mục từng mô hình. Kết quả chi tiết ở [FINAL_DEVELOPMENT_EVIDENCE_2026-10-01.json](FINAL_DEVELOPMENT_EVIDENCE_2026-10-01.json).

Đây là kiểm chứng artifact, không phải chạy suy luận hoặc huấn luyện mới. Python đang dùng chưa có torch/torchvision. Audit trước ghi nhận đã chạy suy luận sáu checkpoint; hiện thư mục này chỉ còn prediction CSV và checkpoint của Basic/L. Số liệu C được dẫn từ bảng so sánh và audit, chưa được kiểm chứng lại từ checkpoint trong lần này. Không tìm thấy artifact kết quả CIFAR trong `docs/results`.

**2. Kết quả hiện tại hỗ trợ điều gì?**

| Mô hình | Mid32 Top-1 | Mid224 Top-1 | Tham số, 5 lớp | GFLOPs Mid32 | GFLOPs Mid224 |
|---|---:|---:|---:|---:|---:|
| Basic | 89,6% | 92,4% | 1.062.223 | 0,158428 | 0,988343 |
| C | 91,2% | 94,4% | 5.155.467 | 0,256830 | 0,821054 |
| L v1 | 91,6% | 95,6% | 1.096.260 | 0,157821 | 0,796760 |

Nguồn: [reported_metrics.csv](model_comparison/reported_metrics.csv), [audit](AUDIT_TICKNET_2026-10-01.md), [verification.json](audit_20261001/verification.json).

L hơn Basic 2,0 điểm phần trăm ở Mid32 và 3,2 điểm ở Mid224; thêm khoảng 3,2% tham số. L giảm 19,4% FLOPs ở Mid224, nhưng chỉ giảm 0,38% ở Mid32. Vì cuối kỳ dùng ảnh 32×32, lợi ích tiết kiệm tính toán lớn ở Mid224 không chuyển nguyên vẹn sang CIFAR.

L là mốc phát triển hợp lý: kết quả quan sát tốt, dung lượng nhỏ, còn nhiều ngân sách và có kiến trúc kế thừa giữa kỳ để giải thích khi vấn đáp. Tuy nhiên:

- Tập test chỉ 250 ảnh; một ảnh tương ứng 0,4 điểm phần trăm. L hơn C ở Mid32 đúng một ảnh.
- Một seed 42 chưa phản ánh độ ổn định. McNemar Basic–L tính lại cho p=0,3323 ở Mid32 và p=0,1153 ở Mid224; chưa đạt 0,05. Điều này không chứng minh hai mô hình tương đương.
- L Mid32 dùng batch 128, Basic/C dùng 64; workers cũng khác. Cùng số epoch không đồng nghĩa cùng số lần cập nhật SGD.
- Nhiều yếu tố kiến trúc thay đổi đồng thời. Chưa thể quy mức tăng accuracy cho riêng mixed kernel, nén kênh hay thêm block.

**3. Điểm yếu cần ưu tiên**

| Chỉ số L v1 | Mid32 | Mid224 |
|---|---:|---:|
| Train Top-1 epoch 200 | 99,984% | 99,992% |
| Test Top-1 | 91,6% | 95,6% |
| Chênh lệch train–test | 8,384 điểm | 4,392 điểm |
| Tổng lỗi test | 21 | 11 |
| Cat ↔ dog | 10/21 lỗi | 7/11 lỗi |
| Bird ↔ frog | 8/21 lỗi | 2/11 lỗi |

Ở Mid32, dog recall chỉ 80%; 9/50 ảnh dog bị nhận là cat. Cat recall 98% nhưng precision chỉ 84,48% vì nhận thêm nhiều ảnh dog. Đây là mất cân đối chất lượng phân loại, dù tập dữ liệu cân bằng theo lớp. Tăng trọng số lớp dog chưa phải lựa chọn đầu tiên vì có thể đổi lỗi dog→cat thành cat→dog.

Train gần 100% trong khi test thấp hơn là dấu hiệu cần khảo sát khả năng khái quát hóa. Train được đo khi có augmentation và cập nhật trọng số, test dùng eval; chênh lệch này không phải phép đo overfitting thuần túy. Không có validation theo epoch nên chưa biết epoch tối ưu hoặc liệu kéo dài huấn luyện có lợi.

SE dùng pooling để tạo trọng số kênh rồi nhân lại với tensor không gian; mã nguồn `models/SE_Attention.py` không thay toàn bộ feature map bằng tensor 1×1. Vì vậy kết luận “SE làm mất chi tiết tai/râu nên gây nhầm chó–mèo” trong kế hoạch cũ chưa được chứng minh. CA có thể là ứng viên thử nghiệm, nhưng không phải cách sửa đã được xác nhận. Xem cơ chế trong [bài SE](https://arxiv.org/abs/1709.01507) và [Coordinate Attention](https://arxiv.org/abs/2103.02907).

**4. Yêu cầu cuối kỳ làm thay đổi ưu tiên**

Đọc trực tiếp [Final exam.docx](../../.doc/Final%20exam.docx): tham số ≤6M, FLOPs <1G; train/test trên CIFAR-10 và CIFAR-100 với nhiều learning rate và cả SGD/Adam. Điểm gồm CIFAR-10 30%, CIFAR-100 30%, vấn đáp 30%, báo cáo 10%; không qua vấn đáp sẽ trượt. Đề cho phép kế thừa mô hình đề xuất giữa kỳ.

Cả hai CIFAR đều là ảnh màu 32×32, có 50.000 ảnh train và 10.000 ảnh test; CIFAR-100 có 100 fine classes, 500 ảnh train mỗi lớp. Cần dùng fine labels cho bài toán 100 lớp. Nguồn: [trang CIFAR của tác giả](https://cave.cs.toronto.edu/kriz/cifar.html).

Độ chính xác Mid224 trên 5 lớp không dự báo được CIFAR-100 trên 100 lớp. Các khoảng dự báo CIFAR-10 93,5–95,2% và CIFAR-100 72–76,5% trong kế hoạch cũ chưa có thí nghiệm hỗ trợ; chỉ nên coi là mục tiêu mong muốn nếu nhóm muốn giữ chúng.

Vì 60% điểm đến từ performance, mô hình nhẹ nhất không tự động là lựa chọn tốt nhất. Tăng dung lượng trong giới hạn có thể có lợi, đặc biệt trên CIFAR-100; cũng có thể làm giảm khái quát hóa. Giữ một kiến trúc L chung cho hai dataset, thay classifier 10/100 lớp và chọn recipe riêng theo validation nếu cần.

**5. Hướng ưu tiên thứ nhất: xây dựng phép thử CIFAR và tối ưu huấn luyện**

Trainer hiện tại chỉ có SGD, Mid32/Mid224, ImageFolder năm lớp và không có validation hay resume. Chuyển sang CIFAR cần bổ sung thật các chức năng này; đổi riêng `num_classes` chưa đủ. Cần:

- Split cố định 45.000 train / 5.000 validation từ official train, phân tầng theo lớp; official test 10.000 ảnh giữ nguyên. CIFAR-100 tương ứng 450 train + 50 validation mỗi lớp.
- Tách seed split khỏi seed huấn luyện. Lưu index split, checksum dataset, commit code, config và phiên bản thư viện.
- Log train/validation loss và Top-1; lưu `best_val.pt` và `last.pt`. Chốt quy tắc chọn checkpoint trước khi chạy test. Nếu retrain trên toàn bộ 50.000 train, dùng số epoch đã chọn từ validation, không chọn lại bằng test.
- Resume đầy đủ model, optimizer, scheduler, epoch, RNG/sampler và AMP scaler nếu dùng. Chốt tổng lịch cosine, tránh khởi động lại lịch khi tiếp tục phiên Kaggle.
- Chuẩn hóa cùng batch/workers/augmentation/epoch cho so sánh kiến trúc. Batch 128 là điểm bắt đầu hợp lý cho CIFAR nếu GPU cho phép; đo thực tế rồi cố định cho cả hai optimizer và các ứng viên.
- Khởi tạo CIFAR từ đầu để có mốc rõ ràng. Pretrain giữa kỳ, nếu thử sau, phải ghi riêng nguồn dữ liệu và kiểm tra điều kiện chấm.

Lưới learning rate khởi đầu đề xuất, chưa phải giá trị đã tối ưu:

| Optimizer | Initial LR | Các thiết lập cần ghi rõ |
|---|---|---|
| SGD | 0,05; 0,10; 0,15 | momentum 0,9; WD khởi đầu 1e-4; Nesterov bật/tắt cố định |
| Adam | 3e-4; 1e-3; 3e-3 | betas (0,9;0,999); eps 1e-8; WD khởi đầu 1e-4 |

Đề dùng 0,1 và 0,15 làm ví dụ, không ghi bắt buộc mọi optimizer dùng cùng LR. Adam có mặc định LR=0,001 trong [tài liệu PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html); việc chọn dải nhỏ hơn cho Adam ở đây là đề xuất thử nghiệm, không phải bằng chứng nó tối ưu cho L. Nếu giảng viên quy định một lưới chung, thêm các giá trị đó và báo trung thực kết quả bất ổn. Cần có thí nghiệm Adam đúng yêu cầu, ghi riêng nếu thử AdamW.

Giữ 200 epoch làm mốc ban đầu, thử warmup 5 epoch + cosine như một thay đổi riêng. Kiểm tra 1–5 epoch trước mỗi recipe mới để phát hiện lỗi dataset/loss hoặc mất ổn định; các lượt ngắn không đủ để chọn cấu hình cuối. Chỉ thử 300 epoch khi đường validation còn cải thiện.

Giữ random crop padding 4 + horizontal flip làm mốc augmentation. So sánh lần lượt Mixup (alpha=0,2), CutMix (alpha=1,0, xác suất áp dụng khởi đầu 0,5), và label smoothing 0,05/0,1. Mỗi lần đổi một yếu tố; chỉ kết hợp những lựa chọn có lợi trên validation. Mixup/CutMix đã có kết quả tích cực trên CIFAR trong [bài Mixup](https://arxiv.org/abs/1710.09412) và [bài CutMix](https://arxiv.org/abs/1905.04899); hiệu quả trên TickNet-L vẫn cần đo. Nhãn trộn phải được xử lý đúng trong loss; train accuracy dưới Mixup không so trực tiếp với train accuracy nhãn cứng.

**6. Hướng ưu tiên thứ hai: tăng dung lượng L có kiểm soát**

L v1 cho CIFAR dùng `cifar=True`, stem stride 1 và stage strides (1,1,2,2,2), tạo spatial 32→32→32→16→8→4. Không dùng lịch stride Mid224 hoặc upscale CIFAR lên 224 để dựa vào kết quả cũ.

Classifier có 768 input channels và bias. Từ profile năm lớp, số tham số tăng `769 × (K−5)`, FLOPs tăng `1536 × (K−5)` khi đổi sang K lớp. Ngân sách tính toán theo mã nguồn/profile hiện tại:

| Ứng viên, đầu vào 32×32 | CIFAR-10 params | CIFAR-100 params | GFLOPs CIFAR-100 |
|---|---:|---:|---:|
| L v1, chỉ đổi classifier | 1.100.105 | 1.169.315 | 0,157967 |
| L + một block lặp Stage 4 | 1.239.875 | 1.309.085 | 0,174383 |

Các số này là phép tính giải tích dựa trên cấu trúc hiện có, không phải profile mới qua PyTorch hay kết quả train. Quy ước: 1 MAC=2 FLOPs, chỉ Conv2d/Linear, batch 1, eval. Phải profile lại implementation cuối với classifier 100 lớp và kiểm tra quy ước giảng viên sử dụng.

Block Stage 4 bổ sung là cùng loại với unit2 hiện tại: Cin=Cout=288, hidden=216, DW (3,5), shortcut identity. Nó thêm 139.770 tham số và 0,016416 GFLOPs ở 32×32. Ở nhánh Mid224, cùng block thêm khoảng 0,050231 GFLOPs, thay vì ước lượng 0,08G trong kế hoạch cũ.

Thứ tự ứng viên nên thử:

1. **A: Stage 4 depth 2→3**, giữ mọi yếu tố còn lại. Đây là thay đổi nhỏ, ngân sách đã tính được và dễ bảo vệ. Chưa có bằng chứng tăng accuracy.
2. **B: hidden ratio Stage 3–5 từ 0,75→1,0**, giữ stem/stride/kênh output/depth/SE. Kiểm tra liệu nén kênh quá mạnh có giới hạn biểu diễn khi phân biệt 100 lớp.
3. **C: tăng đồng bộ width Stage 3–4**, chẳng hạn 144/288→192/384, giữ recipe và depth. Profile lại, chọn bằng validation. Tăng width có cơ sở nghiên cứu trong [Wide Residual Networks](https://arxiv.org/abs/1605.07146), nhưng không bảo đảm hiệu quả trên L.
4. **D: SE→CA**, chỉ thử sau các đối chứng trên; giữ các yếu tố khác cố định. Không tích hợp CA đồng thời thêm depth trong lượt đầu vì sẽ không biết yếu tố nào có ích.

A/B/C đều còn là giả thuyết. Chỉ ghép các thay đổi sau khi từng đối chứng có lợi, rồi xác nhận cấu hình ghép. Ưu tiên khảo sát CIFAR-100 vì bài toán nhiều lớp giúp kiểm tra giới hạn biểu diễn; xác nhận ứng viên tốt nhất trên CIFAR-10 trước khi chọn một kiến trúc chung.

Giữ tham số/FLOPs cách trần để dự phòng khác biệt công cụ; không cần dùng hết 6M/1G. FLOPs thấp không chứng minh train nhanh hơn: mixed DW, concatenation, memory traffic và kernel GPU ảnh hưởng latency. Đo thời gian epoch, throughput và VRAM trên cùng phần cứng trước khi quyết định chi phí grid search.

**7. Lộ trình và tiêu chí chốt kết quả**

| Giai đoạn | Đầu ra cần đạt | Quy tắc ra quyết định |
|---|---|---|
| Chuẩn hóa | CIFAR loaders, validation, SGD/Adam, resume, profile, log | Shape/label đúng; không dùng test để chọn |
| Mốc CIFAR | L v1, hai optimizer, nhiều LR trên cả hai dataset | Chọn bằng validation; ghi cả cấu hình kém |
| Cải thiện recipe | Augmentation/WD/warmup theo từng yếu tố | Chỉ giữ lợi ích validation có thể tái lập |
| Ablation kiến trúc | A/B/C; D nếu đủ tài nguyên | Cùng recipe/split; đo accuracy và chi phí |
| Xác nhận | Cấu hình chốt trên cả CIFAR, seeds 42/43/44 | Báo mean±std; không chọn seed tốt nhất bằng test |
| Nộp bài | PDF, checkpoint/config/code tái lập, chuẩn bị vấn đáp | Giải thích được từng thay đổi và giới hạn |

Lưới đầu tiên gồm 2 dataset × 2 optimizer × 3 LR = 12 lượt nếu chạy đầy đủ. Ước lượng tài nguyên từ lượt pilot trước khi mở rộng; không chạy toàn bộ tổ hợp LR×optimizer×architecture×augmentation cùng lúc. Khi thiếu thời gian, ưu tiên L v1 với protocol tốt và ứng viên A, rồi xác nhận nhiều seed; mở rộng kiến trúc chỉ khi còn ngân sách.

Chốt theo validation Top-1 trên cả hai dataset, đặc biệt tránh cải thiện CIFAR-10 bằng cách làm CIFAR-100 giảm đáng kể. Báo Macro F1/recall lớp yếu để giải thích, nhưng không dùng một metric phụ thay thế tiêu chí performance của đề. Test cuối báo cho cấu hình/seed đã định trước, không chọn kết quả test tốt nhất trong grid.

Không đủ dữ liệu để bảo đảm một mức accuracy hoặc vị trí xếp hạng lớp. Mục tiêu có thể kiểm chứng là cải thiện so với L v1 trên CIFAR dưới cùng protocol, lợi ích lặp lại qua seed, và vẫn đạt giới hạn tài nguyên.

**8. Những diễn giải cần sửa trước khi dùng báo cáo cũ**

- `Model_Basic/README.md` mô tả một số recall/F1 không khớp CSV. Ví dụ Basic Mid32 cat recall=94%, dog=78%; Mid224 bird=94%, cat=90%, dog=86%. Khi viết PDF, lấy CSV đã đối chiếu thay cho số trong phần văn xuôi.
- `Model_L/README.md` ghi Mid32 bird F1≈0,949 và frog≈0,969; CSV đúng là 0,900 và 0,913. Mid224 horse recall=100%, nhưng precision≈98,04% và F1≈0,990, vì một ảnh bird bị nhận là horse.
- Nhận định “L chắc chắn tổng quát hóa tốt hơn C trên CIFAR-100 vì ít tham số” và “train nhanh hơn đáng kể vì FLOPs thấp” chưa có phép đo hỗ trợ.
- Kế hoạch `docs/superpowers/plans/2026-10-01-model-l-optimization-plan.md` đưa dự báo accuracy, nguyên nhân lỗi và chi phí CA chưa được xác nhận. Cần trình bày chúng như giả thuyết, không phải kết luận thực nghiệm.
- Tên L là biến thể đề xuất `ticknet-l-v1`, không phải TickNet-large gốc. SE/residual/mixed depthwise là ý tưởng đã có; đóng góp cần bảo vệ nằm ở cấu hình/phân bổ tài nguyên và bằng chứng ablation trên bài toán này.

Các nguồn cũ được giữ để đối chiếu; tài liệu này ghi rõ số liệu gốc nào đáng tin và phần nào cần thí nghiệm bổ sung.
