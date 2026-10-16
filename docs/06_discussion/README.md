# 6. So sánh, ablation và hạn chế

## Flow so sánh cho báo cáo

1. Basic với L v1: giải thích thay đổi kiến trúc và chi phí.
2. L v1 baseline với từng kỹ thuật trong hai hướng bên dưới.
3. So sánh hiệu quả và FLOPs, phân tích theo lớp và lỗi dự đoán.
4. Trình bày quy tắc chọn cấu hình và hạn chế của kết quả.

| Hướng | Câu hỏi | Bằng chứng ưu tiên |
|---|---|---|
| Exp 1: smoothing/LR | Đổi smoothing hoặc LR có cải thiện validation không? | [Báo cáo](../05_results/experiment_midterm_smoothing_lr/README.md), [validation](../05_results/experiment_midterm_smoothing_lr/selection/validation_summary.csv), [test](../05_results/experiment_midterm_smoothing_lr/comparisons/test_metrics.csv) |
| Exp 2: stage 4 depth | Thêm block có đáng với chi phí tăng không? | [Báo cáo](../05_results/experiment_midterm_stage4_depth/EXPERIMENT_REPORT.md), [metrics](../05_results/experiment_midterm_stage4_depth/comparisons/metrics_summary.csv) |

Hai nhóm dùng cùng đối chứng và được so sánh độc lập theo validation. Mỗi recipe thay riêng smoothing, learning rate hoặc độ sâu stage 4.

## Hình so sánh

- [Validation của các cấu hình](../05_results/selected_checkpoints/figures/validation_comparison.png).
- [Đường học nhóm smoothing/LR trên Mid224](../05_results/experiment_midterm_smoothing_lr/comparisons/learning_curves_mid224.png).
- [Đường học nhóm smoothing/LR trên Mid32](../05_results/experiment_midterm_smoothing_lr/comparisons/learning_curves_mid32.png).
- [Confusion matrix hai checkpoint trình diễn](figures/fig3_winning_confusion_matrices.png).

[Báo cáo tổng hợp cũ](MIDTERM_FINAL_SELECTION_REPORT.md) được giữ để tham khảo hình và bố cục. Khi có khác biệt, ưu tiên số liệu CSV/JSON của run.

## Các điểm cần giải thích

- Trong riêng nhóm smoothing/LR, [selection_lock.json](../05_results/experiment_midterm_smoothing_lr/selection/selection_lock.json) chọn Mid224 baseline (val 93,88%) và Mid32 LR 0,15 (val 91,04%). Mid224 smoothing 0,10 có test 99,2% nhưng val 93,64%; không gọi nó là cấu hình thắng validation trong nhóm đó.
- Basic lịch sử có test 89,6% trên Mid32 và 92,4% trên Mid224 theo [reported_metrics.csv](../05_results/model_comparison/reported_metrics.csv). Bảng Basic trong báo cáo tổng hợp cũ có số khác.
- Có một seed cho mỗi recipe và 250 ảnh test; một ảnh tương ứng 0,4 điểm phần trăm. Đây là historical test split đã được quan sát trong quá trình phát triển.
- Train loss có smoothing và loss với nhãn cứng dùng các phân phối mục tiêu khác nhau; cần giải thích khi đối chiếu đường loss.
- Các script checkpoint trình diễn hai bản trọng số được ghim. Các bản này khác byte với bản trong hồ sơ smoothing/LR; biên bản sắp xếp giữ checksum của cả hai, không giả định chúng là bản sao nguyên byte.

Kiểm định lịch sử: [AUDIT_TICKNET_2026-10-01.md](../05_results/AUDIT_TICKNET_2026-10-01.md).
