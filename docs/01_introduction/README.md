# 1. Bài toán và yêu cầu

Nguồn đề: [Midterm exam.docx](<../../.doc/Midterm exam.docx>).

Báo cáo PDF cần mô tả việc xây dựng hai bộ ảnh, thiết kế mô hình nhẹ L, huấn luyện và kiểm tra mô hình trên cả Mid224 và Mid32.

| Nội dung | Yêu cầu | Trọng số |
|---|---|---:|
| Dữ liệu | Ít nhất 5 lớp; khoảng 5.000 train và 50 test/lớp; Mid224 224×224 ≤25M, Mid32 32×32 ≤20M | 20% |
| Thiết kế L | Tham số học được ≤6M; FLOPs <1G; trình bày khác biệt kiến trúc | 30% |
| Hiệu năng Mid224 | Top-1 và kết quả thực nghiệm | 20% |
| Hiệu năng Mid32 | Top-1 và kết quả thực nghiệm | 20% |
| Báo cáo | Nội dung đầy đủ, số liệu rõ ràng và hình/bảng có giải thích | 10% |

Đề còn yêu cầu không trùng kiến trúc hoặc dataset với nhóm khác. Mô hình đề xuất cần có khác biệt với mạng có sẵn.

Động lực thiết kế L và mốc TickNet-Basic nằm trong [MODEL_L.md](../03_model/MODEL_L.md). Đi tiếp đến [dữ liệu](../02_dataset/README.md).
