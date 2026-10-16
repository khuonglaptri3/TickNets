# Sơ đồ Kiến trúc Tương tác TickNets (Interactive Architecture Diagrams)

Thư mục này chứa các sơ đồ kiến trúc ụ , trực quan hóa chi tiết cấu trúc luồng tensor, các khối PDP (Pointwise-Depthwise-Pointwise), cơ chế nén kênh, Mixed Depthwise và phân bổ tham số / FLOPs của hai dòng mô hình trong đồ án.

Tất cả các sơ đồ đều là file HTML độc lập (self-contained), có thể mở trực tiếp bằng bất kỳ trình duyệt web nào (Chrome, Firefox, Safari, Edge) mà không cần cài đặt thêm server hay web framework.

---

## Danh mục các Sơ đồ Kiến trúc

| Mô hình                  | File Sơ đồ Tương tác (.html)                                                                                 | Cấu hình Đầu vào (.json)            | Đặc điểm Kiến trúc Trực quan hóa                                                                                                                          |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------- | :--------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **TickNet-Basic**    | [ticknet_basic.html](ticknet_basic.html)           | `ticknet_basic.architecture.json`      | 1.06M Params, 0.988 GFLOPs. Thể hiện rõ điểm nghẽn tính toán tại Stage 2 (52.7% FLOPs).                                                                  |
| **TickNet-L v1**     | [ticknet_l.html](ticknet_l.html)                   | `ticknet_l.architecture.json`          | 1.10M Params, 0.796 GFLOPs (-19.4%). Trực quan hóa cơ chế**Pointwise Bottleneck (0.75x)** và **Mixed DW (3x3 + 5x5)** mở rộng receptive field. |

---

## Hướng dẫn mở và tương tác trên Trình duyệt

Bạn có thể mở các file HTML bằng trình duyệt web theo các cách sau:

### Cách 1: Mở trực tiếp từ dòng lệnh (Ubuntu / Linux)

```bash
# Mở sơ đồ TickNet-Basic
xdg-open docs/03_model/architectures/ticknet_basic.html

# Mở sơ đồ TickNet-L v1
xdg-open docs/03_model/architectures/ticknet_l.html


```

### Cách 2: Kéo thả vào trình duyệt

Kéo trực tiếp file `.html` từ trình quản lý tệp (Nautilus/Files) và thả vào cửa sổ Chrome / Edge / Firefox.

---

## Tính năng tương tác trong mỗi Sơ đồ

1. **Phóng to / Thu nhỏ & Xoay chuyển (Pan & Zoom)**: Dùng chuột cuộn hoặc touchpad để xem chi tiết từng tầng bên trong block hoặc thu nhỏ để xem luồng mạng toàn cục.
2. **Chế độ Sáng / Tối (Light & Dark Theme)**: Nút chuyển đổi giao diện tích hợp sẵn ở góc sơ đồ, tối ưu hiển thị cho slide thuyết trình hoặc báo cáo kỹ thuật.
3. **Thanh Thông tin Phân bổ (Metadata Badge)**: Hiển thị kích thước tensor qua từng stage `(B, C, H, W)`, kích thước kernel, stride, tham số và FLOPs tương ứng.
4. **Kiểm định Chất lượng Showcase**: Tất cả các sơ đồ đều đạt chuẩn chất lượng nghiêm ngặt (9/9 tiêu chí kiểm định Archify passed, clearance nhãn $\ge 80$px, không tràn khung nhìn).
