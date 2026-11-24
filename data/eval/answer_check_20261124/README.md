# Kiểm đáp án bộ yêu cầu đánh giá bằng metadata (24/11/2026)

Minh chứng cho bước kiểm đáp án câu 1 đến 6 (kế hoạch D16, ngày 24/11).

## Trạng thái KG khi chạy

- Lớp cấu trúc: Extractor v0.1 (8.260 nút, 21.421 quan hệ), Odoo 18 Community, DB `kg_demo`, 66 module.
- Lớp ngữ nghĩa: sem-v0.2 (không dùng trong bước này).
- Commit mã nguồn gần nhất trước khi chạy: `e35ab73`.
- Chạy trong Neo4j Browser ngày 24/11/2026, export CSV.

## File

| File | Truy vấn | Nội dung |
|---|---|---|
| `QA_adds_field.csv` | Q-A | Model và field mỗi module thêm vào (ADDS_FIELD) |
| `QB_defines_extends.csv` | Q-B | Model mỗi module định nghĩa (DEFINES) hoặc mở rộng (EXTENDS) |
| `QC_depends_on.csv` | Q-C | Module phụ thuộc trực tiếp (DEPENDS_ON 1 bước) |
| `QD_sale_order_links.csv` | Q-D | Field nối `sale.order` với `stock.picking`, `account.move` và module thêm field |

Truy vấn: `queries/answer_check.cypher`. Đáp án nháp: `data/eval/requirements_draft_20261124.csv`.

## Kết quả chính

1. Kiểm tra nhanh đạt: `stock_no_negative` thêm `allow_negative_stock` trên 4 model, khớp R-07.
2. `stock_picking_back2draft` và `sale_stock_picking_invoice_link` không thêm field nào, chỉ EXTENDS model. Model đáp án của các module này phải lấy từ Q-B, không chỉ ADDS_FIELD.
3. `product.product` có field của `stock_no_negative`, `stock_picking_auto_create_lot` nhưng không bị EXTENDS: field có do `_inherits` từ `product.template`. Xếp vào "chấp nhận được".
4. `stock_no_negative` EXTENDS `stock.quant` (chưa có trong đáp án câu 1 theo R-07). Chờ BA chốt.
5. `account.bank.statement.line` có `picking_ids` do `_inherits` `account.move` (R-20): loại khỏi đáp án câu 2.
6. Phụ thuộc trực tiếp: `stock_picking_invoice_link` -> `stock_account` (khớp R-21); 4 module khác -> `stock`; `sale_stock_picking_invoice_link` -> `sale_stock`, `stock_picking_invoice_link`.
7. Câu 6: Odoo chuẩn đã có `sale.order.picking_ids` (module `sale_stock`) và `sale.order.invoice_ids` (module `sale`). Chờ BA chốt hướng câu 6.

## Quyết định của BA (24/11/2026)

- Câu 1: thêm `stock.quant` vào model đáp án (nơi đặt logic chặn tồn kho âm); `product.product` chuyển sang "chấp nhận được". Đây là thay đổi so với R-07 (BC1111: 4 model theo ADDS_FIELD).
- Câu 6: viết lại theo chức năng của `sale_stock_picking_invoice_link` (câu cũ Odoo chuẩn đã đáp ứng, xem Q-D). Nhóm câu "Odoo chuẩn đã đáp ứng" để Thực tập 2, cần độ đo riêng.
- Đáp án chính thức: `data/eval/requirements_v0.csv` (bản nháp `requirements_draft_20261124.csv` giữ lại làm lịch sử).

## Quy tắc chọn đáp án

- Model đáp án là nơi đặt cấu hình hoặc logic chính của chức năng (ADDS_FIELD, DEFINES hoặc EXTENDS).
- Model đáp án chỉ chọn trong danh sách Q-A, Q-B; không thêm model ngoài danh sách.
- Model có field chỉ do kế thừa uỷ quyền (`_inherits`) xếp vào "chấp nhận được" hoặc loại.
- Không dùng lớp ngữ nghĩa (Function, INVOLVES) làm đáp án.
- Cho phép "module chấp nhận được", "model chấp nhận được" cho trường hợp mơ hồ.
