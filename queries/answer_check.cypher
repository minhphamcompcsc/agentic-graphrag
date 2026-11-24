// answer_check.cypher: kiểm đáp án bộ yêu cầu đánh giá bằng metadata trong KG (24/11/2026)
// Chạy trong Neo4j Browser (http://localhost:7474), mỗi lần 1 truy vấn.
// Kết quả đã chạy ngày 24/11/2026: data/eval/answer_check_20261124/
// Không dùng lớp ngữ nghĩa (Function, INVOLVES) làm đáp án: đó là đầu ra của hệ thống, dùng sẽ bị vòng lặp.

// ---------------------------------------------------------------
// Q-A. Model và field mỗi module thêm vào (ADDS_FIELD)
// Kiểm tra nhanh: stock_no_negative phải có allow_negative_stock trên 4 model (R-07)
// Kết quả: QA_adds_field.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[:ADDS_FIELD]->(f:Field)
RETURN mod AS module, f.model AS model, count(f) AS so_field, collect(f.name) AS fields
ORDER BY module, model;

// ---------------------------------------------------------------
// Q-B. Model mỗi module định nghĩa (DEFINES) hoặc mở rộng (EXTENDS)
// transient = true là wizard
// Kết quả: QB_defines_extends.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[r:DEFINES|EXTENDS]->(mo:Model)
RETURN mod AS module, type(r) AS quan_he, mo.name AS model, mo.transient AS transient
ORDER BY module, quan_he, model;

// ---------------------------------------------------------------
// Q-C. Module cần cài kèm trực tiếp (DEPENDS_ON 1 bước)
// Kết quả: QC_depends_on.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[:DEPENDS_ON]->(d:Module)
RETURN mod AS module, collect(d.name) AS phu_thuoc_truc_tiep
ORDER BY module;

// ---------------------------------------------------------------
// Q-D. Câu 6: Odoo chuẩn đã có liên kết đơn bán -> phiếu giao, hoá đơn chưa?
// Nếu module_them_field là module Odoo gốc (sale, sale_stock) thì câu 6 không cần module OCA
// Kết quả: QD_sale_order_links.csv
// ---------------------------------------------------------------
MATCH (f:Field {model: 'sale.order'})-[:RELATES_TO]->(t:Model)
WHERE t.name IN ['stock.picking', 'account.move']
OPTIONAL MATCH (owner:Module)-[:ADDS_FIELD]->(f)
RETURN f.name AS field, f.ttype AS kieu, t.name AS model_dich, owner.name AS module_them_field, owner.source AS nguon;
