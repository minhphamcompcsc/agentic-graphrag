// =====================================================================
// Phân tích Knowledge Graph ban đầu (Thực tập 1, chạy ngày 09/11/2026)
// Chạy trong Neo4j Browser: http://localhost:7474 (từng câu một)
// Yêu cầu: đã chạy extract_schema.py v0.1 (có sẵn cờ l.oca, l.technical)
// Kết quả ghi trong comment là số liệu thực tế đã đo, dùng cho báo cáo.
// =====================================================================


// ---------------------------------------------------------------------
// A. KIỂM TRA TÍNH ĐÚNG CỦA GRAPH
// ---------------------------------------------------------------------

// A1. Đếm node theo loại
// Kết quả: Module 66, Model 439, Field 7.755
MATCH (n) RETURN labels(n)[0] AS loai, count(*) AS so_luong;

// A2. Module OCA đã load
// Kết quả: 6 module (5 chọn tay + sale_stock_picking_invoice_link tự cài)
MATCH (m:Module {source:'oca'}) RETURN m.name, m.label ORDER BY m.name;

// A3. Ai tạo, ai mở rộng stock.picking (kiểm chứng heuristic "id nhỏ nhất")
// Kết quả: stock DEFINES; stock_sms, stock_account, sale_stock + 4 module OCA EXTENDS
MATCH (mo:Module)-[r:DEFINES|EXTENDS]->(md:Model {name:'stock.picking'})
RETURN mo.name, type(r), mo.source ORDER BY type(r);

// A4. Field do stock_no_negative thêm vào
// Kết quả: allow_negative_stock trên product.category, product.product,
//          product.template, stock.location
MATCH (mo:Module {name:'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
RETURN f.model, f.name, f.label, f.ttype;

// A5. Field không xác định được module sở hữu
// Kết quả: 5 field điện thoại trên res.users (kế thừa uỷ quyền từ res.partner)
MATCH (f:Field)
WHERE NOT EXISTS { (:Module)-[:ADDS_FIELD]->(f) }
RETURN f.model, f.name, f.ttype;

// A6. Chuỗi phụ thuộc của stock_picking_invoice_link (xem dạng Graph)
MATCH p = (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON*1..3]->(:Module)
RETURN p;


// ---------------------------------------------------------------------
// B. ẢNH HƯỞNG CỦA MODULE OCA LÊN CẤU TRÚC QUAN HỆ
// ---------------------------------------------------------------------

// B1. Số cạnh do OCA thêm vào
// Kết quả: 23 / 2.512 cạnh LINKS_TO (dưới 1%)
MATCH ()-[l:LINKS_TO {oca:true}]->() RETURN count(l) AS so_canh_oca;

// B2. Liên kết trực tiếp phiếu kho <-> hoá đơn
// Kết quả: 2 cạnh (invoice_ids, picking_ids), cả 2 đều oca = true
MATCH (a:Model {name:'stock.picking'})-[l:LINKS_TO]-(b:Model {name:'account.move'})
RETURN startNode(l).name AS tu, l.via AS qua_field, l.type, endNode(l).name AS den, l.oca;

// B3. Đường ngắn nhất CÓ OCA (hình fig_path_with_oca.svg)
// Kết quả: 1 bước
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..4]-(b:Model {name:'account.move'}))
RETURN p, length(p) AS so_buoc;

// B4. Đường ngắn nhất KHÔNG OCA (hình fig_path_without_oca.svg)
// Kết quả: 2 bước, stock.picking <- sale.order -> account.move
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..6]-(b:Model {name:'account.move'}))
WHERE none(l IN relationships(p) WHERE l.oca)
RETURN p, length(p) AS so_buoc;


// ---------------------------------------------------------------------
// C. NHIỄU KỸ THUẬT TRONG GRAPH
// ---------------------------------------------------------------------

// C1. Model có nhiều kết nối nhất
// Kết quả: res.users 775, res.partner 184, res.company 181, mail.message 113,
//          account.move 95, product.product 95, ... (lẫn model kỹ thuật và nghiệp vụ)
MATCH (m:Model)-[l:LINKS_TO]-()
RETURN m.name, count(l) AS so_ket_noi
ORDER BY so_ket_noi DESC LIMIT 20;

// C2. Field quan hệ phổ biến nhất
// Kết quả: create_uid 321, write_uid 321, company_id 107, currency_id 46,
//          partner_id 40, product_id 36, user_id 35, message_ids 33, ...
MATCH ()-[l:LINKS_TO]->()
RETURN l.via AS ten_field, count(*) AS so_canh
ORDER BY so_canh DESC LIMIT 25;

// C3. Field quan hệ của mixin chatter (nguồn của tiêu chí lọc)
// Kết quả: 7 field (4 của mail.thread, 3 của mail.activity.mixin)
MATCH (m:Model)-[:HAS_FIELD]->(f:Field)
WHERE m.name IN ['mail.thread', 'mail.activity.mixin'] AND f.relation IS NOT NULL
RETURN m.name, f.name, f.relation;

// C4. Số cạnh kỹ thuật
// Kết quả: 824 / 2.512 cạnh (khoảng 33%), từ 9 field
MATCH ()-[l:LINKS_TO {technical:true}]->() RETURN count(l) AS so_canh_ky_thuat;


// ---------------------------------------------------------------------
// D. HIỆU QUẢ BỘ LỌC: ĐƯỜNG 2 BƯỚC stock.picking -> account.move
// ---------------------------------------------------------------------

// D1. Trước lọc (chỉ bỏ cạnh OCA)
// Kết quả: 69 đường. res.users 24, res.partner 24, mail.message 8, stock.move 6,
//          res.company 2, sale.order 2, mail.activity 1, mail.followers 1, mail.activity.type 1
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca)
RETURN [n IN nodes(p) | n.name][1] AS model_trung_gian, count(*) AS so_duong
ORDER BY so_duong DESC;

// D2. Sau lọc (bỏ cạnh OCA và cạnh kỹ thuật)
// Kết quả: 28 đường. res.partner 15, stock.move 6, res.users 3, res.company 2, sale.order 2
// Giữ nguyên cả 8 đường qua chứng từ (stock.move, sale.order)
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca OR l.technical)
RETURN [n IN nodes(p) | n.name][1] AS model_trung_gian, count(*) AS so_duong
ORDER BY so_duong DESC;

// D3. Chi tiết field của từng đường sau lọc
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca OR l.technical)
RETURN [n IN nodes(p) | n.name][1] AS model_trung_gian,
       [l IN relationships(p) | l.via] AS qua_field
ORDER BY model_trung_gian;
