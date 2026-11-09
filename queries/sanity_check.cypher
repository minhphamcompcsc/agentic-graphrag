// ===== Sanity check Extractor v0 (chạy từng câu trong Neo4j Browser: http://localhost:7474) =====

// 1. Đếm node theo loại
MATCH (n) RETURN labels(n)[0] AS loai, count(*) AS so_luong;

// 2. Liệt kê module OCA đã load (kỳ vọng: 5 module)
MATCH (m:Module {source:'oca'}) RETURN m.name, m.label ORDER BY m.name;

// 3. Model "stock.picking" do module nào tạo, module nào mở rộng
MATCH (mo:Module)-[r:DEFINES|EXTENDS]->(md:Model {name:'stock.picking'})
RETURN mo.name, type(r), mo.source ORDER BY type(r);

// 4. stock_no_negative thêm field gì, vào model nào
MATCH (mo:Module {name:'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
RETURN f.model, f.name, f.label, f.ttype;

// 5. Đường đi ngắn nhất từ phiếu kho đến hoá đơn (xem ảnh hưởng của stock_picking_invoice_link)
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..4]-(b:Model {name:'account.move'}))
RETURN p;

// 6. Chuỗi phụ thuộc của stock_picking_invoice_link
MATCH p = (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON*1..3]->(:Module)
RETURN p;

// 7. Vùng lân cận 1 bước của stock.move (giống Local Search sau này)
MATCH (s:Model {name:'stock.move'})-[l:LINKS_TO]-(t:Model)
RETURN s.name, l.via, l.type, t.name LIMIT 50;
