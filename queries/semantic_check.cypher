// ===== Kiểm tra lớp ngữ nghĩa sem-v0 (extract_semantic.py) =====
// Chạy trong Neo4j Browser sau khi ghi lớp ngữ nghĩa.

// S1. Số nút, quan hệ mới
MATCH (f:Function) RETURN 'Function' AS item, count(f) AS total
UNION ALL MATCH (c:Concept) RETURN 'Concept' AS item, count(c) AS total
UNION ALL MATCH ()-[r:PROVIDES]->() RETURN 'PROVIDES' AS item, count(r) AS total
UNION ALL MATCH ()-[r:INVOLVES]->() RETURN 'INVOLVES' AS item, count(r) AS total
UNION ALL MATCH ()-[r:INVOLVES_FIELD]->() RETURN 'INVOLVES_FIELD' AS item, count(r) AS total
UNION ALL MATCH ()-[r:MENTIONS]->() RETURN 'MENTIONS' AS item, count(r) AS total;

// S2. Chức năng của từng module, kèm model liên quan
MATCH (m:Module)-[:PROVIDES]->(f:Function)
OPTIONAL MATCH (f)-[i:INVOLVES]->(x:Model)
RETURN m.name AS module, f.name AS chuc_nang, f.evidence_found AS co_bang_chung,
       collect(x.name + ' [' + i.method + ']') AS models
ORDER BY module, chuc_nang;

// S3. Đối tượng không gắn được vào Model (thành Concept): xem để chỉnh ngưỡng hoặc prompt
MATCH (m:Module)-[:PROVIDES]->(f:Function)-[r:MENTIONS]->(c:Concept)
RETURN m.name AS module, f.name AS chuc_nang, r.mention AS doi_tuong
ORDER BY module;

// S4. Hình minh hoạ: stock_no_negative qua 2 lớp (cấu trúc + ngữ nghĩa)
MATCH p = (:Module {name:'stock_no_negative'})-[:PROVIDES]->(:Function)-[:INVOLVES|INVOLVES_FIELD]->()
RETURN p;

// S5. Thử điểm vào bằng vector (cần 1 vector yêu cầu; ví dụ lấy vector của 1 chức năng có sẵn)
MATCH (q:Function) WITH q LIMIT 1
CALL db.index.vector.queryNodes('function_embedding', 3, q.embedding) YIELD node, score
RETURN q.name AS truy_van, node.module AS module, node.name AS chuc_nang, round(score, 3) AS score;
