// ===== Sanity check of Extractor v0 (run one query at a time in Neo4j Browser: http://localhost:7474) =====

// 1. Count nodes by label
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n;

// 2. List the loaded OCA modules (expected: 5 modules)
MATCH (m:Module {source:'oca'}) RETURN m.name, m.label ORDER BY m.name;

// 3. Which module defines and which modules extend the model "stock.picking"
MATCH (mo:Module)-[r:DEFINES|EXTENDS]->(md:Model {name:'stock.picking'})
RETURN mo.name, type(r), mo.source ORDER BY type(r);

// 4. Which fields stock_no_negative adds, and to which models
MATCH (mo:Module {name:'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
RETURN f.model, f.name, f.label, f.ttype;

// 5. Shortest path from picking to invoice (shows the effect of stock_picking_invoice_link)
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..4]-(b:Model {name:'account.move'}))
RETURN p;

// 6. Dependency chain of stock_picking_invoice_link
MATCH p = (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON*1..3]->(:Module)
RETURN p;

// 7. 1-hop neighbourhood of stock.move (similar to Local Search later)
MATCH (s:Model {name:'stock.move'})-[l:LINKS_TO]-(t:Model)
RETURN s.name, l.via, l.type, t.name LIMIT 50;
