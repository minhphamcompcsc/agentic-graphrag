// =====================================================================
// Analysis of the initial Knowledge Graph (Internship 1, run on 2026-11-09)
// Run in Neo4j Browser: http://localhost:7474 (one query at a time)
// Requires: extract_schema.py v0.1 has been run (flags l.oca, l.technical exist)
// Results in comments are the actual measured numbers, used in reports.
// =====================================================================


// ---------------------------------------------------------------------
// A. GRAPH CORRECTNESS CHECKS
// ---------------------------------------------------------------------

// A1. Count nodes by label
// Result: Module 66, Model 439, Field 7,755
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n;

// A2. Loaded OCA modules
// Result: 6 modules (5 chosen by hand + sale_stock_picking_invoice_link auto-installed)
MATCH (m:Module {source:'oca'}) RETURN m.name, m.label ORDER BY m.name;

// A3. Who defines and who extends stock.picking (checks the "smallest id" heuristic)
// Result: stock DEFINES; stock_sms, stock_account, sale_stock + 4 OCA modules EXTEND
MATCH (mo:Module)-[r:DEFINES|EXTENDS]->(md:Model {name:'stock.picking'})
RETURN mo.name, type(r), mo.source ORDER BY type(r);

// A4. Fields added by stock_no_negative
// Result: allow_negative_stock on product.category, product.product,
//          product.template, stock.location
MATCH (mo:Module {name:'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
RETURN f.model, f.name, f.label, f.ttype;

// A5. Fields without an identifiable owner module
// Result: 5 phone fields on res.users (delegation inheritance from res.partner)
MATCH (f:Field)
WHERE NOT EXISTS { (:Module)-[:ADDS_FIELD]->(f) }
RETURN f.model, f.name, f.ttype;

// A6. Dependency chain of stock_picking_invoice_link (view as Graph)
MATCH p = (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON*1..3]->(:Module)
RETURN p;


// ---------------------------------------------------------------------
// B. IMPACT OF OCA MODULES ON THE RELATIONSHIP STRUCTURE
// ---------------------------------------------------------------------

// B1. Number of edges added by OCA
// Result: 23 / 2,512 LINKS_TO edges (below 1%)
MATCH ()-[l:LINKS_TO {oca:true}]->() RETURN count(l) AS n_oca_edges;

// B2. Direct link picking <-> invoice
// Result: 2 edges (invoice_ids, picking_ids), both with oca = true
MATCH (a:Model {name:'stock.picking'})-[l:LINKS_TO]-(b:Model {name:'account.move'})
RETURN startNode(l).name AS from_model, l.via AS via_field, l.type, endNode(l).name AS to_model, l.oca;

// B3. Shortest path WITH OCA (figure fig_path_with_oca.svg)
// Result: 1 hop
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..4]-(b:Model {name:'account.move'}))
RETURN p, length(p) AS hops;

// B4. Shortest path WITHOUT OCA (figure fig_path_without_oca.svg)
// Result: 2 hops, stock.picking <- sale.order -> account.move
MATCH p = shortestPath((a:Model {name:'stock.picking'})-[:LINKS_TO*..6]-(b:Model {name:'account.move'}))
WHERE none(l IN relationships(p) WHERE l.oca)
RETURN p, length(p) AS hops;


// ---------------------------------------------------------------------
// C. TECHNICAL NOISE IN THE GRAPH
// ---------------------------------------------------------------------

// C1. Models with the most links
// Result: res.users 775, res.partner 184, res.company 181, mail.message 113,
//          account.move 95, product.product 95, ... (technical and business models mixed)
MATCH (m:Model)-[l:LINKS_TO]-()
RETURN m.name, count(l) AS n_links
ORDER BY n_links DESC LIMIT 20;

// C2. Most frequent relational fields
// Result: create_uid 321, write_uid 321, company_id 107, currency_id 46,
//          partner_id 40, product_id 36, user_id 35, message_ids 33, ...
MATCH ()-[l:LINKS_TO]->()
RETURN l.via AS field_name, count(*) AS n_edges
ORDER BY n_edges DESC LIMIT 25;

// C3. Relational fields of the chatter mixins (source of the filter criterion)
// Result: 7 fields (4 from mail.thread, 3 from mail.activity.mixin)
MATCH (m:Model)-[:HAS_FIELD]->(f:Field)
WHERE m.name IN ['mail.thread', 'mail.activity.mixin'] AND f.relation IS NOT NULL
RETURN m.name, f.name, f.relation;

// C4. Number of technical edges
// Result: 824 / 2,512 edges (about 33%), from 9 fields
MATCH ()-[l:LINKS_TO {technical:true}]->() RETURN count(l) AS n_technical_edges;


// ---------------------------------------------------------------------
// D. FILTER EFFECT: 2-HOP PATHS stock.picking -> account.move
// ---------------------------------------------------------------------

// D1. Before filtering (only OCA edges removed)
// Result: 69 paths. res.users 24, res.partner 24, mail.message 8, stock.move 6,
//          res.company 2, sale.order 2, mail.activity 1, mail.followers 1, mail.activity.type 1
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca)
RETURN [n IN nodes(p) | n.name][1] AS middle_model, count(*) AS n_paths
ORDER BY n_paths DESC;

// D2. After filtering (OCA and technical edges removed)
// Result: 28 paths. res.partner 15, stock.move 6, res.users 3, res.company 2, sale.order 2
// All 8 paths through business documents are kept (stock.move, sale.order)
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca OR l.technical)
RETURN [n IN nodes(p) | n.name][1] AS middle_model, count(*) AS n_paths
ORDER BY n_paths DESC;

// D3. Field details of each path after filtering
MATCH p = (a:Model {name:'stock.picking'})-[:LINKS_TO*2]-(b:Model {name:'account.move'})
WHERE none(l IN relationships(p) WHERE l.oca OR l.technical)
RETURN [n IN nodes(p) | n.name][1] AS middle_model,
       [l IN relationships(p) | l.via] AS via_field
ORDER BY middle_model;

// ===== Query D: DEPENDS_ON chain of stock_picking_invoice_link (2026-11-17) =====
// D1. Direct dependency. Result: stock_account (matches PostgreSQL 1/1)
MATCH (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON]->(d:Module)
RETURN d.name AS module ORDER BY module;

// D2. Transitive chain. Result: 22 modules, depth 1 to 5
// (matches PostgreSQL WITH RECURSIVE 22/22; 353 paths)
MATCH p = (m:Module {name:'stock_picking_invoice_link'})-[:DEPENDS_ON*]->(d:Module)
RETURN d.name AS module, min(length(p)) AS depth
ORDER BY depth, module;
