// answer_check.cypher: check the answers of the evaluation set using KG metadata (2026-11-24)
// Run in Neo4j Browser (http://localhost:7474), one query at a time.
// Results of the 2026-11-24 run: data/eval/answer_check_20261124/ (questions 1 to 6, 6 seen modules)
// and data/eval/answer_check_20261124b/ (questions 17 to 20, unseen module stock_request, queries Q-A2 to Q-C2)
// Do not use the semantic layer (Function, INVOLVES) as ground truth: it is a system output (circular).

// ---------------------------------------------------------------
// Q-A. Models and fields added by each module (ADDS_FIELD)
// Sanity check: stock_no_negative must add allow_negative_stock on 4 models (R-07)
// Result: QA_adds_field.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[:ADDS_FIELD]->(f:Field)
RETURN mod AS module, f.model AS model, count(f) AS n_fields, collect(f.name) AS fields
ORDER BY module, model;

// ---------------------------------------------------------------
// Q-B. Models defined (DEFINES) or extended (EXTENDS) by each module
// transient = true means a wizard
// Result: QB_defines_extends.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[r:DEFINES|EXTENDS]->(mo:Model)
RETURN mod AS module, type(r) AS relation, mo.name AS model, mo.transient AS transient
ORDER BY module, relation, model;

// ---------------------------------------------------------------
// Q-C. Direct dependencies (DEPENDS_ON, 1 hop)
// Result: QC_depends_on.csv
// ---------------------------------------------------------------
UNWIND ['stock_no_negative','stock_picking_invoice_link','stock_picking_auto_create_lot',
        'stock_move_location','stock_picking_back2draft','sale_stock_picking_invoice_link'] AS mod
MATCH (:Module {name: mod})-[:DEPENDS_ON]->(d:Module)
RETURN mod AS module, collect(d.name) AS direct_depends
ORDER BY module;

// ---------------------------------------------------------------
// Q-D. Question 6: does standard Odoo already link a sales order to deliveries and invoices?
// If adding_module is a standard Odoo module (sale, sale_stock), question 6 needs no OCA module
// Result: QD_sale_order_links.csv
// Note: the column aliases were renamed to English on 2026-11-24; the exported CSV headers were renamed the same way (values unchanged).
// ---------------------------------------------------------------
MATCH (f:Field {model: 'sale.order'})-[:RELATES_TO]->(t:Model)
WHERE t.name IN ['stock.picking', 'account.move']
OPTIONAL MATCH (owner:Module)-[:ADDS_FIELD]->(f)
RETURN f.name AS field, f.ttype AS ttype, t.name AS target_model, owner.name AS adding_module, owner.source AS source;

// ===============================================================
// Questions 17 to 20: unseen module stock_request (added to the KG by the incremental update, R-30)
// Same queries as Q-A, Q-B, Q-C for one module. Results: data/eval/answer_check_20261124b/
// ===============================================================

// Q-A2. Models and fields added by stock_request (expected: 162 fields on 8 models). Result: QA_stock_request.csv
MATCH (:Module {name: 'stock_request'})-[:ADDS_FIELD]->(f:Field)
RETURN 'stock_request' AS module, f.model AS model, count(f) AS n_fields, collect(f.name) AS fields
ORDER BY model;

// Q-B2. Models defined or extended by stock_request (expected: 4 DEFINES, 10 EXTENDS). Result: QB_stock_request.csv
MATCH (:Module {name: 'stock_request'})-[r:DEFINES|EXTENDS]->(mo:Model)
RETURN 'stock_request' AS module, type(r) AS relation, mo.name AS model, mo.transient AS transient
ORDER BY relation, model;

// Q-C2. Direct dependencies of stock_request (expected: stock). Result: QC_stock_request.csv
MATCH (:Module {name: 'stock_request'})-[:DEPENDS_ON]->(d:Module)
RETURN 'stock_request' AS module, collect(d.name) AS direct_depends;
