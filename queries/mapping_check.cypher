// Mapping Agent (map-v0.1): Cypher used to load the KG view (src/mapping/graph_view.py) and
// queries to check one answer by hand in Neo4j Browser. Scoring itself runs in Python on the loaded view.

// M1. Footprint T(m) of a module: models it defines, extends or adds fields to
MATCH (m:Module {name: 'stock_no_negative'})-[r:DEFINES|EXTENDS]->(x:Model)
RETURN x.name AS model, type(r) AS relation
UNION
MATCH (m:Module {name: 'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
RETURN f.model AS model, 'ADDS_FIELD' AS relation;

// M2. Entry functions by vector search (same idea as the Python cosine; needs the query embedding as $q)
// CALL db.index.vector.queryNodes('function_embedding', 5, $q) YIELD node, score
// MATCH (m:Module)-[:PROVIDES]->(node) RETURN m.name, node.name, score;

// M3. Shortest business path from a module to an entry model (technical edges excluded, at most 3 hops)
MATCH (m:Module {name: 'stock_request'})-[:DEFINES|EXTENDS]->(start:Model), (target:Model {name: 'stock.picking'})
MATCH p = shortestPath((start)-[:LINKS_TO*..3]-(target))
WHERE all(l IN relationships(p) WHERE l.technical = false)
RETURN [n IN nodes(p) | n.name] AS models, [l IN relationships(p) | l.via] AS via, length(p) + 1 AS d
ORDER BY d LIMIT 5;

// M4. Models of the answer module where it has code (DEFINES or EXTENDS) vs fields only through _inherits
MATCH (m:Module {name: 'stock_no_negative'})-[:ADDS_FIELD]->(f:Field)
WITH m, f.model AS model, collect(f.name) AS fields
OPTIONAL MATCH (m)-[r:DEFINES|EXTENDS]->(:Model {name: model})
RETURN model, fields, CASE WHEN r IS NULL THEN 'delegated (_inherits)' ELSE type(r) END AS code;
