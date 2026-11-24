// ===== Checks of the semantic layer sem-v0 (extract_semantic.py) =====
// Run in Neo4j Browser after writing the semantic layer.

// S1. Number of new nodes and relationships
MATCH (f:Function) RETURN 'Function' AS item, count(f) AS total
UNION ALL MATCH (c:Concept) RETURN 'Concept' AS item, count(c) AS total
UNION ALL MATCH ()-[r:PROVIDES]->() RETURN 'PROVIDES' AS item, count(r) AS total
UNION ALL MATCH ()-[r:INVOLVES]->() RETURN 'INVOLVES' AS item, count(r) AS total
UNION ALL MATCH ()-[r:INVOLVES_FIELD]->() RETURN 'INVOLVES_FIELD' AS item, count(r) AS total
UNION ALL MATCH ()-[r:MENTIONS]->() RETURN 'MENTIONS' AS item, count(r) AS total;

// S2. Functions of each module with related models
MATCH (m:Module)-[:PROVIDES]->(f:Function)
OPTIONAL MATCH (f)-[i:INVOLVES]->(x:Model)
RETURN m.name AS module, f.name AS function, f.evidence_found AS has_evidence,
       collect(x.name + ' [' + i.method + ']') AS models
ORDER BY module, function;

// S3. Mentions not linked to a Model (became Concept): review to tune the threshold or prompt
MATCH (m:Module)-[:PROVIDES]->(f:Function)-[r:MENTIONS]->(c:Concept)
RETURN m.name AS module, f.name AS function, r.mention AS mention
ORDER BY module;

// S4. Illustration: stock_no_negative across the 2 layers (structural + semantic)
MATCH p = (:Module {name:'stock_no_negative'})-[:PROVIDES]->(:Function)-[:INVOLVES|INVOLVES_FIELD]->()
RETURN p;

// S5. Test vector entry points (needs 1 requirement vector; e.g. reuse the vector of an existing function)
MATCH (q:Function) WITH q LIMIT 1
CALL db.index.vector.queryNodes('function_embedding', 3, q.embedding) YIELD node, score
RETURN q.name AS query_function, node.module AS module, node.name AS function, round(score, 3) AS score;
