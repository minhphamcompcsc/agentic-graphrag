"""
Extractor v0.2: reads the Odoo ORM schema from PostgreSQL and loads it into Neo4j.
No LLM. Safe to rerun (uses MERGE).

Changes from v0:
  - Flag LINKS_TO.oca      : edges added by OCA modules
  - Flag LINKS_TO.technical: technical edges (audit fields + fields inherited from chatter mixins)
  - Save graph statistics to data/snapshots/kg_stats_YYYYMMDD.json (used in reports)

Changes from v0.1 (incremental update, RQ2):
  - --incremental: read from PostgreSQL only the metadata of the new modules
    (modules installed in Odoo but not yet in the graph, or the ones given with --modules)
    and MERGE only that part into Neo4j. The semantic layer is not touched.
  - Every run saves data/snapshots/kg_update_YYYYMMDD_<mode>.json: rows read, nodes and
    relationships created (Neo4j counters), time per phase, graph statistics before and after,
    and a fingerprint (SHA-256) of the structural layer.
  - Consistency check: after an incremental run, a full run without --reset must create
    0 nodes and 0 relationships and give the same fingerprint.
  - The full mode reads and writes exactly as v0.1 (same numbers).

Usage (from ~/agentic-graphrag, venv activated):
    python src/extractor/extract_schema.py --reset               # rebuild from scratch (deletes the semantic layer too)
    python src/extractor/extract_schema.py                       # full run with MERGE, keeps the semantic layer
    python src/extractor/extract_schema.py --incremental         # only modules not yet in the graph
    python src/extractor/extract_schema.py --incremental --modules stock_request
"""
import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

PG = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    dbname=os.getenv("PG_DB", "kg_demo"),
    user=os.getenv("PG_USER", "odoo"),
    password=os.getenv("PG_PASSWORD", "odoo"),
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "thesis12345"))
BATCH = 1000

# Technical edge criteria (derived from the Odoo structure, not a hand-made field list):
#   1. Audit fields that Odoo adds to every model
#   2. All relational fields defined by the chatter mixins
AUDIT_FIELDS = ["create_uid", "write_uid"]
TECH_MIXINS = ["mail.thread", "mail.activity.mixin"]

# ============ 1. SQL: read data from Odoo ============
SQL_MODULES = """
SELECT m.name,
       COALESCE(m.shortdesc->>'en_US', m.name) AS label,
       COALESCE(m.summary->>'en_US', '')       AS summary,
       COALESCE(m.author, '')                  AS author,
       m.latest_version                        AS version,
       COALESCE(m.author, '') ILIKE '%Odoo Community Association%' AS is_oca
FROM ir_module_module m
WHERE m.state = 'installed'
"""

SQL_DEPS = """
SELECT m.name AS module, d.name AS depends_on
FROM ir_module_module_dependency d
JOIN ir_module_module m ON m.id = d.module_id
WHERE m.state = 'installed'
"""

SQL_MODELS = """
SELECT im.model AS name,
       COALESCE(im.name->>'en_US', im.model) AS label,
       COALESCE(im.info, '') AS info,
       im.transient
FROM ir_model im
"""

# Which module declares the model (smallest id = defining module, others = extending modules)
SQL_MODEL_OWNERS = """
SELECT im.model, imd.module
FROM ir_model_data imd
JOIN ir_model im ON imd.model = 'ir.model' AND imd.res_id = im.id
ORDER BY im.model, imd.id
"""

SQL_FIELDS = """
SELECT f.model, f.name,
       COALESCE(f.field_description->>'en_US', f.name) AS label,
       f.ttype, f.relation, f.store, f.required
FROM ir_model_fields f
"""

# Which module adds the field (smallest id = first module that declares the field)
SQL_FIELD_OWNERS = """
SELECT f.model, f.name, imd.module
FROM ir_model_data imd
JOIN ir_model_fields f ON imd.model = 'ir.model.fields' AND imd.res_id = f.id
ORDER BY f.model, f.name, imd.id
"""

# ---- Incremental versions (v0.2): only the part that belongs to %(mods)s ----
SQL_INSTALLED = "SELECT name FROM ir_module_module WHERE state = 'installed'"

# "%" must be doubled when the query is sent with parameters (ILIKE pattern in SQL_MODULES)
SQL_MODULES_INC = SQL_MODULES.replace("%", "%%") + "  AND m.name = ANY(%(mods)s)\n"
SQL_DEPS_INC = SQL_DEPS + "  AND m.name = ANY(%(mods)s)\n"

# Models where a new module has an xmlid (defines or extends); all their owner records are read
# so that "smallest id = defining module" is decided exactly as in the full run
SUB_MODELS_INC = "SELECT res_id FROM ir_model_data WHERE model = 'ir.model' AND module = ANY(%(mods)s)"
SQL_MODEL_OWNERS_INC = f"""
SELECT im.model, imd.module
FROM ir_model_data imd
JOIN ir_model im ON imd.model = 'ir.model' AND imd.res_id = im.id
WHERE im.id IN ({SUB_MODELS_INC})
ORDER BY im.model, imd.id
"""
SQL_MODELS_INC = SQL_MODELS + f"WHERE im.id IN ({SUB_MODELS_INC})\n"

# Fields where a new module has an xmlid; all their owner records are read (same reason)
SUB_FIELDS_INC = "SELECT res_id FROM ir_model_data WHERE model = 'ir.model.fields' AND module = ANY(%(mods)s)"
SQL_FIELD_OWNERS_INC = f"""
SELECT f.model, f.name, imd.module
FROM ir_model_data imd
JOIN ir_model_fields f ON imd.model = 'ir.model.fields' AND imd.res_id = f.id
WHERE f.id IN ({SUB_FIELDS_INC})
ORDER BY f.model, f.name, imd.id
"""
# Fields touched by the new modules + every field of the models they define
SQL_FIELDS_INC = SQL_FIELDS + f"WHERE f.id IN ({SUB_FIELDS_INC}) OR f.model = ANY(%(new_models)s)\n"

# ============ 2. Cypher: write to Neo4j ============
CONSTRAINTS = [
    "CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT model_name IF NOT EXISTS FOR (m:Model) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT field_key IF NOT EXISTS FOR (f:Field) REQUIRE f.key IS UNIQUE",
]

Q_MODULES = """
UNWIND $rows AS r
MERGE (m:Module {name: r.name})
SET m.label = r.label, m.summary = r.summary, m.author = r.author,
    m.version = r.version,
    m.source = CASE WHEN r.is_oca THEN 'oca' ELSE 'odoo' END
"""

Q_DEPS = """
UNWIND $rows AS r
MATCH (a:Module {name: r.module})
MERGE (b:Module {name: r.depends_on})
MERGE (a)-[:DEPENDS_ON]->(b)
"""

Q_MODELS = """
UNWIND $rows AS r
MERGE (m:Model {name: r.name})
SET m.label = r.label, m.info = r.info, m.transient = r.transient
"""

Q_DEFINES = """
UNWIND $rows AS r
MATCH (mo:Module {name: r.module}), (md:Model {name: r.model})
MERGE (mo)-[:DEFINES]->(md)
"""

Q_EXTENDS = """
UNWIND $rows AS r
MATCH (mo:Module {name: r.module}), (md:Model {name: r.model})
MERGE (mo)-[:EXTENDS]->(md)
"""

Q_FIELDS = """
UNWIND $rows AS r
MATCH (md:Model {name: r.model})
MERGE (f:Field {key: r.key})
SET f.name = r.name, f.model = r.model, f.label = r.label,
    f.ttype = r.ttype, f.relation = r.relation,
    f.store = r.store, f.required = r.required
MERGE (md)-[:HAS_FIELD]->(f)
"""

# Relational field -> target model, plus a "shortcut" Model-[:LINKS_TO]->Model for fast traversal
Q_RELATIONS = """
UNWIND $rows AS r
MATCH (f:Field {key: r.key}), (s:Model {name: r.model}), (t:Model {name: r.relation})
MERGE (f)-[:RELATES_TO]->(t)
MERGE (s)-[l:LINKS_TO {via: r.name}]->(t)
SET l.type = r.ttype
"""

Q_ADDS_FIELD = """
UNWIND $rows AS r
MATCH (mo:Module {name: r.module}), (f:Field {key: r.key})
MERGE (mo)-[:ADDS_FIELD]->(f)
"""

# ---- Flags (v0.1) ----
Q_FLAGS_DEFAULT = """
MATCH ()-[l:LINKS_TO]->()
SET l.oca = false, l.technical = false
"""

Q_TAG_OCA = """
MATCH (:Module {source: 'oca'})-[:ADDS_FIELD]->(f:Field)
MATCH (:Model {name: f.model})-[l:LINKS_TO {via: f.name}]->()
SET l.oca = true
RETURN count(l) AS n_edges
"""

Q_TAG_TECHNICAL = """
MATCH (mx:Model)-[:HAS_FIELD]->(f:Field)
WHERE mx.name IN $mixins AND f.relation IS NOT NULL
WITH collect(DISTINCT f.name) + $audit AS tech
MATCH ()-[l:LINKS_TO]->() WHERE l.via IN tech
SET l.technical = true
RETURN tech AS fields, count(l) AS n_edges
"""

# Incremental: only edges created in this run have no flag yet
Q_FLAGS_DEFAULT_NEW = """
MATCH ()-[l:LINKS_TO]->() WHERE l.oca IS NULL
SET l.oca = false, l.technical = false
RETURN count(l) AS n_edges
"""

# ---- Fingerprint of the structural layer (v0.2) ----
STRUCT_NODES = {
    "Module": ("name", ["label", "summary", "author", "version", "source"]),
    "Model": ("name", ["label", "info", "transient"]),
    "Field": ("key", ["name", "model", "label", "ttype", "relation", "store", "required"]),
}
STRUCT_RELS = ["DEPENDS_ON", "DEFINES", "EXTENDS", "HAS_FIELD", "ADDS_FIELD", "RELATES_TO", "LINKS_TO"]
LINK_PROPS = ["via", "type", "oca", "technical"]

Q_STATS = """
CALL () { MATCH (n) RETURN labels(n)[0] AS item, count(*) AS total
          UNION ALL
          MATCH ()-[r]->() RETURN type(r) AS item, count(*) AS total }
RETURN item, total ORDER BY item
"""


# ============ 3. Helpers ============
def fetch(cur, sql, params=None):
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def split_owners(rows, key_fn):
    """First record of each key = owner, later records = extenders."""
    owners, extenders, seen = [], [], set()
    for r in rows:
        k = key_fn(r)
        (extenders if k in seen else owners).append(r)
        seen.add(k)
    return owners, extenders


def write(driver, query, rows, name, created):
    """Write in batches; add the Neo4j counters to `created` (nodes, relationships created)."""
    n_nodes = n_rels = 0
    for i in range(0, len(rows), BATCH):
        c = driver.execute_query(query, rows=rows[i:i + BATCH], database_="neo4j").summary.counters
        n_nodes += c.nodes_created
        n_rels += c.relationships_created
    created[name] = {"rows": len(rows), "nodes_created": n_nodes, "relationships_created": n_rels}
    print(f"  {name:<14} {len(rows):>6} rows  (+{n_nodes} nodes, +{n_rels} relationships)")


def run(driver, query, **params):
    records, _, _ = driver.execute_query(query, database_="neo4j", **params)
    return records


def graph_stats(driver):
    return {r["item"]: r["total"] for r in run(driver, Q_STATS)}


def fingerprint(driver):
    """SHA-256 of the structural layer: nodes with their structural properties and relationships
    with their end keys (and LINKS_TO properties). Semantic nodes and properties are ignored."""
    lines = []
    for label, (key, props) in STRUCT_NODES.items():
        for r in run(driver, f"MATCH (n:{label}) RETURN n.{key} AS k, [p IN $props | n[p]] AS v", props=props):
            lines.append(json.dumps([label, r["k"], r["v"]], default=str))
    key_of = "CASE WHEN {x}:Field THEN {x}.key ELSE {x}.name END"
    for t in STRUCT_RELS:
        extra = ", [p IN $props | r[p]] AS v" if t == "LINKS_TO" else ", null AS v"
        q = (f"MATCH (a)-[r:{t}]->(b) RETURN {key_of.format(x='a')} AS ka, "
             f"{key_of.format(x='b')} AS kb{extra}")
        for r in run(driver, q, props=LINK_PROPS):
            lines.append(json.dumps([t, r["ka"], r["kb"], r["v"]], default=str))
    lines.sort()
    return {"sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), "items": len(lines)}


# ============ 4. Main ============
def read_full(cur):
    return {
        "modules": fetch(cur, SQL_MODULES),
        "deps": fetch(cur, SQL_DEPS),
        "models": fetch(cur, SQL_MODELS),
        "model_owners": fetch(cur, SQL_MODEL_OWNERS),
        "fields": fetch(cur, SQL_FIELDS),
        "field_owners": fetch(cur, SQL_FIELD_OWNERS),
    }


def read_incremental(cur, mods):
    p = {"mods": mods}
    data = {
        "modules": fetch(cur, SQL_MODULES_INC, p),
        "deps": fetch(cur, SQL_DEPS_INC, p),
        "models": fetch(cur, SQL_MODELS_INC, p),
        "model_owners": fetch(cur, SQL_MODEL_OWNERS_INC, p),
        "field_owners": fetch(cur, SQL_FIELD_OWNERS_INC, p),
    }
    defines, _ = split_owners(data["model_owners"], lambda r: r["model"])
    new_models = sorted({r["model"] for r in defines if r["module"] in mods})
    data["fields"] = fetch(cur, SQL_FIELDS_INC, {"mods": mods, "new_models": new_models})
    return data


def check():
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()
        stats = graph_stats(driver)
        fp = fingerprint(driver)
        in_graph = {r["n"] for r in run(driver, "MATCH (m:Module) RETURN m.name AS n")}
    with psycopg2.connect(**PG) as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        installed = {r["name"] for r in fetch(cur, SQL_INSTALLED)}
    print("Graph statistics:")
    for k, v in stats.items():
        print(f"  {k:<14} {v:>6}")
    print(f"Structural fingerprint: {fp['sha256']} ({fp['items']} items)")
    print(f"Installed in Odoo: {len(installed)} modules; in the graph: {len(in_graph)} modules")
    print(f"Installed but not in the graph: {sorted(installed - in_graph) or 'none'}")
    print(f"In the graph but not installed: {sorted(in_graph - installed) or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Delete the whole graph before loading")
    ap.add_argument("--incremental", action="store_true",
                    help="Only load the modules that are installed in Odoo but not yet in the graph")
    ap.add_argument("--modules", nargs="*",
                    help="With --incremental: load these modules instead of detecting them")
    ap.add_argument("--check", action="store_true",
                    help="Only print graph statistics, the structural fingerprint and the modules "
                         "installed in Odoo but not in the graph; write nothing")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.reset and args.incremental:
        ap.error("--reset and --incremental cannot be used together")
    mode = "incremental" if args.incremental else ("reset" if args.reset else "full")
    today = f"{date.today():%Y%m%d}"
    snap_dir = Path("data/snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    timing = {}

    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()
        print(f"Extractor v0.2, mode: {mode}")
        print("0) Graph before the run ...")
        stats_before = graph_stats(driver)
        fp_before = fingerprint(driver)
        print(f"  {sum(v for k, v in stats_before.items())} items, structural fingerprint {fp_before['sha256'][:12]}")
        t_start = time.perf_counter()  # timing excludes the fingerprints (they are checks, not part of the update)

        # --- Read PostgreSQL ---
        t0 = time.perf_counter()
        with psycopg2.connect(**PG) as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if args.incremental:
                if args.modules:
                    mods = sorted(args.modules)
                else:
                    in_graph = {r["n"] for r in run(driver, "MATCH (m:Module) RETURN m.name AS n")}
                    mods = sorted(r["name"] for r in fetch(cur, SQL_INSTALLED) if r["name"] not in in_graph)
                print(f"1) Reading the schema of the new modules from PostgreSQL: {mods or 'none'}")
                if not mods:
                    raise SystemExit("Nothing to do: every installed module is already in the graph.")
                data = read_incremental(cur, mods)
            else:
                mods = None
                print("1) Reading the schema from PostgreSQL ...")
                data = read_full(cur)
        timing["read_postgres"] = round(time.perf_counter() - t0, 2)
        for k, v in data.items():
            print(f"  {k:<14} {len(v):>6} rows")

        suffix = "_incremental" if args.incremental else ""
        snap = snap_dir / f"schema_{today}{suffix}.json"
        snap.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        print(f"  Snapshot: {snap}")

        # --- Prepare data ---
        for f in data["fields"]:
            f["key"] = f"{f['model']}:{f['name']}"
        for f in data["field_owners"]:
            f["key"] = f"{f['model']}:{f['name']}"
        relations = [f for f in data["fields"] if f["relation"]]
        defines, extends = split_owners(data["model_owners"], lambda r: r["model"])
        adds_field, _ = split_owners(data["field_owners"], lambda r: r["key"])
        if args.incremental:  # keep only what the new modules own or extend
            defines = [r for r in defines if r["module"] in mods]
            extends = [r for r in extends if r["module"] in mods]
            adds_field = [r for r in adds_field if r["module"] in mods]

        # --- Write Neo4j ---
        t0 = time.perf_counter()
        created = {}
        print("2) Writing to Neo4j ...")
        if args.reset:
            run(driver, "MATCH (n) DETACH DELETE n")
            print("  Old graph deleted")
        for c in CONSTRAINTS:
            run(driver, c)

        write(driver, Q_MODULES, data["modules"], "Module", created)
        write(driver, Q_DEPS, data["deps"], "DEPENDS_ON", created)
        write(driver, Q_MODELS, data["models"], "Model", created)
        write(driver, Q_DEFINES, defines, "DEFINES", created)
        write(driver, Q_EXTENDS, extends, "EXTENDS", created)
        write(driver, Q_FIELDS, data["fields"], "Field", created)
        write(driver, Q_RELATIONS, relations, "RELATES_TO", created)
        write(driver, Q_ADDS_FIELD, adds_field, "ADDS_FIELD", created)

        # --- Flags ---
        print("3) Flagging LINKS_TO edges ...")
        if args.incremental:
            new_edges = run(driver, Q_FLAGS_DEFAULT_NEW)[0]["n_edges"]
            print(f"  new LINKS_TO   {new_edges:>6} edges")
        else:
            run(driver, Q_FLAGS_DEFAULT)
        oca = run(driver, Q_TAG_OCA)[0]
        tech = run(driver, Q_TAG_TECHNICAL, mixins=TECH_MIXINS, audit=AUDIT_FIELDS)[0]
        print(f"  oca            {oca['n_edges']:>6} edges")
        print(f"  technical      {tech['n_edges']:>6} edges ({len(tech['fields'])} fields: {', '.join(sorted(tech['fields']))})")
        timing["write_neo4j"] = round(time.perf_counter() - t0, 2)
        timing["total"] = round(time.perf_counter() - t_start, 2)

        # --- Statistics ---
        print("4) Graph statistics:")
        stats = graph_stats(driver)
        for k, v in stats.items():
            diff = v - stats_before.get(k, 0)
            print(f"  {k:<14} {v:>6}" + (f"  (+{diff})" if diff else ""))
        fp_after = fingerprint(driver)

    tot_nodes = sum(v["nodes_created"] for v in created.values())
    tot_rels = sum(v["relationships_created"] for v in created.values())
    print(f"  Created in this run: {tot_nodes} nodes, {tot_rels} relationships")
    print(f"  Time: read {timing['read_postgres']} s, write {timing['write_neo4j']} s, "
          f"total {timing['total']} s (without the fingerprint)")
    print(f"  Structural fingerprint: before {fp_before['sha256'][:12]}, after {fp_after['sha256'][:12]} "
          f"({fp_after['items']} items)")

    report = {
        "date": today,
        "extractor_version": "v0.2",
        "mode": mode,
        "modules": mods,
        "rows_read": {k: len(v) for k, v in data.items()},
        "created": created,
        "created_total": {"nodes": tot_nodes, "relationships": tot_rels},
        "seconds": timing,
        "llm_calls": 0,
        "graph_before": stats_before,
        "graph_after": stats,
        "fingerprint_before": fp_before,
        "fingerprint_after": fp_after,
        "links_to_oca": oca["n_edges"],
        "links_to_technical": tech["n_edges"],
    }
    upd = snap_dir / f"kg_update_{today}_{mode}.json"
    upd.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Run report saved to: {upd}")

    if not args.incremental:  # same file as v0.1
        kg = {
            "date": today,
            "graph": stats,
            "links_to_oca": oca["n_edges"],
            "links_to_technical": tech["n_edges"],
            "technical_fields": sorted(tech["fields"]),
            "technical_criteria": {"audit_fields": AUDIT_FIELDS, "mixins": TECH_MIXINS},
        }
        stats_file = snap_dir / f"kg_stats_{today}.json"
        stats_file.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Statistics saved to: {stats_file}")
    print("Done.")


if __name__ == "__main__":
    main()
