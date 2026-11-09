"""
Extractor v0.1: đọc ORM schema của Odoo từ PostgreSQL và đẩy vào Neo4j.
Không dùng LLM. Chạy lại nhiều lần được (dùng MERGE).

Thay đổi so với v0:
  - Gắn cờ LINKS_TO.oca      : cạnh do module OCA thêm vào
  - Gắn cờ LINKS_TO.technical: cạnh kỹ thuật (field kiểm toán + field kế thừa từ mixin chatter)
  - Lưu thống kê graph ra data/snapshots/kg_stats_YYYYMMDD.json (dùng cho báo cáo)

Cách chạy (từ thư mục ~/agentic-graphrag, đã activate venv):
    python src/extractor/extract_schema.py --reset
"""
import argparse
import json
import os
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

# Tiêu chí cạnh kỹ thuật (lấy từ cấu trúc Odoo, không liệt kê tay từng field):
#   1. Field kiểm toán Odoo tự thêm cho mọi model
#   2. Mọi field quan hệ do các mixin chatter định nghĩa
AUDIT_FIELDS = ["create_uid", "write_uid"]
TECH_MIXINS = ["mail.thread", "mail.activity.mixin"]

# ============ 1. SQL: lấy dữ liệu từ Odoo ============
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

# Module nào khai báo model (id nhỏ nhất = module tạo ra model, còn lại = module mở rộng)
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

# Module nào thêm field (id nhỏ nhất = module đầu tiên khai báo field)
SQL_FIELD_OWNERS = """
SELECT f.model, f.name, imd.module
FROM ir_model_data imd
JOIN ir_model_fields f ON imd.model = 'ir.model.fields' AND imd.res_id = f.id
ORDER BY f.model, f.name, imd.id
"""

# ============ 2. Cypher: ghi vào Neo4j ============
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

# Field quan hệ -> model đích, kèm "đường tắt" Model-[:LINKS_TO]->Model để duyệt graph nhanh
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

# ---- Gắn cờ (v0.1) ----
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

Q_STATS = """
CALL () { MATCH (n) RETURN labels(n)[0] AS item, count(*) AS total
          UNION ALL
          MATCH ()-[r]->() RETURN type(r) AS item, count(*) AS total }
RETURN item, total ORDER BY item
"""


# ============ 3. Hàm phụ ============
def fetch(cur, sql):
    cur.execute(sql)
    return [dict(r) for r in cur.fetchall()]


def split_owners(rows, key_fn):
    """Bản ghi đầu tiên của mỗi key = owner, các bản ghi sau = extender."""
    owners, extenders, seen = [], [], set()
    for r in rows:
        k = key_fn(r)
        (extenders if k in seen else owners).append(r)
        seen.add(k)
    return owners, extenders


def write(driver, query, rows, name):
    for i in range(0, len(rows), BATCH):
        driver.execute_query(query, rows=rows[i:i + BATCH], database_="neo4j")
    print(f"  {name:<14} {len(rows):>6} dòng")


def run(driver, query, **params):
    records, _, _ = driver.execute_query(query, database_="neo4j", **params)
    return records


# ============ 4. Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Xoá toàn bộ graph trước khi load")
    args = ap.parse_args()
    today = f"{date.today():%Y%m%d}"
    snap_dir = Path("data/snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)

    # --- Đọc PostgreSQL ---
    print("1) Đọc schema từ PostgreSQL ...")
    with psycopg2.connect(**PG) as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        data = {
            "modules": fetch(cur, SQL_MODULES),
            "deps": fetch(cur, SQL_DEPS),
            "models": fetch(cur, SQL_MODELS),
            "model_owners": fetch(cur, SQL_MODEL_OWNERS),
            "fields": fetch(cur, SQL_FIELDS),
            "field_owners": fetch(cur, SQL_FIELD_OWNERS),
        }
    for k, v in data.items():
        print(f"  {k:<14} {len(v):>6} dòng")

    snap = snap_dir / f"schema_{today}.json"
    snap.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    print(f"  Snapshot: {snap}")

    # --- Chuẩn bị dữ liệu ---
    for f in data["fields"]:
        f["key"] = f"{f['model']}:{f['name']}"
    for f in data["field_owners"]:
        f["key"] = f"{f['model']}:{f['name']}"
    relations = [f for f in data["fields"] if f["relation"]]
    defines, extends = split_owners(data["model_owners"], lambda r: r["model"])
    adds_field, _ = split_owners(data["field_owners"], lambda r: r["key"])

    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        driver.verify_connectivity()

        # --- Ghi Neo4j ---
        print("2) Ghi vào Neo4j ...")
        if args.reset:
            run(driver, "MATCH (n) DETACH DELETE n")
            print("  Đã xoá graph cũ")
        for c in CONSTRAINTS:
            run(driver, c)

        write(driver, Q_MODULES, data["modules"], "Module")
        write(driver, Q_DEPS, data["deps"], "DEPENDS_ON")
        write(driver, Q_MODELS, data["models"], "Model")
        write(driver, Q_DEFINES, defines, "DEFINES")
        write(driver, Q_EXTENDS, extends, "EXTENDS")
        write(driver, Q_FIELDS, data["fields"], "Field")
        write(driver, Q_RELATIONS, relations, "RELATES_TO")
        write(driver, Q_ADDS_FIELD, adds_field, "ADDS_FIELD")

        # --- Gắn cờ ---
        print("3) Gắn cờ cạnh LINKS_TO ...")
        run(driver, Q_FLAGS_DEFAULT)
        oca = run(driver, Q_TAG_OCA)[0]
        tech = run(driver, Q_TAG_TECHNICAL, mixins=TECH_MIXINS, audit=AUDIT_FIELDS)[0]
        print(f"  oca            {oca['n_edges']:>6} cạnh")
        print(f"  technical      {tech['n_edges']:>6} cạnh ({len(tech['fields'])} field: {', '.join(sorted(tech['fields']))})")

        # --- Thống kê ---
        print("4) Thống kê graph:")
        stats = {r["item"]: r["total"] for r in run(driver, Q_STATS)}
        for k, v in stats.items():
            print(f"  {k:<14} {v:>6}")

    report = {
        "date": today,
        "graph": stats,
        "links_to_oca": oca["n_edges"],
        "links_to_technical": tech["n_edges"],
        "technical_fields": sorted(tech["fields"]),
        "technical_criteria": {"audit_fields": AUDIT_FIELDS, "mixins": TECH_MIXINS},
    }
    stats_file = snap_dir / f"kg_stats_{today}.json"
    stats_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Thống kê lưu tại: {stats_file}")
    print("Xong.")


if __name__ == "__main__":
    main()
