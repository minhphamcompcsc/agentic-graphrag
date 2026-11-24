"""Load a schema snapshot into a mock Odoo catalog (only the tables and columns extract_schema.py reads).

Used by tests/test_incremental_offline.py. Needs a THROWAWAY PostgreSQL database (tables are dropped):
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/load_mock_catalog.py data/snapshots/schema_20261109.json
Never point it at the real Odoo database.
"""
import json, os, sys
import psycopg2
from psycopg2.extras import Json, execute_values

db = os.getenv("PG_DB", "kg_mock")
if db == "kg_demo":
    sys.exit("Refusing to drop tables in kg_demo (the real Odoo database). Use a throwaway database.")
d = json.load(open(sys.argv[1]))
conn = psycopg2.connect(host=os.getenv("PG_HOST", "/tmp"), port=int(os.getenv("PG_PORT", "5433")),
                        user=os.getenv("PG_USER", "odoo"), password=os.getenv("PG_PASSWORD", ""), dbname=db)
cur = conn.cursor()
cur.execute("""
DROP TABLE IF EXISTS ir_module_module, ir_module_module_dependency, ir_model, ir_model_fields, ir_model_data;
CREATE TABLE ir_module_module (id serial, name varchar, shortdesc jsonb, summary jsonb, author varchar,
                               latest_version varchar, state varchar);
CREATE TABLE ir_module_module_dependency (id serial, module_id int, name varchar);
CREATE TABLE ir_model (id serial, model varchar, name jsonb, info text, transient boolean);
CREATE TABLE ir_model_fields (id serial, model varchar, model_id int, name varchar, field_description jsonb,
                              ttype varchar, relation varchar, store boolean, required boolean);
CREATE TABLE ir_model_data (id serial, module varchar, model varchar, res_id int);
""")
mod_id = {}
for m in d["modules"]:
    cur.execute("INSERT INTO ir_module_module (name, shortdesc, summary, author, latest_version, state) "
                "VALUES (%s,%s,%s,%s,%s,'installed') RETURNING id",
                (m["name"], Json({"en_US": m["label"]}), Json({"en_US": m["summary"]}), m["author"], m["version"]))
    mod_id[m["name"]] = cur.fetchone()[0]
execute_values(cur, "INSERT INTO ir_module_module_dependency (module_id, name) VALUES %s",
               [(mod_id[r["module"]], r["depends_on"]) for r in d["deps"]])
model_id = {}
for m in d["models"]:
    cur.execute("INSERT INTO ir_model (model, name, info, transient) VALUES (%s,%s,%s,%s) RETURNING id",
                (m["name"], Json({"en_US": m["label"]}), m["info"], m["transient"]))
    model_id[m["name"]] = cur.fetchone()[0]
field_id = {}
for f in d["fields"]:
    cur.execute("INSERT INTO ir_model_fields (model, model_id, name, field_description, ttype, relation, store, required) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (f["model"], model_id[f["model"]], f["name"], Json({"en_US": f["label"]}), f["ttype"], f["relation"],
                 f["store"], f["required"]))
    field_id[(f["model"], f["name"])] = cur.fetchone()[0]
# xmlids, in snapshot order (= id order inside each key)
rows = [(r["module"], "ir.model", model_id[r["model"]]) for r in d["model_owners"]]
rows += [(r["module"], "ir.model.fields", field_id[(r["model"], r["name"])]) for r in d["field_owners"]]
execute_values(cur, "INSERT INTO ir_model_data (module, model, res_id) VALUES %s", rows)
conn.commit()
print("loaded", len(d["modules"]), "modules,", len(d["models"]), "models,", len(d["fields"]), "fields,", len(rows), "xmlids")
