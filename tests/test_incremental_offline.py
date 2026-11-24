"""Offline test of the incremental mode of extract_schema.py (v0.2), without Neo4j.

The real SQL queries run against a mock Odoo catalog (tests/load_mock_catalog.py); graph writes are
emulated in Python with the same MATCH/MERGE rules as the Cypher queries.
For a module X (or a group of modules):
    before = graph built from the catalog without X
    after  = before + incremental run for the modules detected as "installed but not in the graph"
The test passes when `after` equals the graph of a full run and the detected modules are exactly X.

    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/test_incremental_offline.py
"""
import copy, importlib.util, json, os, sys
from pathlib import Path
os.environ.setdefault("PG_HOST", "/tmp"); os.environ.setdefault("PG_PORT", "5433")
os.environ.setdefault("PG_DB", "kg_mock"); os.environ.setdefault("PG_PASSWORD", "")
src = Path(__file__).resolve().parents[1] / "src/extractor/extract_schema.py"
spec = importlib.util.spec_from_file_location("es", src); es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)
import psycopg2, psycopg2.extras

def prepare(data, mods=None):
    data = copy.deepcopy(data)
    for f in data["fields"]: f["key"] = f"{f['model']}:{f['name']}"
    for f in data["field_owners"]: f["key"] = f"{f['model']}:{f['name']}"
    defines, extends = es.split_owners(data["model_owners"], lambda r: r["model"])
    adds, _ = es.split_owners(data["field_owners"], lambda r: r["key"])
    if mods is not None:
        defines = [r for r in defines if r["module"] in mods]
        extends = [r for r in extends if r["module"] in mods]
        adds = [r for r in adds if r["module"] in mods]
    return data, defines, extends, adds

def apply(G, data, defines, extends, adds):
    """Same semantics as Q_MODULES ... Q_ADDS_FIELD (MATCH skips missing endpoints)."""
    for r in data["modules"]: G["Module"][r["name"]] = (r["label"], r["summary"], r["author"], r["version"], r["is_oca"])
    for r in data["deps"]:
        if r["module"] in G["Module"]:
            G["Module"].setdefault(r["depends_on"], None); G["DEPENDS_ON"].add((r["module"], r["depends_on"]))
    for r in data["models"]: G["Model"][r["name"]] = (r["label"], r["info"], r["transient"])
    for rel, rows in (("DEFINES", defines), ("EXTENDS", extends)):
        for r in rows:
            if r["module"] in G["Module"] and r["model"] in G["Model"]: G[rel].add((r["module"], r["model"]))
    for r in data["fields"]:
        if r["model"] in G["Model"]:
            G["Field"][r["key"]] = (r["label"], r["ttype"], r["relation"], r["store"], r["required"])
            G["HAS_FIELD"].add((r["model"], r["key"]))
    for r in data["fields"]:
        if r["relation"] and r["key"] in G["Field"] and r["model"] in G["Model"] and r["relation"] in G["Model"]:
            G["RELATES_TO"].add((r["key"], r["relation"])); G["LINKS_TO"].add((r["model"], r["relation"], r["name"]))
    for r in adds:
        if r["module"] in G["Module"] and r["key"] in G["Field"]: G["ADDS_FIELD"].add((r["module"], r["key"]))
    return G

def empty():
    return {k: {} for k in ("Module", "Model", "Field")} | {k: set() for k in es.STRUCT_RELS}

def without(full, mods):
    """Catalog as it was before installing `mods`."""
    d = copy.deepcopy(full)
    first_model, first_field = {}, {}
    for r in d["model_owners"]: first_model.setdefault(r["model"], r["module"])
    for r in d["field_owners"]: first_field.setdefault((r["model"], r["name"]), r["module"])
    gone_models = {m for m, o in first_model.items() if o in mods}
    gone_fields = {k for k, o in first_field.items() if o in mods} | {k for k in first_field if k[0] in gone_models}
    d["modules"] = [r for r in d["modules"] if r["name"] not in mods]
    d["deps"] = [r for r in d["deps"] if r["module"] not in mods]
    d["models"] = [r for r in d["models"] if r["name"] not in gone_models]
    d["model_owners"] = [r for r in d["model_owners"] if r["module"] not in mods and r["model"] not in gone_models]
    d["fields"] = [r for r in d["fields"] if (r["model"], r["name"]) not in gone_fields]
    d["field_owners"] = [r for r in d["field_owners"] if r["module"] not in mods and (r["model"], r["name"]) not in gone_fields]
    return d

conn = psycopg2.connect(**es.PG); cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
full = es.read_full(cur)
G_full = apply(empty(), *prepare(full))
print("full graph:", {k: len(v) for k, v in G_full.items()})
ok_all = True
for mods in (["stock_no_negative"], ["stock_move_location"], ["stock_picking_back2draft"],
             ["stock_picking_auto_create_lot"], ["sale_stock_picking_invoice_link"],
             ["stock_picking_invoice_link", "sale_stock_picking_invoice_link"]):
    before = without(full, set(mods))
    G = apply(empty(), *prepare(before))
    installed = {r["name"] for r in es.fetch(cur, es.SQL_INSTALLED)}
    detected = sorted(installed - set(G["Module"]))
    inc = es.read_incremental(cur, detected)
    G = apply(G, *prepare(inc, detected))
    diff = {k: (len(G_full[k]) - len(G[k])) for k in G_full if G[k] != G_full[k]}
    ok = not diff and detected == sorted(mods)
    ok_all &= ok
    print(f"{'OK  ' if ok else 'FAIL'} {mods}: detected {detected}; rows read "
          f"{ {k: len(v) for k, v in inc.items()} }; diff {diff}")
print("ALL OK" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
