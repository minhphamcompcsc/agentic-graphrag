"""
In-memory view of the 2-layer KG used by the Mapping Agent (and later the Verifier).

The view is loaded once per run with a few Cypher queries; scoring and path search then run in Python,
so every requirement is scored on exactly the same graph and the run is reproducible.

What the view keeps:
  modules     name -> {label, summary, source}
  deps        module -> set of direct dependencies (DEPENDS_ON)
  touch       module -> {model: set of relations}   relations among DEFINES, EXTENDS, ADDS_FIELD
  added       module -> {model: [(field, label)]}    fields the module adds (ADDS_FIELD)
  models      name -> {label, transient}
  adj         model -> [(model, via)]                LINKS_TO with technical = false, both directions
  functions   key -> {module, name, description, involves: [models], fields: [model:field]}

Two loaders:
  GraphView.from_neo4j(driver)                      the real KG (Minh's machine)
  GraphView.from_snapshot(schema.json, semantic_dir) offline tests in the sandbox: rebuilds the structural
      layer exactly as extract_schema.py does, and the semantic layer from the LLM cache (exact_name links only)
"""
import json
from pathlib import Path

AUDIT_FIELDS = ["create_uid", "write_uid"]            # same criteria as extract_schema.py
TECH_MIXINS = ["mail.thread", "mail.activity.mixin"]
SYSTEM_FIELDS = {"id", "display_name", "create_uid", "create_date", "write_uid", "write_date", "__last_update"}

Q_MODULES = "MATCH (m:Module) RETURN m.name AS name, m.label AS label, m.summary AS summary, m.source AS source"
Q_DEPS = "MATCH (a:Module)-[:DEPENDS_ON]->(b:Module) RETURN a.name AS a, b.name AS b"
Q_TOUCH = "MATCH (m:Module)-[r:DEFINES|EXTENDS]->(x:Model) RETURN m.name AS m, type(r) AS rel, x.name AS x"
Q_ADDS = ("MATCH (m:Module)-[:ADDS_FIELD]->(f:Field) "
          "RETURN m.name AS m, f.model AS x, f.name AS f, f.label AS label")
Q_MODELS = "MATCH (x:Model) RETURN x.name AS name, x.label AS label, x.transient AS transient"
Q_LINKS = ("MATCH (a:Model)-[l:LINKS_TO]->(b:Model) WHERE l.technical = false "
           "RETURN a.name AS a, b.name AS b, l.via AS via")
Q_FUNCTIONS = """
MATCH (m:Module)-[:PROVIDES]->(f:Function)
OPTIONAL MATCH (f)-[:INVOLVES]->(x:Model)
OPTIONAL MATCH (f)-[:INVOLVES_FIELD]->(fd:Field)
RETURN m.name AS module, f.key AS key, f.name AS name, f.description AS description,
       collect(DISTINCT x.name) AS involves, collect(DISTINCT fd.key) AS fields
ORDER BY key
"""


class GraphView:
    def __init__(self):
        self.modules, self.deps, self.touch, self.added = {}, {}, {}, {}
        self.models, self.adj, self.functions = {}, {}, {}
        self.source = None

    # ---------- helpers ----------
    def _touch(self, m, x, rel):
        self.touch.setdefault(m, {}).setdefault(x, set()).add(rel)

    def _link(self, a, b, via):
        self.adj.setdefault(a, []).append((b, via))
        self.adj.setdefault(b, []).append((a, via))

    def finish(self):
        for m in self.modules:
            self.deps.setdefault(m, set())
            self.touch.setdefault(m, {})
            self.added.setdefault(m, {})
        for v in self.adj.values():
            v.sort()
        return self

    def footprint(self, m):
        """T(m): models the module defines, extends or adds fields to."""
        return set(self.touch.get(m, {}))

    def code_models(self, m):
        """Models where the module has code (DEFINES or EXTENDS). A model with only ADDS_FIELD gets the field
        through delegation inheritance (_inherits), e.g. product.product from product.template (R-28)."""
        return {x for x, rels in self.touch.get(m, {}).items() if rels & {"DEFINES", "EXTENDS"}}

    def functions_of(self, m):
        return [k for k, f in self.functions.items() if f["module"] == m]

    def stats(self):
        return {"modules": len(self.modules), "models": len(self.models), "functions": len(self.functions),
                "module_model_pairs": sum(len(v) for v in self.touch.values()),
                "links_non_technical": sum(len(v) for v in self.adj.values()) // 2}

    # ---------- loaders ----------
    @classmethod
    def from_neo4j(cls, driver, database="neo4j"):
        g = cls()
        g.source = "neo4j"
        q = lambda c: driver.execute_query(c, database_=database)[0]  # noqa: E731
        for r in q(Q_MODULES):
            g.modules[r["name"]] = {"label": r["label"], "summary": r["summary"] or "", "source": r["source"]}
        for r in q(Q_DEPS):
            g.deps.setdefault(r["a"], set()).add(r["b"])
        for r in q(Q_MODELS):
            g.models[r["name"]] = {"label": r["label"], "transient": bool(r["transient"])}
        for r in q(Q_TOUCH):
            g._touch(r["m"], r["x"], r["rel"])
        for r in q(Q_ADDS):
            g._touch(r["m"], r["x"], "ADDS_FIELD")
            g.added.setdefault(r["m"], {}).setdefault(r["x"], []).append((r["f"], r["label"]))
        for r in q(Q_LINKS):
            g._link(r["a"], r["b"], r["via"])
        for r in q(Q_FUNCTIONS):
            g.functions[r["key"]] = {"module": r["module"], "name": r["name"], "description": r["description"],
                                     "involves": sorted(r["involves"]), "fields": sorted(r["fields"])}
        for m in g.added.values():
            for lst in m.values():
                lst.sort()
        return g.finish()

    @classmethod
    def from_snapshot(cls, schema_path, semantic_dir=None):
        d = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        g = cls()
        g.source = f"snapshot:{Path(schema_path).name}"
        for m in d["modules"]:
            g.modules[m["name"]] = {"label": m["label"], "summary": m.get("summary") or "",
                                    "source": "oca" if m.get("is_oca") else "odoo"}
        for r in d["deps"]:
            g.deps.setdefault(r["module"], set()).add(r["depends_on"])
        for r in d["models"]:
            g.models[r["name"]] = {"label": r["label"], "transient": bool(r["transient"])}
        seen = {}
        for r in d["model_owners"]:                      # sorted by model, id: first = DEFINES, others = EXTENDS
            if r["model"] not in g.models:
                continue
            first = seen.setdefault(r["model"], r["module"])
            g._touch(r["module"], r["model"], "DEFINES" if r["module"] == first else "EXTENDS")
        labels = {(f["model"], f["name"]): f["label"] for f in d["fields"]}
        owner = {}
        for r in d["field_owners"]:
            owner.setdefault((r["model"], r["name"]), r["module"])
        for (x, f), m in owner.items():
            g._touch(m, x, "ADDS_FIELD")
            g.added.setdefault(m, {}).setdefault(x, []).append((f, labels.get((x, f), f)))
        tech = set(AUDIT_FIELDS) | {f["name"] for f in d["fields"] if f["model"] in TECH_MIXINS and f["relation"]}
        for f in d["fields"]:
            if f["relation"] and f["relation"] in g.models and f["name"] not in tech:
                g._link(f["model"], f["relation"], f["name"])
        if semantic_dir:
            import re
            for p in sorted(Path(semantic_dir).glob("*.llm.json")):
                c = json.loads(p.read_text(encoding="utf-8"))
                m = c["module"]
                if m not in g.modules:
                    continue
                for fn in c["output"]["functions"]:
                    key = f"{m}:" + re.sub(r"[^a-z0-9]+", "_", fn["name"].lower()).strip("_")[:60]
                    inv = sorted({o["odoo_model"] for o in fn["objects"] if o.get("odoo_model") in g.models})
                    g.functions[key] = {"module": m, "name": fn["name"], "description": fn["description"],
                                        "involves": inv, "fields": []}
        for m in g.added.values():
            for lst in m.values():
                lst.sort()
        return g.finish()
