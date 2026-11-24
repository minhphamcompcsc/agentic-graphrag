"""
Semantic layer extractor sem-v0: an LLM reads the module text (Collector output),
extracts BUSINESS FUNCTIONS, and links them to the structural layer already in Neo4j.

Based on iText2KG (Lairgi et al., 2024), using 2 components of the itext2kg 1.1.0 library:
  1. DocumentsDistiller: the LLM distills the text into a fixed schema (blueprint).
  2. Matcher: entity linking by embedding similarity with a threshold (default 0.7, as in iText2KG).
The free entity and relation extraction of iText2KG is not used: the structural layer already has
Model and Field nodes, which serve as "anchors" and stop the LLM from inventing new labels.

Entity linking (each object mentioned by the LLM), in order:
  1. exact_name : technical name guessed by the LLM (e.g. stock.picking) exists in the KG
  2. exact_label: display name equals a Model label (e.g. "Product Category")
  3. embedding  : iText2KG Matcher, cosine(bge-m3) > threshold
  4. new        : no match, create a Concept node

Graph additions:
  (:Function {key, module, name, description, evidence, evidence_found, section, embedding, ...})
  (:Module)-[:PROVIDES]->(:Function)
  (:Function)-[:INVOLVES {mention, method, score}]->(:Model)
  (:Function)-[:INVOLVES_FIELD {mention, method}]->(:Field)
  (:Function)-[:MENTIONS {mention}]->(:Concept)
  Module.embedding (module text) for sim(q,m) and baseline B1

Outputs:
  data/semantic/<module>.llm.json        raw LLM output (cache, for reproducibility)
  data/semantic/links_YYYYMMDD.csv       each linked object, with a gold_model column for manual labels
  data/snapshots/semantic_stats_YYYYMMDD.json

Usage (in ~/agentic-graphrag, venv activated):
  python src/extractor/extract_semantic.py --dry-run     # test, no write to Neo4j
  python src/extractor/extract_semantic.py               # write to Neo4j (uses the LLM cache if present)
  python src/extractor/extract_semantic.py --refresh     # force new LLM calls
Note: extract_schema.py --reset deletes the whole graph. Run this file again afterwards
(it uses the cache, so no LLM calls).
"""
import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=True)   # .env wins over stale environment variables in the terminal

EXTRACTOR_VERSION = "sem-v0.2"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")   # changed from gemini-3.8-flash on 2026-11-23 (3.8 was overloaded)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "thesis12345"))

# readme sections given to the LLM. Skip ROADMAP (planned features) and INSTALL (technical)
SECTIONS_FOR_LLM = ["DESCRIPTION", "CONTEXT", "USAGE", "CONFIGURE"]


# ============ 1. LLM output schema (blueprint for DocumentsDistiller) ============
class ObjectMention(BaseModel):
    mention: str = Field(description="Business object exactly as written in the text, "
                                     "e.g. 'picking', 'invoice', 'product category', 'location'")
    odoo_model: Optional[str] = Field(default=None, description=(
        "Your best guess of the Odoo 18 technical model name, e.g. 'stock.picking'. "
        "Null if not confident."))


class FieldMention(BaseModel):
    mention: str = Field(description="Option, setting or field exactly as written, e.g. 'Allow Negative Stock'")
    odoo_model: Optional[str] = Field(default=None, description="Odoo 18 technical model of the field, null if not confident")
    odoo_field: Optional[str] = Field(default=None, description="Odoo technical field name, e.g. 'allow_negative_stock', null if not confident")


class BusinessFunction(BaseModel):
    name: str = Field(description="Short verb phrase in English, at most 10 words")
    description: str = Field(description="1 to 2 sentences in plain business language")
    evidence: str = Field(description="One sentence copied VERBATIM from the context that supports this function")
    section: str = Field(description="Section the evidence comes from: SUMMARY, DESCRIPTION, CONTEXT, USAGE or CONFIGURE")
    objects: List[ObjectMention] = Field(default_factory=list, description="Business objects involved")
    fields: List[FieldMention] = Field(default_factory=list, description="Options or fields explicitly mentioned")


class ModuleFunctions(BaseModel):
    functions: List[BusinessFunction] = Field(description="Business functions provided by the module")


PROMPT = """
# DIRECTIVES
- You are an Odoo 18 functional consultant building a knowledge graph of module capabilities.
- The context is the documentation of ONE Odoo module: {module} ("{label}").
- Extract the BUSINESS FUNCTIONS the module provides: a capability, business rule or automation
  that a business user could ask for in a requirement
  (e.g. "Block stock operations that would make stock negative").
- Do not extract installation steps, developer or code notes, contributors, or planned features.
- Merge the steps of one procedure into one function. Usually 1 to 5 functions; a very short text may give only 1.
- For each function fill: name, description, evidence (copy one sentence VERBATIM from the context),
  section, objects (business objects as written + best guess of the Odoo 18 technical model, null if unsure),
  fields (options or fields explicitly named in the context; technical names only if confident).
- Do not invent capabilities that are not stated in the context.
"""


# ============ 2. Text preparation ============
def build_input_text(rec: dict) -> str:
    """Text given to the LLM: name, summary, manifest description, business readme sections.
    Code blocks and image links are removed (no business meaning)."""
    man = rec.get("manifest", {})
    parts = [f"# {man.get('name', rec['module'])}"]
    if man.get("summary"):
        parts.append(f"## SUMMARY\n{man['summary']}")
    if man.get("description"):
        parts.append(f"## DESCRIPTION (manifest)\n{man['description']}")
    for sec in SECTIONS_FOR_LLM:
        if sec in rec.get("readme", {}):
            parts.append(f"## {sec}\n{rec['readme'][sec]['text']}")
    text = "\n\n".join(parts)
    text = re.sub(r"```.*?```", "", text, flags=re.S)        # code blocks
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # markdown images
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def norm(s: str) -> str:
    """Normalize for quote matching: drop markdown marks, collapse whitespace, lowercase."""
    s = re.sub(r"[*_`\\]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ============ 3. LLM (itext2kg DocumentsDistiller) ============
# Gemini free tier: 20 calls/day/model. Every retry is counted too.
# sem-v0 had 3 stacked retry layers: LangChain (6) x itext2kg (3, fixed in
# LangchainOutputParser) x script (3), so 1 module could cost dozens of calls. sem-v0.1:
#   - disable LangChain retries (max_retries=0)
#   - the script calls once (LLM_TRIES=1); itext2kg retries up to 3 times -> at most 3 calls/module
#   - stop at once on a daily quota error, keeping the cache of finished modules
#   - long timeout to avoid 504 Deadline errors with structured output
#   - pause between modules to stay under the per-minute limit
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_TRIES = int(os.getenv("LLM_TRIES", "1"))
LLM_PAUSE = int(os.getenv("LLM_PAUSE", "15"))


class DailyQuotaExceeded(RuntimeError):
    pass


def make_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, max_retries=0, timeout=LLM_TIMEOUT)


async def distill_module(llm, module: str, label: str, text: str, tries: int = LLM_TRIES) -> ModuleFunctions:
    from itext2kg import DocumentsDistiller
    distiller = DocumentsDistiller(llm_model=llm)
    query = PROMPT.format(module=module, label=label)
    last_err = None
    for i in range(tries):
        try:
            out = await distiller.distill(documents=[text], output_data_structure=ModuleFunctions,
                                          IE_query=query)
            if isinstance(out, dict):
                out = ModuleFunctions(**out)
            if out and out.functions:
                return out
            last_err = "LLM returned an empty result"
        except Exception as e:  # format error or rate limit: retry
            last_err = repr(e)
            if "PerDay" in last_err or "per day" in last_err:
                raise DailyQuotaExceeded(last_err[:300])
        if i + 1 < tries:
            print(f"    attempt {i + 1}/{tries} failed: {last_err[:200]}; waiting 60 s")
            time.sleep(60)
    raise RuntimeError(f"{module}: no functions extracted after {tries} attempts ({last_err[:300]})")


def get_functions(llm_factory, rec: dict, cache_dir: Path, refresh: bool):
    """Use the cache if the input text is unchanged; otherwise call the LLM and save the cache."""
    module = rec["module"]
    label = rec.get("manifest", {}).get("name", module)
    text = build_input_text(rec)
    cache = cache_dir / f"{module}.llm.json"
    if cache.exists() and not refresh:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("input_sha256") == sha(text) and c.get("llm") == LLM_MODEL:
            return ModuleFunctions(**c["output"]), c, True
        why = "input text changed" if c.get("input_sha256") != sha(text) else f"cache from model {c.get('llm')}"
        print(f"  {module}: {why}, calling the LLM again with {LLM_MODEL}")
    print(f"  {module}: calling the LLM ...")
    out = asyncio.run(distill_module(llm_factory(), module, label, text))
    time.sleep(LLM_PAUSE)
    c = {
        "module": module,
        "extractor_version": EXTRACTOR_VERSION,
        "llm": LLM_MODEL, "temperature": 0,
        "itext2kg_version": pkg_version("itext2kg"),
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": rec.get("source"),
        "input_sha256": sha(text),
        "input_text": text,
        "output": out.model_dump(),
    }
    cache.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, c, False


# ============ 4. Embedding ============
def make_embedder():
    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(model=EMBED_MODEL)

    def embed(texts: List[str]) -> np.ndarray:
        v = np.array(emb.embed_documents(texts), dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return embed


# ============ 5. Structural layer catalog (Neo4j or snapshot) ============
class Catalog:
    """models: name -> label; fields: (model, name) -> label;
    own[m]: models that module m defines, extends or adds fields to; dep[m]: models defined by its (transitive) dependencies."""

    def __init__(self, models, fields, own, dep):
        self.models, self.fields, self.own, self.dep = models, fields, own, dep
        self.label_index = {}
        for name, lab in models.items():
            self.label_index.setdefault(norm(lab), []).append(name)
        self.fields_by_model = {}
        for (mo, f), lab in fields.items():
            self.fields_by_model.setdefault(mo, {})[f] = lab


def catalog_from_neo4j(driver, modules):
    def q(cypher, **p):
        return driver.execute_query(cypher, database_="neo4j", **p)[0]
    models = {r["n"]: r["l"] for r in q("MATCH (m:Model) RETURN m.name AS n, m.label AS l")}
    fields = {(r["mo"], r["n"]): r["l"] for r in
              q("MATCH (f:Field) RETURN f.model AS mo, f.name AS n, f.label AS l")}
    own, dep = {}, {}
    for m in modules:
        own[m] = {r["x"] for r in q("""
            MATCH (:Module {name:$m})-[:DEFINES|EXTENDS]->(x:Model) RETURN x.name AS x
            UNION MATCH (:Module {name:$m})-[:ADDS_FIELD]->(f:Field) RETURN f.model AS x""", m=m)}
        dep[m] = {r["x"] for r in q("""
            MATCH (:Module {name:$m})-[:DEPENDS_ON*]->(:Module)-[:DEFINES]->(x:Model)
            RETURN DISTINCT x.name AS x""", m=m)}
    return Catalog(models, fields, own, dep)


def catalog_from_snapshot(path: Path, modules):
    """Reproduce the extract_schema.py v0.1 logic from the JSON snapshot (used by --dry-run without Neo4j)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    models = {r["name"]: r["label"] for r in d["models"]}
    fields = {(r["model"], r["name"]): r["label"] for r in d["fields"]}
    definer, first_field = {}, {}
    for r in d["model_owners"]:                       # already sorted by model, id
        definer.setdefault(r["model"], r["module"])
    for r in d["field_owners"]:
        first_field.setdefault((r["model"], r["name"]), r["module"])
    deps = {}
    for r in d["deps"]:
        deps.setdefault(r["module"], set()).add(r["depends_on"])
    own, dep = {}, {}
    for m in modules:
        own[m] = {r["model"] for r in d["model_owners"] if r["module"] == m}
        own[m] |= {mo for (mo, _), mod in first_field.items() if mod == m}
        closure, stack = set(), list(deps.get(m, ()))
        while stack:
            x = stack.pop()
            if x not in closure:
                closure.add(x)
                stack += deps.get(x, ())
        dep[m] = {mo for mo, mod in definer.items() if mod in closure}
    return Catalog(models, fields, own, dep)


# ============ 6. Entity linking ============
class Linker:
    def __init__(self, catalog: Catalog, embed, threshold: float):
        from itext2kg.itext2kg_star.graph_matching import Matcher
        self.cat, self.embed, self.th = catalog, embed, threshold
        self.matcher = Matcher()
        self.model_names = sorted(catalog.models)
        texts = [f"{catalog.models[n]} ({n})" for n in self.model_names]
        self.model_vecs = embed(texts)
        self.model_entities = [self._entity(t, v) for t, v in zip(texts, self.model_vecs)]

    @staticmethod
    def _entity(name, vec):
        from itext2kg.itext2kg_star.models import Entity
        from itext2kg.itext2kg_star.models.knowledge_graph import EntityProperties
        return Entity(label="Model", name=name, properties=EntityProperties(embeddings=vec))

    def _match(self, mention: str, cand_entities, cand_vecs, cand_keys):
        """iText2KG Matcher.find_match decides; top-3 is computed here for logging."""
        v = self.embed([mention])[0]
        sims = cand_vecs @ v
        top = np.argsort(-sims)[:3]
        top3 = [(cand_keys[i], round(float(sims[i]), 3)) for i in top]
        best = self.matcher.find_match(self._entity(mention, v), cand_entities, threshold=self.th)
        if best.name == mention or best.label != "Model":     # no match
            return None, top3
        i = [e.name for e in cand_entities].index(best.name)
        return (cand_keys[i], round(float(sims[i]), 3)), top3

    def link_model(self, mention: str, guess: Optional[str]):
        if guess and guess in self.cat.models:
            return guess, "exact_name", 1.0, []
        hits = self.cat.label_index.get(norm(mention), [])
        if len(hits) == 1:
            return hits[0], "exact_label", 1.0, []
        best, top3 = self._match(mention, self.model_entities, self.model_vecs, self.model_names)
        if best:
            return best[0], "embedding", best[1], top3
        return None, "new", (top3[0][1] if top3 else 0.0), top3

    def link_field(self, fm: FieldMention, linked_models: List[str]):
        """Return a list of (model, field, method). One option can exist on several models
        (e.g. allow_negative_stock on product, category, location): link it to every linked model that has the field."""
        fbm = self.cat.fields_by_model
        scope = list(dict.fromkeys(linked_models + ([fm.odoo_model] if fm.odoo_model in fbm else [])))
        name, method = None, None
        if fm.odoo_field and any(fm.odoo_field in fbm.get(m, {}) for m in scope):
            name, method = fm.odoo_field, "exact_name"
        else:
            for m in scope:
                for f, lab in fbm.get(m, {}).items():
                    if norm(lab) == norm(fm.mention):
                        name, method = f, "exact_label"
                        break
                if name:
                    break
        if not name:
            keys = [(m, f) for m in scope for f in fbm.get(m, {})]
            if keys:
                texts = [f"{fbm[m][f]} ({m}.{f})" for m, f in keys]
                vecs = self.embed(texts)
                ents = [self._entity(t, v) for t, v in zip(texts, vecs)]
                best, _ = self._match(fm.mention, ents, vecs, keys)
                if best:
                    name, method = best[0][1], "embedding"
        if not name:
            return []
        return [(m, name, method) for m in scope if name in fbm.get(m, {})]


# ============ 7. Ghi Neo4j ============
NEO4J_SETUP = [
    "CREATE CONSTRAINT function_key IF NOT EXISTS FOR (f:Function) REQUIRE f.key IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
]
Q_CLEAR_MODULE = """
MATCH (:Module {name:$module})-[:PROVIDES]->(f:Function) DETACH DELETE f
"""
Q_CLEAR_ORPHAN_CONCEPTS = "MATCH (c:Concept) WHERE NOT (c)<-[:MENTIONS]-() DELETE c"
Q_MODULE_EMB = """
MATCH (m:Module {name:$module})
SET m.embedding = $embedding, m.text_chars = $chars, m.semantic_version = $version
"""
Q_FUNCTION = """
MATCH (m:Module {name:$module})
MERGE (f:Function {key:$key})
SET f += $props
MERGE (m)-[:PROVIDES]->(f)
"""
Q_INVOLVES = """
MATCH (f:Function {key:$key}), (x:Model {name:$model})
MERGE (f)-[r:INVOLVES]->(x)
SET r.mention = $mention, r.method = $method, r.score = $score
"""
Q_INVOLVES_FIELD = """
MATCH (f:Function {key:$key}), (x:Field {key:$fkey})
MERGE (f)-[r:INVOLVES_FIELD]->(x)
SET r.mention = $mention, r.method = $method
"""
Q_MENTIONS = """
MATCH (f:Function {key:$key})
MERGE (c:Concept {name:$name})
MERGE (f)-[r:MENTIONS]->(c)
SET r.mention = $mention
"""
Q_VECTOR_INDEX = """
CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.embedding
OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}
"""


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


# ============ 8. Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected-dir", default="data/collected")
    ap.add_argument("--cache-dir", default="data/semantic")
    ap.add_argument("--modules", nargs="*", help="default: every file in collected-dir")
    ap.add_argument("--threshold", type=float, default=0.7, help="Matcher cosine threshold (iText2KG default 0.7)")
    ap.add_argument("--refresh", action="store_true", help="call the LLM again even if a cache exists")
    ap.add_argument("--dry-run", action="store_true", help="do not write to Neo4j")
    ap.add_argument("--catalog", choices=["neo4j", "snapshot"], default=None,
                    help="structural layer source; default neo4j, or snapshot with --dry-run")
    ap.add_argument("--snapshot", default="data/snapshots/schema_20261109.json")
    args = ap.parse_args()

    today = f"{date.today():%Y%m%d}"
    col_dir, cache_dir = Path(args.collected_dir), Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path("data/snapshots").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in col_dir.glob("*.json") if not p.name.startswith("_"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    if args.modules:
        recs = [r for r in recs if r["module"] in args.modules]
    modules = [r["module"] for r in recs]
    catalog_src = args.catalog or ("snapshot" if args.dry_run else "neo4j")
    print(f"Extractor {EXTRACTOR_VERSION}: {len(recs)} module, LLM {LLM_MODEL}, embedding {EMBED_MODEL}, "
          f"threshold {args.threshold}, structural layer from {catalog_src}{', DRY RUN' if args.dry_run else ''}")

    driver = None
    if catalog_src == "neo4j" or not args.dry_run:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
    cat = (catalog_from_neo4j(driver, modules) if catalog_src == "neo4j"
           else catalog_from_snapshot(Path(args.snapshot), modules))
    print(f"  Structural layer: {len(cat.models)} models, {len(cat.fields)} fields")

    t0 = time.time()
    embed = make_embedder()
    print("1) Embedding the model catalog ...")
    linker = Linker(cat, embed, args.threshold)

    print("2) Extracting functions with the LLM (DocumentsDistiller) ...")
    results, link_rows, n_llm_calls = [], [], 0
    for rec in recs:
        m = rec["module"]
        try:
            out, meta, cached = get_functions(make_llm, rec, cache_dir, args.refresh)
        except DailyQuotaExceeded as e:
            done = sorted(p.name.split(".")[0] for p in cache_dir.glob("*.llm.json"))
            raise SystemExit(f"\nSTOPPED: daily Gemini quota exhausted ({LLM_MODEL}).\n"
                             f"  Modules with a cache: {done or 'none'}\n"
                             f"  Rerun this command after the quota resets; cached modules are not called again.\n"
                             f"  Details: {e}")
        n_llm_calls += 0 if cached else 1
        text_n = norm(meta["input_text"])
        funcs = []
        for fn in out.functions:
            key = f"{m}:{slug(fn.name)}"
            ev_ok = norm(fn.evidence) in text_n
            obj_links = []
            for om in fn.objects:
                target, method, score, top3 = linker.link_model(om.mention, om.odoo_model)
                where = ("own" if target in cat.own[m] else "dependency" if target in cat.dep[m]
                         else "outside") if target else "-"
                obj_links.append(dict(mention=om.mention, guess=om.odoo_model, target=target,
                                      method=method, score=score, where=where))
                link_rows.append(dict(module=m, function=fn.name, mention=om.mention,
                                      llm_guess=om.odoo_model or "",
                                      guess_in_kg=("" if not om.odoo_model else
                                                   "yes" if om.odoo_model in cat.models else "no"),
                                      linked_model=target or "",
                                      method=method, score=score, structural=where,
                                      top3="; ".join(f"{k}={s}" for k, s in top3), gold_model=""))
            linked_models = list(dict.fromkeys(l["target"] for l in obj_links if l["target"]))
            fld_links = [dict(mention=fm.mention, model=mo, field=f, method=mt)
                         for fm in fn.fields for mo, f, mt in linker.link_field(fm, linked_models)]
            funcs.append(dict(key=key, fn=fn, evidence_found=ev_ok, objects=obj_links, fields=fld_links))
        results.append(dict(module=m, rec=rec, meta=meta, cached=cached, funcs=funcs))
        print(f"  {m:35} {len(funcs)} functions{' (cache)' if cached else ''}")
        for f in funcs:
            objs = ", ".join(f"{o['mention']}->{o['target'] or 'NEW'}[{o['method']}]" for o in f["objects"])
            flds = ", ".join(sorted({f"{x['model']}.{x['field']}" for x in f["fields"]}))
            print(f"    - {f['fn'].name} | evidence {'found' if f['evidence_found'] else 'NOT found'} in the text")
            print(f"      objects: {objs or '-'}")
            if flds:
                print(f"      field: {flds}")

    # --- Embeddings of functions and modules ---
    all_funcs = [f for r in results for f in r["funcs"]]
    fvecs = embed([f"{f['fn'].name}. {f['fn'].description}" for f in all_funcs]) if all_funcs else []
    mvecs = embed([build_input_text(r["rec"]) for r in results]) if results else []

    # --- Ghi Neo4j ---
    if not args.dry_run:
        print("3) Writing to Neo4j ...")
        run = lambda q, **p: driver.execute_query(q, database_="neo4j", **p)  # noqa: E731
        for c in NEO4J_SETUP:
            run(c)
        i = 0
        for r, mv in zip(results, mvecs):
            m = r["module"]
            run(Q_CLEAR_MODULE, module=m)
            run(Q_MODULE_EMB, module=m, embedding=mv.tolist(), chars=len(r["meta"]["input_text"]),
                version=EXTRACTOR_VERSION)
            for f in r["funcs"]:
                fn = f["fn"]
                props = dict(module=m, name=fn.name, description=fn.description, evidence=fn.evidence,
                             evidence_found=f["evidence_found"], section=fn.section,
                             embedding=fvecs[i].tolist(), llm=r["meta"]["llm"],
                             extracted_at=r["meta"]["extracted_at"], version=EXTRACTOR_VERSION,
                             source_commit=(r["meta"].get("source") or {}).get("commit"))
                i += 1
                run(Q_FUNCTION, module=m, key=f["key"], props=props)
                for o in f["objects"]:
                    if o["target"]:
                        run(Q_INVOLVES, key=f["key"], model=o["target"], mention=o["mention"],
                            method=o["method"], score=o["score"])
                    else:
                        run(Q_MENTIONS, key=f["key"], name=norm(o["mention"]), mention=o["mention"])
                for x in f["fields"]:
                    run(Q_INVOLVES_FIELD, key=f["key"], fkey=f"{x['model']}:{x['field']}",
                        mention=x["mention"], method=x["method"])
        run(Q_CLEAR_ORPHAN_CONCEPTS)
        dim = len(mvecs[0]) if len(mvecs) else 1024
        for name, label in [("function_embedding", "Function"), ("module_embedding", "Module")]:
            run(Q_VECTOR_INDEX.format(name=name, label=label, dim=dim))
        print(f"  Wrote {len(all_funcs)} functions; created vector indexes (dim {dim})")

    # --- Statistics ---
    def count(items, key):
        out = {}
        for it in items:
            out[it[key]] = out.get(it[key], 0) + 1
        return out

    per_module = {}
    for r in results:
        m = r["module"]
        objs = [o for f in r["funcs"] for o in f["objects"]]
        linked = {o["target"] for o in objs if o["target"]}
        per_module[m] = dict(
            text_chars=len(r["meta"]["input_text"]),
            functions=len(r["funcs"]),
            evidence_found=sum(f["evidence_found"] for f in r["funcs"]),
            object_mentions=len(objs),
            methods=count(objs, "method"),
            structural=count([o for o in objs if o["target"]], "where"),
            field_links=len({(x["model"], x["field"]) for f in r["funcs"] for x in f["fields"]}),
            own_models=sorted(cat.own[m]),
            own_models_covered=sorted(linked & cat.own[m]),
            llm_cached=r["cached"],
        )
    objs_all = [o for r in results for f in r["funcs"] for o in f["objects"]]
    linked_all = [o for o in objs_all if o["target"]]
    totals = dict(
        modules=len(results), functions=len(all_funcs),
        evidence_found=sum(f["evidence_found"] for f in all_funcs),
        object_mentions=len(objs_all),
        methods=count(objs_all, "method"),
        linked_rate=round(len(linked_all) / len(objs_all), 3) if objs_all else None,
        structural=count(linked_all, "where"),
        structural_consistency=round(sum(o["where"] != "outside" for o in linked_all) / len(linked_all), 3)
        if linked_all else None,
        llm_calls=n_llm_calls, seconds=round(time.time() - t0, 1),
    )
    print("4) Statistics:")
    print(f"  {totals['functions']} functions / {totals['modules']} modules; evidence found in the text "
          f"{totals['evidence_found']}/{totals['functions']}")
    print(f"  {totals['object_mentions']} objects; linking methods: {totals['methods']}")
    print(f"  Linked to a Model: {totals['linked_rate']}; consistent with the structural layer (own or dependency): "
          f"{totals['structural_consistency']} {totals['structural']}")
    print(f"  New LLM calls: {n_llm_calls}; time {totals['seconds']} s")

    stats = dict(date=today, extractor_version=EXTRACTOR_VERSION, llm=LLM_MODEL, embedding=EMBED_MODEL,
                 threshold=args.threshold, catalog=catalog_src, dry_run=args.dry_run,
                 totals=totals, per_module=per_module)
    suffix = "_dryrun" if args.dry_run else ""
    (Path("data/snapshots") / f"semantic_stats_{today}{suffix}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    links_csv = cache_dir / f"links_{today}{suffix}.csv"
    with links_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(link_rows[0].keys()) if link_rows else ["module"])
        w.writeheader()
        w.writerows(link_rows)
    print(f"  Saved: data/snapshots/semantic_stats_{today}{suffix}.json, {links_csv}")
    print("  Next step: fill the gold_model column in the CSV file, then run eval_entity_links.py")
    if driver:
        driver.close()


if __name__ == "__main__":
    main()
