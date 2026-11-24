"""
Test of LightRAG (lightrag-hku 1.5.7) with the project's Neo4j database.

Questions to answer:
  1. Which labels, keys and properties does LightRAG use for the nodes and relationships it writes to Neo4j?
  2. Do LightRAG nodes connect by themselves to the 2-layer KG (Module, Model, Field, Function)?
  3. Do the entity names extracted by LightRAG match existing Model or Module names or labels?
  4. Does LightRAG collide with existing labels or change the numbers of the existing KG?

Runs on 1 short module only to save quota (default stock_picking_back2draft, 206 characters).
Hard limit on LLM calls (default 8). Normally only about 2 to 3 calls.

Usage (in ~/agentic-graphrag, venv activated):
  python src/experiments/test_lightrag_neo4j.py --inspect      # only count the KG, no LLM call
  python src/experiments/test_lightrag_neo4j.py                # run LightRAG + report
  python src/experiments/test_lightrag_neo4j.py --llm ollama   # use a local LLM if the Gemini quota is exhausted
  python src/experiments/test_lightrag_neo4j.py --cleanup      # delete LightRAG nodes, count again

Note: the main KG can be fully rebuilt if something goes wrong:
  python src/extractor/extract_schema.py --reset && python src/extractor/extract_semantic.py
  (the semantic layer uses its cache, 0 LLM calls)
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)  # .env wins over stale variables in the terminal

WORKSPACE = "lightrag_test"
os.environ.setdefault("NEO4J_DATABASE", "neo4j")     # Community edition has only 1 database
os.environ.setdefault("NEO4J_WORKSPACE", WORKSPACE)  # some LightRAG versions read this variable

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "thesis12345"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
MAX_CALLS = int(os.getenv("LIGHTRAG_MAX_CALLS", "8"))
WORK_DIR = Path("data/lightrag_test")
OUT_DIR = Path("data/experiments")

OUR_LABELS = ["Module", "Model", "Field", "Function", "Concept"]
OUR_RELS = ["DEPENDS_ON", "DEFINES", "EXTENDS", "HAS_FIELD", "ADDS_FIELD", "RELATES_TO",
            "LINKS_TO", "PROVIDES", "INVOLVES", "INVOLVES_FIELD", "MENTIONS"]

CALLS = {"llm": 0}


# ============ Neo4j: counts and checks ============
def neo4j_driver():
    from neo4j import GraphDatabase
    d = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
    d.verify_connectivity()
    return d


def q(driver, cypher, **p):
    return [r.data() for r in driver.execute_query(cypher, database_="neo4j", **p)[0]]


def our_counts(driver):
    """Node and relationship counts of the 2-layer KG (LightRAG nodes, recognised by the entity_id property, are skipped)."""
    nodes = {l: q(driver, f"MATCH (n:`{l}`) WHERE n.entity_id IS NULL RETURN count(n) AS c")[0]["c"]
             for l in OUR_LABELS}
    rels = {t: q(driver, f"MATCH ()-[r:`{t}`]->() RETURN count(r) AS c")[0]["c"] for t in OUR_RELS}
    lr = q(driver, "MATCH (n) WHERE n.entity_id IS NOT NULL RETURN count(n) AS c")[0]["c"]
    return {"nodes": nodes, "rels": rels, "lightrag_nodes": lr}


def print_counts(title, c):
    print(f"\n== {title} ==")
    print("  2-layer KG nodes: " + ", ".join(f"{k} {v:,}".replace(",", ".") for k, v in c["nodes"].items()))
    print("  Relationships   : " + ", ".join(f"{k} {v:,}".replace(",", ".") for k, v in c["rels"].items()))
    print(f"  LightRAG nodes (with entity_id): {c['lightrag_nodes']}")


def inspect_lightrag(driver):
    """Answer questions 1 to 4 with direct queries."""
    rep = {}
    rep["labels"] = q(driver, """
        MATCH (n) WHERE n.entity_id IS NOT NULL
        UNWIND labels(n) AS l RETURN l AS label, count(*) AS c ORDER BY c DESC""")
    rep["node_keys"] = q(driver, """
        MATCH (n) WHERE n.entity_id IS NOT NULL
        UNWIND keys(n) AS k RETURN k AS key, count(*) AS c ORDER BY key""")
    rep["rels_inside"] = q(driver, """
        MATCH (a)-[r]->(b) WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL
        RETURN type(r) AS type, count(*) AS c""")
    rep["rel_keys"] = q(driver, """
        MATCH (a)-[r]->(b) WHERE a.entity_id IS NOT NULL AND b.entity_id IS NOT NULL
        UNWIND keys(r) AS k RETURN k AS key, count(*) AS c ORDER BY key""")
    rep["rels_to_our_kg"] = q(driver, """
        MATCH (a)-[r]-(b) WHERE a.entity_id IS NOT NULL AND b.entity_id IS NULL
        RETURN type(r) AS type, labels(b) AS labels, count(*) AS c""")
    rep["label_collision"] = q(driver, """
        MATCH (n) WHERE n.entity_id IS NOT NULL
          AND any(l IN labels(n) WHERE l IN $ours)
        RETURN n.entity_id AS entity, labels(n) AS labels""", ours=OUR_LABELS)
    rep["name_match"] = q(driver, """
        MATCH (n) WHERE n.entity_id IS NOT NULL
        OPTIONAL MATCH (m:Model) WHERE m.entity_id IS NULL
          AND (toLower(m.name) = toLower(n.entity_id) OR toLower(m.label) = toLower(n.entity_id))
        OPTIONAL MATCH (mo:Module) WHERE mo.entity_id IS NULL
          AND (toLower(mo.name) = toLower(n.entity_id) OR toLower(mo.label) = toLower(n.entity_id))
        RETURN n.entity_id AS entity, n.entity_type AS type,
               collect(DISTINCT m.name) AS models, collect(DISTINCT mo.name) AS modules
        ORDER BY entity""")
    return rep


def print_report(rep):
    print("\n== (1) Nodes and relationships written by LightRAG ==")
    print("  Labels     : " + ", ".join(f"{r['label']} ({r['c']})" for r in rep["labels"]))
    print("  Properties : " + ", ".join(r["key"] for r in rep["node_keys"]))
    print("  Relations  : " + (", ".join(f"{r['type']} ({r['c']})" for r in rep["rels_inside"]) or "none"))
    print("  Relationship properties: " + (", ".join(r["key"] for r in rep["rel_keys"]) or "-"))

    print("\n== (2) Relationships to the 2-layer KG ==")
    if rep["rels_to_our_kg"]:
        for r in rep["rels_to_our_kg"]:
            print(f"  {r['type']} -> {r['labels']}: {r['c']}")
    else:
        print("  None (LightRAG does not connect to Module, Model, Field, Function by itself)")

    print("\n== (3) Entity names compared with existing Model, Module ==")
    hit = 0
    for r in rep["name_match"]:
        found = r["models"] + r["modules"]
        hit += bool(found)
        print(f"  {r['entity'][:45]:45} [{r['type'] or '-'}] -> {', '.join(found) or 'no match'}")
    n = len(rep["name_match"])
    print(f"  Name matches: {hit}/{n}")
    rep["name_match_rate"] = f"{hit}/{n}"

    print("\n== (4) Label collisions with the 2-layer KG ==")
    if rep["label_collision"]:
        for r in rep["label_collision"]:
            print(f"  {r['entity']}: {r['labels']}")
        print("  WARNING: LightRAG nodes carry labels of the 2-layer KG. MATCH (c:Concept) queries ...")
        print("  in extract_semantic.py would touch these nodes.")
    else:
        print("  None")


# ============ LightRAG ============
def make_llm(kind):
    """LLM function with the signature LightRAG expects. Hard call limit to protect the quota."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    if kind == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        chat = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0,
                                      timeout=int(os.getenv("LLM_TIMEOUT", "300")))
    else:
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_LLM", "qwen2.5:7b")
        chat = ChatOllama(model=model, temperature=0)

    async def llm_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
        if CALLS["llm"] >= MAX_CALLS:
            raise RuntimeError(f"Reached {MAX_CALLS} LLM calls, stopping to protect the quota")
        CALLS["llm"] += 1
        msgs = [SystemMessage(system_prompt)] if system_prompt else []
        for h in history_messages or []:
            msgs.append((HumanMessage if h.get("role") == "user" else AIMessage)(h.get("content", "")))
        msgs.append(HumanMessage(prompt))
        t = time.time()
        resp = await chat.ainvoke(msgs)
        content = resp.content
        if isinstance(content, list):  # some Gemini models return a list of parts
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        print(f"  [LLM {CALLS['llm']}] {model}, {time.time() - t:.1f} s, "
              f"{'keyword extraction' if keyword_extraction else 'entity extraction/other'}")
        return content

    return llm_func, model


def make_embedding():
    from langchain_ollama import OllamaEmbeddings
    from lightrag.utils import EmbeddingFunc
    kw = {"base_url": os.getenv("OLLAMA_BASE_URL")} if os.getenv("OLLAMA_BASE_URL") else {}
    emb = OllamaEmbeddings(model=EMBED_MODEL, **kw)

    async def embed(texts):
        return np.array(await emb.aembed_documents(list(texts)), dtype=np.float32)

    try:
        return EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=embed)
    except TypeError:
        return EmbeddingFunc(embedding_dim=1024, func=embed)


def module_text(module):
    sys.path.insert(0, str(Path("src/extractor").resolve()))
    from extract_semantic import build_input_text  # same text as the semantic layer
    rec = json.loads((Path("data/collected") / f"{module}.json").read_text(encoding="utf-8"))
    return build_input_text(rec)


async def run_lightrag(args):
    from lightrag import LightRAG, QueryParam
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
    except ImportError:
        initialize_pipeline_status = None

    llm_func, llm_name = make_llm(args.llm)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    common = dict(working_dir=str(WORK_DIR), llm_model_func=llm_func, llm_model_name=llm_name,
                  llm_model_max_async=1, embedding_func=make_embedding(),
                  graph_storage="Neo4JStorage", entity_extract_max_gleaning=0)
    try:
        rag = LightRAG(workspace=WORKSPACE, **common)
    except TypeError:
        print("  (this LightRAG version has no workspace parameter, using the default)")
        rag = LightRAG(**common)
    await rag.initialize_storages()
    if initialize_pipeline_status:
        await initialize_pipeline_status()

    text = module_text(args.module)
    print(f"\n== Running LightRAG: module {args.module} ({len(text)} characters), LLM {llm_name} ==")
    t0 = time.time()
    await rag.ainsert(text)
    t_insert = time.time() - t0
    print(f"  Insert done: {t_insert:.1f} s, {CALLS['llm']} LLM calls")

    answers = {}
    for mode in ["local", "hybrid"]:
        try:
            ctx = await rag.aquery(args.query, param=QueryParam(mode=mode, only_need_context=True))
            answers[mode] = str(ctx)
            print(f"\n== Retrieval mode={mode} (context only, no answer generation) ==")
            print("  " + str(ctx)[:1500].replace("\n", "\n  "))
        except Exception as e:
            answers[mode] = f"ERROR: {e!r}"
            print(f"  mode={mode}: ERROR {e!r}")
    await rag.finalize_storages()
    return dict(module=args.module, chars=len(text), llm=llm_name, insert_seconds=round(t_insert, 1),
                query=args.query, retrieval=answers)


# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="stock_picking_back2draft")
    ap.add_argument("--query", default="Allow users to reset a cancelled transfer back to draft")
    ap.add_argument("--llm", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument("--inspect", action="store_true", help="only count the KG, no LLM call")
    ap.add_argument("--cleanup", action="store_true", help="delete LightRAG nodes and the working folder")
    args = ap.parse_args()

    driver = neo4j_driver()
    before = our_counts(driver)
    print_counts("Current KG", before)

    if args.inspect:
        if before["lightrag_nodes"]:
            print_report(inspect_lightrag(driver))
        return

    if args.cleanup:
        q(driver, "MATCH (n) WHERE n.entity_id IS NOT NULL DETACH DELETE n")
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
        after = our_counts(driver)
        print_counts("After cleanup", after)
        same = after["nodes"] == before["nodes"] and after["rels"] == before["rels"]
        print(f"\n2-layer KG {'UNCHANGED' if same else 'CHANGED, please check'}")
        return

    if before["lightrag_nodes"]:
        raise SystemExit("LightRAG nodes from a previous run exist. Run --cleanup before trying again.")

    run = asyncio.run(run_lightrag(args))
    after = our_counts(driver)
    print_counts("KG after running LightRAG", after)
    unchanged = after["nodes"] == before["nodes"] and after["rels"] == before["rels"]
    print(f"  2-layer KG {'unchanged' if unchanged else 'CHANGED'} compared with before the run")
    rep = inspect_lightrag(driver)
    print_report(rep)
    print(f"\nTotal LLM calls: {CALLS['llm']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lightrag_neo4j_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(dict(run=run, llm_calls=CALLS["llm"], before=before, after=after,
                                   kg_unchanged=unchanged, report=rep),
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out}")
    driver.close()


if __name__ == "__main__":
    main()
