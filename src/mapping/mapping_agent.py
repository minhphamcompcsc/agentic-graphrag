"""
Mapping Agent (configuration P, first version map-v0.1): maps a BA requirement to Odoo modules and models
by searching the 2-layer KG.

Steps for one requirement:
  1. Analysis (1 LLM call, cached; src/mapping/analyze.py): English restatement, low-level keywords,
     high-level keywords, business objects with a model guess.
  2. Entry points (2-level retrieval, idea of LightRAG, D13):
       - entry functions: Function nodes closest to the requirement + low-level keywords (top n_functions)
       - entry models   : objects of step 1 linked to Model nodes (exact name, exact label, embedding)
  3. Score every installed module m:
       score(m) = alpha * sim(q,m) + beta * cov(q,m) + gamma * prox(m)
       sim  : max cosine between the query texts and (a) the module document (same text as B1,
              query texts include the high-level keywords) or (b) a Function of the module
       cov  : overlap between the entry models Q and the module footprint T(m) (DEFINES, EXTENDS, ADDS_FIELD)
              "ratio" = |Q n T| / |Q| (proposal), "ochiai" = |Q n T| / sqrt(|Q| |T|) (default: does not favour
              modules with a very large footprint, e.g. stock touches most stock models)
       prox : 1 / (1 + d), d = shortest path from the module to an entry point through
              Function-INVOLVES-Model and Model-LINKS_TO-Model edges (technical edges excluded,
              context fields company_id, currency_id, user_id cost context_weight instead of 1);
              0 if d > max_distance
  4. Answer: top k modules; for each, its models ranked by
       rel(x) = cos(query, "label (model): fields the module adds on it")
                + lam_entry * [x is an entry model] + lam_function * [an entry function of m involves x]
                - delegated_penalty * [x only receives fields through _inherits, no code there]
     top m_models of them; plus direct dependencies and the KG path that explains the choice.
  5. Low confidence (best score < tau): flagged; the agentic loop (src/verifier/run_pipeline.py) then asks the
     Collector for more evidence before the Verifier judges the candidates.

Usage (in ~/agentic-graphrag, venv activated, Neo4j and Odoo running):
  python src/mapping/mapping_agent.py --lang en --split dev --dry-llm          # check, 0 LLM calls
  python src/mapping/mapping_agent.py --lang en --split dev                    # 4 calls (cached afterwards)
  python src/mapping/mapping_agent.py --lang en --params data/mapping/params_dev.json --run 1
  python src/evaluation/evaluate.py data/runs/p_en_<date>_run1/predictions.json --split test
"""
import argparse
import heapq
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/mapping", "src/baselines", "src/extractor", "src/evaluation"):
    sys.path.insert(0, str(ROOT / sub))
from analyze import PROMPT_VERSION, Analyzer, DailyQuotaExceeded  # noqa: E402
from graph_view import SYSTEM_FIELDS, GraphView  # noqa: E402

MAPPING_VERSION = "map-v0.1"
CONTEXT_FIELDS = {"company_id", "currency_id", "user_id"}     # second-tier context fields (R-14, open question 1)


@dataclass
class Params:
    # decision parameters (tuned on the dev split, src/mapping/tune.py)
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.25
    cov: str = "ochiai"            # ochiai | ratio
    k_modules: int = 1
    m_models: int = 3
    lam_entry: float = 0.2
    lam_function: float = 0.1
    delegated_penalty: float = 0.3
    tau: float = 0.0               # Collector trigger (agentic loop, next version)
    # feature parameters (fixed before tuning)
    n_functions: int = 5
    max_distance: int = 3
    context_weight: float = 2.0
    link_threshold: float = 0.75   # cosine needed to link an object to a model by embedding


def norm(s):
    import re
    s = re.sub(r"[*_`\\]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


# ============ resources shared by all requirements ============
class Resources:
    """Graph view + embeddings of module documents, functions, models and (module, model) pairs."""

    def __init__(self, graph: GraphView, module_docs: dict, embed):
        self.g, self.embed = graph, embed
        self.docs = module_docs                      # module documents (same text as B1), also read by the Verifier
        self.mod_names = sorted(m for m in graph.modules if m in module_docs)
        self.mod_vecs = embed([module_docs[m] for m in self.mod_names])
        self.fun_keys = sorted(graph.functions)
        self.fun_vecs = (embed([f"{graph.functions[k]['name']}. {graph.functions[k]['description']}"
                                for k in self.fun_keys]) if self.fun_keys else np.zeros((0, 1)))
        self.model_names = sorted(graph.models)
        self.model_vecs = embed([f"{graph.models[x]['label']} ({x})" for x in self.model_names])
        self.label_index = {}
        for x in self.model_names:
            self.label_index.setdefault(norm(graph.models[x]["label"]), []).append(x)
        # one document per (module, model of its footprint): model label + fields the module adds there
        self.pairs = [(m, x) for m in self.mod_names for x in sorted(graph.footprint(m)) if x in graph.models]
        self.pair_index = {p: i for i, p in enumerate(self.pairs)}
        self.pair_vecs = embed([self.pair_doc(m, x) for m, x in self.pairs])

    def pair_doc(self, m, x):
        labs = [lab for f, lab in self.g.added.get(m, {}).get(x, [])
                if f not in SYSTEM_FIELDS and not f.startswith(("message_", "activity_", "website_message"))]
        doc = f"{self.g.models[x]['label']} ({x})"
        return doc + (": " + ", ".join(labs[:15]) if labs else "")


# ============ features of one requirement (independent of the decision parameters) ============
def link_objects(res: Resources, objects, threshold):
    out = []
    for o in objects or []:
        mention, guess = o.get("mention", ""), o.get("odoo_model")
        if guess and guess in res.g.models:
            out.append({"mention": mention, "model": guess, "method": "exact_name", "score": 1.0})
            continue
        hits = res.label_index.get(norm(mention), [])
        if len(hits) == 1:
            out.append({"mention": mention, "model": hits[0], "method": "exact_label", "score": 1.0})
            continue
        v = res.embed([mention])[0]
        sims = res.model_vecs @ v
        i = int(np.argmax(sims))
        if sims[i] >= threshold:
            out.append({"mention": mention, "model": res.model_names[i], "method": "embedding",
                        "score": round(float(sims[i]), 4)})
        else:
            out.append({"mention": mention, "model": None, "method": "unlinked", "score": round(float(sims[i]), 4),
                        "nearest": res.model_names[i]})
    return out


def distances(g: GraphView, sources, context_weight):
    """Multi-source Dijkstra over Model and Function nodes. Returns dist and predecessor maps."""
    dist, pred, heap = {}, {}, []
    for s in sources:
        dist[s] = 0.0
        heap.append((0.0, s))
    heapq.heapify(heap)
    fun_of_model = {}
    for k, f in g.functions.items():
        for x in f["involves"]:
            fun_of_model.setdefault(x, []).append(k)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, math.inf):
            continue
        if u in g.functions:
            nbrs = [(x, 1.0, "INVOLVES") for x in g.functions[u]["involves"]]
        else:
            nbrs = [(v, context_weight if via in CONTEXT_FIELDS else 1.0, via) for v, via in g.adj.get(u, [])]
            nbrs += [(k, 1.0, "INVOLVES") for k in fun_of_model.get(u, [])]
        for v, w, via in nbrs:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], pred[v] = nd, (u, via)
                heapq.heappush(heap, (nd, v))
    return dist, pred


def path_to_entry(node, pred):
    """Path from a node back to its nearest entry point: [node, (via), node, ...]."""
    out = [node]
    while node in pred:
        node, via = pred[node]
        out += [f"-[{via}]-", node]
    return out


def features(res: Resources, req_text: str, analysis: dict, p: Params) -> dict:
    g = res.g
    q_texts = list(dict.fromkeys(t for t in [req_text, analysis.get("normalized_en", "")] if t))
    low = "; ".join(analysis.get("low_level_keywords") or [])
    high = "; ".join(analysis.get("high_level_keywords") or [])
    texts_fun = q_texts + ([low] if low else [])
    texts_mod = q_texts + ([high] if high else [])
    v_fun, v_mod = res.embed(texts_fun), res.embed(texts_mod)
    v_q = res.embed(q_texts)

    # entry functions (low level)
    s_f = (res.fun_vecs @ v_fun.T).max(axis=1) if len(res.fun_keys) else np.zeros(0)
    order = np.argsort(-s_f, kind="stable")[:p.n_functions]
    entry_funs = [{"key": res.fun_keys[i], "module": g.functions[res.fun_keys[i]]["module"],
                   "name": g.functions[res.fun_keys[i]]["name"], "score": round(float(s_f[i]), 4)} for i in order]
    # entry models (objects of the requirement)
    links = link_objects(res, analysis.get("objects"), p.link_threshold)
    Q = sorted({l["model"] for l in links if l["model"]})

    dist, pred = distances(g, Q + [f["key"] for f in entry_funs], p.context_weight)
    s_doc = (res.mod_vecs @ v_mod.T).max(axis=1)
    fun_best = {}
    for i, k in enumerate(res.fun_keys):
        m = g.functions[k]["module"]
        fun_best[m] = max(fun_best.get(m, -1.0), float(s_f[i]))
    entry_keys = {f["key"] for f in entry_funs}

    mods = {}
    for i, m in enumerate(res.mod_names):
        T = g.footprint(m)
        inter = sorted(set(Q) & T)
        cov_ratio = len(inter) / len(Q) if Q else 0.0
        cov_ochiai = len(inter) / math.sqrt(len(Q) * len(T)) if Q and T else 0.0
        nbrs = list(T) + g.functions_of(m)
        best = min(((dist[n], n) for n in nbrs if n in dist), default=(math.inf, None))
        d = 1 + best[0] if best[1] is not None else math.inf
        sd, sf = float(s_doc[i]), fun_best.get(m)
        # model relevance inside the footprint
        involved = {x for k in g.functions_of(m) if k in entry_keys for x in g.functions[k]["involves"]}
        code = g.code_models(m)
        mrows = []
        for x in sorted(T):
            j = res.pair_index.get((m, x))
            if j is None:
                continue
            mrows.append({"model": x, "cos": round(float((v_q @ res.pair_vecs[j]).max()), 4),
                          "entry": x in Q, "function": x in involved, "code": x in code,
                          "relations": sorted(g.touch[m][x])})
        mods[m] = {"sim_doc": round(sd, 4), "sim_fun": None if sf is None else round(sf, 4),
                   "sim": round(max(sd, sf if sf is not None else -1.0), 4),
                   "cov_ratio": round(cov_ratio, 4), "cov_ochiai": round(cov_ochiai, 4), "cov_models": inter,
                   "d": None if d == math.inf else d,
                   "prox": round(1.0 / (1.0 + d), 4) if d <= p.max_distance else 0.0,
                   "path": ([m, "-[PROVIDES]-" if best[1] in g.functions else
                             "-[" + "|".join(sorted(g.touch[m][best[1]])) + "]-"] + path_to_entry(best[1], pred)
                            if best[1] else []),
                   "models": mrows}
    return {"q_texts": q_texts, "entry_functions": entry_funs, "entry_models": links, "Q": Q, "modules": mods}


# ============ decision (cheap, used by tuning) ============
def rank_models(feat: dict, m: str, p: Params) -> list:
    """Models of the footprint of module m, most relevant first (rel, see step 4). Also used by the Verifier."""
    rows = []
    for r in feat["modules"][m]["models"]:
        rel = (r["cos"] + p.lam_entry * r["entry"] + p.lam_function * r["function"]
               - p.delegated_penalty * (not r["code"]))
        rows.append({**r, "rel": round(rel, 4)})
    rows.sort(key=lambda r: (-r["rel"], r["model"]))
    return rows


def decide(feat: dict, p: Params, g: GraphView, exclude=()) -> dict:
    """exclude: modules rejected by the Verifier (agentic loop); they are moved to the end of the ranking and
    the answer is taken from the remaining modules."""
    scored = []
    for m, f in feat["modules"].items():
        cov = f["cov_ochiai"] if p.cov == "ochiai" else f["cov_ratio"]
        s = p.alpha * f["sim"] + p.beta * cov + p.gamma * f["prox"]
        scored.append((round(s, 6), m))
    scored.sort(key=lambda t: (t[1] in exclude, -t[0], t[1]))
    ranking = [m for _, m in scored]
    answer = []
    for s, m in [t for t in scored if t[1] not in exclude][:p.k_modules]:
        rows = rank_models(feat, m, p)
        answer.append({"module": m, "score": s, "models": rows[:p.m_models],
                       "depends": sorted(g.deps.get(m, ())), "path": feat["modules"][m]["path"],
                       "functions": [k for k in g.functions_of(m)]})
    models = list(dict.fromkeys(r["model"] for a in answer for r in a["models"]))
    kept = [t for t in scored if t[1] not in exclude]
    top = [{"module": m, "score": s, **{k: feat["modules"][m][k] for k in
            ("sim", "sim_doc", "sim_fun", "cov_ratio", "cov_ochiai", "cov_models", "prox", "d")}}
           for s, m in kept[:10]]
    return {"module_ranking": ranking, "modules": [a["module"] for a in answer], "models": models,
            "answer": answer, "top_modules": top,
            "low_confidence": bool(kept and kept[0][0] < p.tau)}


# ============ loading ============
def load_resources(source: str, snapshot: str, semantic_dir: str, collected_dir: str, cache_dir: str,
                   embed_fn=None, pg=None):
    from b1_vector_rag import CachedEmbedder, module_documents, read_odoo_rows
    if source == "neo4j":
        import os
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        load_dotenv(override=True)
        drv = GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                                   auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "")))
        g = GraphView.from_neo4j(drv)
        drv.close()
    else:
        g = GraphView.from_snapshot(snapshot, semantic_dir)
    mod_rows, _ = read_odoo_rows(pg)
    docs = {d["name"]: d["text"] for d in module_documents(mod_rows, Path(collected_dir))}
    embed = CachedEmbedder(Path(cache_dir), embed_fn)
    return Resources(g, docs, embed), embed


def run(args, embed_fn=None, llm_fn=None, pg=None):
    from metrics import load_requirements
    t0 = time.time()
    p = Params(**json.loads(Path(args.params).read_text())["params"]) if args.params else Params()
    res, embed = load_resources(args.source, args.snapshot, args.semantic_dir, args.collected_dir,
                                args.cache_dir, embed_fn, pg)
    print(f"Mapping {MAPPING_VERSION} ({args.lang}, split {args.split}, run {args.run}): graph from {res.g.source} "
          f"{res.g.stats()}")
    backend = "none" if args.dry_llm else args.llm
    an = Analyzer(backend, args.llm_model, args.run, args.llm_cache, args.max_llm_calls, llm_fn)
    reqs = load_requirements(args.requirements, args.split)
    items, stopped = [], None
    for r in reqs:
        text = r[f"requirement_{args.lang}"]
        try:
            analysis, cached = an.analyze(r["id"], args.lang, text)
        except DailyQuotaExceeded as e:
            stopped = str(e)
            break
        feat = features(res, text, analysis, p)
        dec = decide(feat, p, res.g)
        items.append({"id": r["id"], "query": text, "analysis": analysis, "llm_cached": cached,
                      "entry_functions": feat["entry_functions"], "entry_models": feat["entry_models"], **dec})
        a = dec["answer"][0]
        print(f"  {r['id']}: {a['module']} ({a['score']:.3f}) | models: {', '.join(dec['models'])}"
              f"{' | from cache' if cached else ''}")
    today = f"{date.today():%Y%m%d}"
    tag = "pnollm" if backend == "none" else "p"
    out_dir = Path(args.out_root) / f"{tag}_{args.lang}_{today}_{args.split}_run{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"config": "P" if backend != "none" else "P-noLLM", "lang": args.lang, "date": today, "run": args.run,
            "mapping_version": MAPPING_VERSION, "prompt_version": PROMPT_VERSION, "llm_backend": backend,
            "llm": an.model, "llm_calls": an.calls, "embedding": "see data/cache", "graph": res.g.stats(),
            "graph_source": res.g.source, "params": asdict(p), "params_file": args.params,
            "requirements_file": args.requirements, "split": args.split, "new_embeddings": embed.new,
            "seconds": round(time.time() - t0, 1), "stopped": stopped, "items": items}
    (out_dir / "predictions.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"New LLM calls {an.calls}; new embeddings {embed.new}; {info['seconds']} s")
    if stopped:
        print(f"STOPPED early: {stopped}\n  Rerun the same command after the quota resets; answers already "
              f"received are cached and are not called again.")
    print(f"Saved: {out_dir}/predictions.json\nNext: python src/evaluation/evaluate.py {out_dir}/predictions.json "
          f"--split {args.split}")
    return out_dir, info


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "vi"], default="en")
    ap.add_argument("--split", default="all", help="all, dev or test")
    ap.add_argument("--run", type=int, default=1, help="run number (1 to 3); each run has its own LLM cache")
    ap.add_argument("--params", help="JSON file from tune.py; default: Params() defaults")
    ap.add_argument("--llm", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument("--llm-model", default=None, help="default GEMINI_MODEL (gemini-3.6-flash) or OLLAMA_MODEL")
    ap.add_argument("--dry-llm", action="store_true", help="no LLM: query text only (0 calls)")
    ap.add_argument("--max-llm-calls", type=int, default=20)
    ap.add_argument("--llm-cache", default="data/mapping/llm")
    ap.add_argument("--source", choices=["neo4j", "snapshot"], default="neo4j")
    ap.add_argument("--snapshot", default="data/snapshots/schema_20261124.json")
    ap.add_argument("--semantic-dir", default="data/semantic")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--collected-dir", default="data/collected_v1")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out-root", default="data/runs")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
