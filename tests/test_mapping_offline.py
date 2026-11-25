"""
Offline test of the Mapping Agent (no Neo4j, no Ollama, no Gemini). Runs in a sandbox with a THROWAWAY
PostgreSQL that holds the mock Odoo catalog (tests/load_mock_catalog.py):
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/load_mock_catalog.py data/snapshots/schema_20261124.json
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/test_mapping_offline.py

Fakes:
  - embedding: hashed bag of words (cosine = word overlap), deterministic
  - LLM      : finds model labels written in the requirement (no real understanding)
The scores printed here only check the plumbing; they are NOT results for the report.
Checks:
  1. the graph view rebuilt from the snapshot has the counts of the real KG (kg_stats_20261124.json)
  2. cov, prox and the delegated-model rule behave as designed on hand-made cases
  3. with beta = gamma = 0 and no LLM, the module ranking of P equals the ranking of B1 (same documents)
  4. a full run on 20 requirements writes a predictions file that evaluate.py scores
  5. a second run uses the LLM cache (0 calls); tune.py runs on the dev split with 0 calls
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/mapping", "src/baselines", "src/extractor", "src/evaluation"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import b1_vector_rag as b1  # noqa: E402
import mapping_agent as ma  # noqa: E402
import tune  # noqa: E402
from graph_view import GraphView  # noqa: E402
from metrics import evaluate, load_requirements  # noqa: E402

PG = dict(host=os.getenv("PG_HOST", "/tmp"), port=int(os.getenv("PG_PORT", "5433")),
          dbname=os.getenv("PG_DB", "kg_mock"), user=os.getenv("PG_USER", "odoo"), password="")
assert PG["dbname"] != "kg_demo", "use the throwaway database"
SNAP = "data/snapshots/schema_20261124.json"
FAILS = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def fake_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(512)
        for w in re.findall(r"[a-z]+", t.lower()):
            w = w[:-1] if w.endswith("s") and len(w) > 3 else w
            v[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 512] += 1
        out.append(v if v.any() else np.ones(512))
    return out


g0 = GraphView.from_snapshot(SNAP, "data/semantic")
LABELS = sorted(((x, g0.models[x]["label"].lower()) for x in g0.models), key=lambda t: -len(t[1]))


def fake_llm(prompt):
    text = prompt.split("<<<")[1].split(">>>")[0].strip()
    low = text.lower()
    objs = [{"mention": lab, "odoo_model": x} for x, lab in LABELS if len(lab) > 4 and re.search(rf"\b{re.escape(lab)}\b", low)]
    words = [w for w in re.findall(r"[a-z]{5,}", low)][:6]
    return {"normalized_en": text, "low_level_keywords": words, "high_level_keywords": words[:2], "objects": objs[:5]}


# 1. graph counts
stats = json.loads(Path("data/snapshots/kg_stats_20261124.json").read_text())["graph"]
s = g0.stats()
check(s["modules"] == stats["Module"] and s["models"] == stats["Model"] and s["functions"] == stats["Function"],
      f"graph view: {s['modules']} modules, {s['models']} models, {s['functions']} functions as in the KG")
check(s["links_non_technical"] == stats["LINKS_TO"] - 851, f"non-technical LINKS_TO {s['links_non_technical']} = 2580 - 851")
inv = sum(len(f["involves"]) for f in g0.functions.values())
check(inv == stats["INVOLVES"], f"INVOLVES {inv} = {stats['INVOLVES']}")
check(sum(len(g0.code_models(m)) for m in g0.modules) == stats["DEFINES"] + stats["EXTENDS"],
      "DEFINES + EXTENDS pairs = 778")

# 2. hand-made cases
tmp = Path(tempfile.mkdtemp())
res, embed = ma.load_resources("snapshot", SNAP, "data/semantic", "data/collected_v1", str(tmp / "cache"),
                               fake_embed, PG)
p = ma.Params()
an = {"normalized_en": "Block outgoing shipments when there is not enough stock on hand",
      "low_level_keywords": ["negative stock"], "high_level_keywords": ["inventory control"],
      "objects": [{"mention": "shipment", "odoo_model": "stock.picking"},
                  {"mention": "stock on hand", "odoo_model": "stock.quant"},
                  {"mention": "product", "odoo_model": "product.template"}]}
f = ma.features(res, an["normalized_en"], an, p)
st, nn = f["modules"]["stock"], f["modules"]["stock_no_negative"]
check(st["cov_ratio"] == 1.0 and nn["cov_ratio"] < 1.0, f"cov ratio favours the big module: stock {st['cov_ratio']}, "
      f"stock_no_negative {nn['cov_ratio']}")
check(nn["cov_ochiai"] > st["cov_ochiai"], f"cov ochiai corrects it: stock {st['cov_ochiai']}, "
      f"stock_no_negative {nn['cov_ochiai']}")
check(nn["d"] == 1 and nn["prox"] == 0.5, f"module touching an entry model: d = {nn['d']}, prox = {nn['prox']}")
dist_ok = all(f["modules"][m]["d"] is None or f["modules"][m]["d"] >= 1 for m in f["modules"])
check(dist_ok, "every distance >= 1")
rows = {r["model"]: r for r in nn["models"]}
check(rows["product.product"]["code"] is False and rows["stock.quant"]["code"] is True,
      "product.product is delegated-only for stock_no_negative (R-28), stock.quant has code")
dec = ma.decide(f, replace(p, delegated_penalty=5.0, m_models=10), res.g)
nn_ans = next((a for a in dec["answer"] if a["module"] == "stock_no_negative"), None)
if nn_ans:
    check(nn_ans["models"][-1]["model"] == "product.product", "a large delegated penalty puts product.product last")

# 3. P without graph and LLM = B1
reqs = load_requirements("data/eval/requirements_v0.csv")
out_b1, info_b1 = b1.run("en", 1, 3, "data/eval/requirements_v0.csv", Path("data/collected_v1"), tmp / "runs",
                         tmp / "cache", fake_embed, PG)
p0 = replace(p, beta=0.0, gamma=0.0)
same = 0
for r, it in zip(reqs, info_b1["items"]):
    a0 = {"normalized_en": r["requirement_en"], "low_level_keywords": [], "high_level_keywords": [], "objects": []}
    f0 = ma.features(res, r["requirement_en"], a0, p0)
    rank_doc = sorted(f0["modules"], key=lambda m: (-f0["modules"][m]["sim_doc"], m))
    same += rank_doc[:5] == it["module_ranking"][:5]
check(same == len(reqs), f"sim_doc ranking = B1 ranking (top 5) on {same}/{len(reqs)} requirements")

# 4. full run with the fake LLM, then evaluate
args = ma.build_parser().parse_args(["--lang", "en", "--source", "snapshot", "--out-root", str(tmp / "runs"),
                                     "--llm-cache", str(tmp / "llm"), "--cache-dir", str(tmp / "cache"),
                                     "--max-llm-calls", "40"])
out1, info1 = ma.run(args, fake_embed, fake_llm, PG)
check(len(info1["items"]) == 20 and info1["llm_calls"] == 20, f"20 requirements, {info1['llm_calls']} LLM calls")
preds = {it["id"]: it for it in info1["items"]}
ev = evaluate(reqs, preds)["overall"]
check(ev["n"] == 20, f"evaluate: hit@1 {ev['hit@1']}, MRR {ev['mrr']}, model F1 {ev['model_f1']} (fake, not for the report)")
it = preds["Q05"]
check(all(k in it for k in ("module_ranking", "modules", "models", "answer", "entry_functions", "entry_models")),
      "prediction items have the evaluate.py keys and the explanation keys")
check(len(it["module_ranking"]) == 67, "ranking covers the 67 installed modules")

# 5. cache and tuning
out2, info2 = ma.run(args, fake_embed, fake_llm, PG)
check(info2["llm_calls"] == 0 and info2["items"][0]["llm_cached"], "second run: 0 LLM calls (cache)")
check([i["modules"] for i in info2["items"]] == [i["modules"] for i in info1["items"]], "second run gives the same answers")
info_t = tune.main(["--lang", "en", "--source", "snapshot", "--llm-cache", str(tmp / "llm"),
                    "--cache-dir", str(tmp / "cache"), "--out", str(tmp / "params.json")], fake_embed, fake_llm, PG)
check(info_t["best"]["objective"] >= info_t["default"]["objective"], "tune: best >= default on dev")
args.params = str(tmp / "params.json")
args.split = "test"
out3, info3 = ma.run(args, fake_embed, fake_llm, PG)
check(len(info3["items"]) == 16 and info3["llm_calls"] == 0, "test split with tuned params: 16 requirements, 0 calls")

print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
