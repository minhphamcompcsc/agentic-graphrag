"""
Offline test of the BA demo (src/demo/demo.py): no Neo4j, no Ollama, no Gemini. Same throwaway PostgreSQL
as tests/test_mapping_offline.py:
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/load_mock_catalog.py data/snapshots/schema_20261124.json
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/test_demo_offline.py

Fakes: hashed bag-of-words embedding, fake Mapping LLM and scripted Verifier LLM (as in
tests/test_verifier_offline.py). The answers printed here only check the plumbing; they are NOT results.
Checks:
  1. the demo gives the same answer as run_pipeline.py (configuration P) for a requirement of the evaluation
     set, reading the official caches with 0 LLM calls
  2. --dry-llm without any cache: 0 calls, Mapping without LLM (= P-noLLM answer), Verifier skipped
  3. a new requirement text: new calls go to the demo cache only (official caches unchanged);
     a rerun with --dry-llm uses the demo cache (0 calls, same answer)
  4. language detection; a text equal to an evaluation requirement reuses its id
  5. --max-llm-calls caps the calls; a Verifier rejection sends the next candidates to round 2 or is skipped
     when the budget is used up (Mapping answer kept, "not verified")
  6. output: text, Markdown and JSON contain the module, models with relations, fields, dependencies,
     KG path, verdict; --out writes the file; no secret default in the code
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/demo", "src/verifier", "src/mapping", "src/baselines", "src/extractor", "src/evaluation",
            "src/collector"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import demo  # noqa: E402
import mapping_agent as ma  # noqa: E402
import run_pipeline as rp  # noqa: E402
from graph_view import GraphView  # noqa: E402

PG = dict(host=os.getenv("PG_HOST", "/tmp"), port=int(os.getenv("PG_PORT", "5433")),
          dbname=os.getenv("PG_DB", "kg_mock"), user=os.getenv("PG_USER", "odoo"), password="")
assert PG["dbname"] != "kg_demo", "use the throwaway database"
SNAP = "data/snapshots/schema_20261124.json"
G0 = GraphView.from_snapshot(SNAP, "data/semantic")
LABELS = sorted(((x, G0.models[x]["label"].lower()) for x in G0.models), key=lambda t: -len(t[1]))
FAILS = []
COUNT = {"map": 0, "ver": 0}


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


def fake_llm(prompt):
    COUNT["map"] += 1
    text = prompt.split("<<<")[1].split(">>>")[0].strip()
    low = text.lower()
    objs = [{"mention": lab, "odoo_model": x} for x, lab in LABELS
            if len(lab) > 4 and re.search(rf"\b{re.escape(lab)}\b", low)]
    words = re.findall(r"[a-z]{5,}", low)[:6]
    return {"normalized_en": text + " (restated)", "low_level_keywords": words, "high_level_keywords": words[:2],
            "objects": objs[:5]}


def make_ver(mode="accept"):
    """Scripted Verifier: accepts the first candidate (2 models with code), or rejects round 1."""
    def fake(prompt):
        COUNT["ver"] += 1
        names = re.findall(r"\n### Candidate \d+: (\S+) ", prompt)
        block = prompt.split(f": {names[0]} (")[1].split("\n### Candidate")[0]
        code = [x for x, dl in re.findall(r"^- ([a-z0-9_.]+) \(.*?\) \[[A-Z_, ]+(, delegated: no code here)?\]",
                                          block, flags=re.M) if not dl]
        quote = names[0]
        base = {"candidates": [{"module": m, "fit": "yes" if i == 0 else "no", "reason": f"fake {i}"}
                               for i, m in enumerate(names)],
                "chosen_module": names[0], "verdict": "partial", "models": code[:2],
                "missing": ["approval by a manager"], "evidence": [quote, "an invented quote"], "reason": "fake"}
        if mode == "reject_first" and "round1" not in COUNT:
            COUNT["round1"] = True
            return {**base, "chosen_module": None, "verdict": "not_supported", "models": []}
        return base
    return fake


tmp = Path(tempfile.mkdtemp())
COMMON = ["--source", "snapshot", "--cache-dir", str(tmp / "cache"), "--oca-dir", str(tmp / "no_oca"),
          "--llm-cache", str(tmp / "official_map"), "--verifier-cache", str(tmp / "official_ver"),
          "--demo-cache", str(tmp / "demo_llm")]


def run_demo(argv, ver_mode="accept"):
    COUNT.update(map=0, ver=0)
    COUNT.pop("round1", None)
    args = demo.build_parser().parse_args(argv + COMMON)
    rep = demo.run_demo(args, fake_embed, fake_llm, make_ver(ver_mode), PG)
    return rep, {k: COUNT[k] for k in ("map", "ver")}


def files(d):
    return sorted(str(p.relative_to(d)) for p in Path(d).rglob("*.json")) if Path(d).exists() else []


print("\n=== Demo offline checks ===")
# 1. same answer as the pipeline (official caches written by run_pipeline.py)
pargs = rp.build_parser().parse_args(["--lang", "en", "--split", "dev", "--source", "snapshot",
                                      "--params", "data/mapping/params_final_en.json",
                                      "--out-root", str(tmp / "runs"), "--llm-cache", str(tmp / "official_map"),
                                      "--verifier-cache", str(tmp / "official_ver"), "--cache-dir", str(tmp / "cache"),
                                      "--oca-dir", str(tmp / "no_oca"), "--no-new-collect"])
_, _, items, items_pv, _ = rp.run(pargs, fake_embed, fake_llm, make_ver(), PG)
p08 = next(i for i in items if i["id"] == "Q08")
official = files(tmp / "official_map") + files(tmp / "official_ver")
rep, n = run_demo(["--id", "Q08", "--dry-llm"])
r = rep["recommendation"]
check(n == {"map": 0, "ver": 0} and sum(rep["run"]["new_llm_calls"].values()) == 0,
      f"--id Q08 --dry-llm with the official caches: 0 LLM calls ({rep['run']['mapping']})")
check(r["module"] == p08["modules"][0] and [x["model"] for x in r["models"]] == p08["models"],
      f"same module and models as run_pipeline.py (P): {r['module']} {p08['models']}")
check(r["verdict"] == "partial" and r["status"] == "verified" and r["missing"] == ["approval by a manager"]
      and rep["run"]["verifier"] == ["official run cache"],
      "Verifier verdict, missing parts and source (official cache) are reported")
check(r["evidence_found"] == "1/2", f"evidence check of the quotes: {r['evidence_found']}")
check(rep["other_candidates"][0]["module"] == p08["module_ranking"][1] and rep["other_candidates"][0]["verifier"],
      "other candidates follow the ranking and carry the Verifier fit")

# 2. --dry-llm without cache = P-noLLM
rep2, n2 = run_demo(["--id", "Q08", "--dry-llm", "--run", "2"])
ma_args = ma.build_parser().parse_args(["--lang", "en", "--split", "dev", "--source", "snapshot", "--dry-llm",
                                        "--params", "data/mapping/params_final_en.json", "--out-root",
                                        str(tmp / "runs"), "--cache-dir", str(tmp / "cache")])
_, minfo = ma.run(ma_args, fake_embed, None, PG)
m08 = next(i for i in minfo["items"] if i["id"] == "Q08")
check(n2 == {"map": 0, "ver": 0} and rep2["run"]["mapping"].startswith("no LLM"),
      f"--dry-llm, no cache: 0 calls, {rep2['run']['mapping'][:60]}")
check(rep2["recommendation"]["module"] == m08["modules"][0]
      and [x["model"] for x in rep2["recommendation"]["models"]] == m08["models"],
      "answer equals P-noLLM (mapping_agent.py --dry-llm)")
check(rep2["recommendation"]["verdict"] is None and rep2["recommendation"]["status"].startswith("not verified")
      and "Mapping ran without LLM" in rep2["run"]["verifier"][0], "Verifier skipped and flagged 'not verified'")

# 3. new text: demo cache only, rerun from cache
text = "Users must not validate a delivery order when the stock location has no stock quant left"
rep3, n3 = run_demo([text])
check(n3 == {"map": 1, "ver": 1} and rep3["requirement"]["id"].startswith("ADHOC_")
      and rep3["requirement"]["lang"] == "en", f"new English text: 1 Mapping + 1 Verifier call, id "
      f"{rep3['requirement']['id']}")
check(files(tmp / "official_map") + files(tmp / "official_ver") == official and len(files(tmp / "demo_llm")) == 2,
      "new calls saved in the demo cache only; official caches unchanged")
rep3b, n3b = run_demo([text, "--dry-llm"])
check(n3b == {"map": 0, "ver": 0} and rep3b["recommendation"]["module"] == rep3["recommendation"]["module"]
      and rep3b["recommendation"]["verdict"] == rep3["recommendation"]["verdict"]
      and rep3b["run"]["verifier"] == ["demo cache"], "rerun with --dry-llm: 0 calls, same answer from the demo cache")

# 4. language and evaluation ids
rep4, _ = run_demo(["Không cho xuất kho khi tồn kho không đủ", "--dry-llm"])
check(rep4["requirement"]["lang"] == "vi" and rep4["requirement"]["id"] == "Q01",
      "Vietnamese text of Q01: language vi detected, id Q01 reused")
rep4b, _ = run_demo(["Chặn xuất kho khi thiếu hàng ở kho tổng", "--dry-llm"])
check(rep4b["requirement"]["lang"] == "vi" and rep4b["requirement"]["id"].startswith("ADHOC_"),
      "new Vietnamese text: language vi, new id")
rep4c, _ = run_demo(["--id", "Q08", "--lang", "vi", "--dry-llm"])
check(rep4c["requirement"]["text"].startswith("Không") or demo.detect_lang(rep4c["requirement"]["text"]) == "vi",
      "--id Q08 --lang vi takes requirement_vi")

# 5. budget and Verifier rejection
rep5, n5 = run_demo(["Lot numbers are created for each incoming product move line", "--max-llm-calls", "1"])
check(n5 == {"map": 1, "ver": 0} and rep5["recommendation"]["status"].startswith("not verified")
      and "budget" in rep5["run"]["verifier"][0], "--max-llm-calls 1: Mapping only, Verifier skipped (budget)")
rep6, n6 = run_demo(["Reset a cancelled transfer back to draft so it can be processed again"], "reject_first")
rounds = rep6["verifier_rounds"]
check(n6["ver"] == 2 and len(rounds) == 2 and not set(rounds[0]["candidates"]) & set(rounds[1]["candidates"])
      and rep6["rejected"] == sorted(rounds[0]["candidates"]) and rep6["recommendation"]["module"] in rounds[1]["candidates"],
      f"rejection: round 2 judges the next candidates, answer {rep6['recommendation']['module']}")
rep7, n7 = run_demo(["Reset a cancelled picking back to draft state", "--max-llm-calls", "2"], "reject_first")
check(n7 == {"map": 1, "ver": 1} and len(rep7["verifier_rounds"]) == 2 and rep7["verifier_rounds"][1]["verdict"] is None
      and rep7["recommendation"]["status"].startswith("not verified: no round accepted"),
      "rejection with budget 2: round 2 skipped, Mapping answer kept, 'not verified'")

# 6. rendering and files
txt, md = demo.render(rep), demo.render(rep, md=True)
for name, s in (("text", txt.lower()), ("markdown", md.lower())):
    check(all(k.lower() in s for k in (r["module"], r["models"][0]["model"], "EXTENDS" if "EXTENDS" in str(r["models"]) else
                               "DEFINES", "Direct dependencies", "path in the knowledge graph", "Verdict",
                               "approval by a manager")), f"{name} output has module, models, relations, "
                                                          f"dependencies, path, verdict, missing parts")
check("| Model | Relation of the module |" in md and "\u2014" not in md + txt and "\u2013" not in md + txt,
      "Markdown table; no em or en dash")
fields = [f for x in r["models"] for f in x["fields_added"]]
check(any(fields) and all(x["relations"] for x in r["models"]), f"fields added are listed ({len(fields)})")
out = tmp / "out" / "demo_Q08.json"
rep8 = demo.main(["--id", "Q08", "--dry-llm", "--format", "json", "--out", str(out), "--show-gold"] + COMMON,
                 embed_fn=fake_embed, llm_fn=fake_llm, ver_llm_fn=make_ver(), pg=PG)
saved = json.loads(out.read_text(encoding="utf-8"))
check(saved["recommendation"]["module"] == rep8["recommendation"]["module"] and saved["expected"]["main_module"]
      == "stock_no_negative", "--format json --out writes the report; --show-gold adds the expected answer")
md_out = tmp / "out" / "demo_Q08.md"
demo.main(["--id", "Q08", "--dry-llm", "--format", "md", "--out", str(md_out)] + COMMON,
          embed_fn=fake_embed, llm_fn=fake_llm, ver_llm_fn=make_ver(), pg=PG)
check(md_out.read_text(encoding="utf-8").startswith("# Agentic GraphRAG for Odoo"), "--format md --out writes Markdown")
src = Path("src/demo/demo.py").read_text(encoding="utf-8")
check(not re.search(r"(password|api_key|thesis12345)\s*=\s*['\"]", src, re.I) and "\u2014" not in src,
      "no secret default and no em dash in src/demo/demo.py")

print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
