"""
Offline test of the Verifier Agent and the agentic loop (no Neo4j, no Ollama, no Gemini). Same throwaway
PostgreSQL as tests/test_mapping_offline.py:
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/load_mock_catalog.py data/snapshots/schema_20261124.json
    PG_HOST=/tmp PG_PORT=5433 PG_DB=kg_mock python tests/test_verifier_offline.py

Fakes: hashed bag-of-words embedding and the fake Mapping LLM of test_mapping_offline.py; a scripted Verifier LLM
whose behaviour is chosen per test (accept, reject, insufficient evidence, invalid output).
The scores printed here only check the plumbing; they are NOT results for the report.
Checks:
  1. dossiers list the footprint of each candidate (relations, delegated models, added fields) and the
     documentation; the prompt is of a reasonable size
  2. validation drops a module that is not a candidate and models outside the footprint
  3. not_supported -> back to the Mapping Agent: round 2 judges the next candidates, never more than 2 rounds
  4. insufficient_evidence -> the Collector adds cached GitHub issues (0 LLM calls), round 2 on the same
     candidates with the issues in the prompt; Odoo core modules have no Collector source
  5. low confidence (tau) -> the Collector is called before round 1
  6. P-V equals the Mapping Agent output; P and P-V are scored by evaluate.py
  7. the budget stops the run cleanly; a rerun uses the caches (0 calls) and gives the same answers
  8. --dry-verifier saves the prompts and makes 0 calls
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
for sub in ("src/verifier", "src/mapping", "src/baselines", "src/extractor", "src/evaluation",
            "src/collector"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import mapping_agent as ma  # noqa: E402
import run_pipeline as rp  # noqa: E402
from metrics import evaluate, load_requirements  # noqa: E402
from verifier import validate  # noqa: E402

from graph_view import GraphView  # noqa: E402

PG = dict(host=os.getenv("PG_HOST", "/tmp"), port=int(os.getenv("PG_PORT", "5433")),
          dbname=os.getenv("PG_DB", "kg_mock"), user=os.getenv("PG_USER", "odoo"), password="")
assert PG["dbname"] != "kg_demo", "use the throwaway database"
SNAP = "data/snapshots/schema_20261124.json"
FAILS = []
G0 = GraphView.from_snapshot(SNAP, "data/semantic")
LABELS = sorted(((x, G0.models[x]["label"].lower()) for x in G0.models), key=lambda t: -len(t[1]))


def fake_embed(texts):
    """Same fake as tests/test_mapping_offline.py: hashed bag of words."""
    out = []
    for t in texts:
        v = np.zeros(512)
        for w in re.findall(r"[a-z]+", t.lower()):
            w = w[:-1] if w.endswith("s") and len(w) > 3 else w
            v[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 512] += 1
        out.append(v if v.any() else np.ones(512))
    return out


def fake_llm(prompt):
    """Same fake Mapping LLM as tests/test_mapping_offline.py: model labels written in the requirement."""
    text = prompt.split("<<<")[1].split(">>>")[0].strip()
    low = text.lower()
    objs = [{"mention": lab, "odoo_model": x} for x, lab in LABELS
            if len(lab) > 4 and re.search(rf"\b{re.escape(lab)}\b", low)]
    words = [w for w in re.findall(r"[a-z]{5,}", low)][:6]
    return {"normalized_en": text, "low_level_keywords": words, "high_level_keywords": words[:2], "objects": objs[:5]}


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def parse(prompt):
    """Candidates and their model lists from a Verifier prompt."""
    blocks = re.split(r"\n### Candidate \d+: ", prompt.split("\nAnswer:")[0])[1:]
    out = {}
    for b in blocks:
        name = b.split(" ", 1)[0]
        models = re.findall(r"^- ([a-z0-9_.]+) \(.*?\) \[([A-Z_, ]+)(, delegated: no code here)?\]", b, flags=re.M)
        out[name] = [(x, rel, bool(dl)) for x, rel, dl in models]
    return out


def make_fake(mode):
    log = []

    def fake(prompt):
        c = parse(prompt)
        names = list(c)
        log.append({"candidates": names, "issues": "GitHub issues selected" in prompt})
        req = prompt.split("<<<")[1].split(">>>")[0].lower()
        words = set(re.findall(r"[a-z]{4,}", req))

        def overlap(m):
            block = prompt.split(f": {m} (")[1].split("\n### Candidate")[0].lower()
            return len(words & set(re.findall(r"[a-z]{4,}", block)))
        best = max(names, key=overlap)
        code = [x for x, rel, dl in c[best] if not dl]
        base = {"candidates": [{"module": m, "fit": "yes" if m == best else "no", "reason": "fake"} for m in names],
                "chosen_module": best, "verdict": "supported", "models": code[:2], "missing": [],
                "evidence": [best], "reason": "fake"}
        rnd = len(log)
        if mode == "reject_first" and rnd % 2 == 1:
            return {**base, "chosen_module": None, "verdict": "not_supported", "models": []}
        if mode == "insufficient_first" and rnd % 2 == 1:
            return {**base, "verdict": "insufficient_evidence"}
        if mode == "always_reject":
            return {**base, "chosen_module": None, "verdict": "not_supported", "models": []}
        if mode == "invalid":
            return {**base, "chosen_module": "not_a_candidate", "models": ["res.partner"]}
        return base
    return fake, log


tmp = Path(tempfile.mkdtemp())


def run(mode, extra=(), split="dev"):
    fake, log = make_fake(mode)
    args = rp.build_parser().parse_args(["--lang", "en", "--split", split, "--source", "snapshot",
                                         "--out-root", str(tmp / f"runs_{mode}"), "--llm-cache", str(tmp / "llm"),
                                         "--cache-dir", str(tmp / "cache"), "--max-llm-calls", "60",
                                         "--verifier-cache", str(tmp / f"ver_{mode}"),
                                         "--prompt-dir", str(tmp / "prompts"), "--oca-dir", str(tmp / "no_oca"),
                                         *extra])
    out = rp.run(args, fake_embed, fake_llm, fake, PG)
    return out, log, args


print("\n=== Verifier offline checks ===")
# 1. dossier and prompt
(out_p, out_pv, items, items_pv, calls), log, args = run("accept")
it = next(i for i in items if i["id"] == "Q08")
r1 = it["verification"]["rounds"][0]
prompt = (tmp / "ver_accept").glob("**/en_Q08_round1.json").__next__().read_text()
pr = json.loads(prompt)["prompt"]
check(len(r1["candidates"]) == 3 and all(f": {m} (" in pr for m in r1["candidates"]),
      f"round 1 judges the top 3 Mapping candidates: {r1['candidates']}")
check("stock_no_negative" in pr and "stock.quant (" in pr and "allow_negative_stock" in pr,
      "dossier of stock_no_negative lists stock.quant and the field allow_negative_stock")
check("product.product (" in pr and "delegated: no code here" in pr, "delegated models are marked (R-28)")
sizes = [x["verification"]["rounds"][0]["prompt_chars"] for x in items]
check(max(sizes) < 30000, f"prompt size {min(sizes)} to {max(sizes)} characters")
check(calls["verifier"] == 4 and all(len(x["verification"]["rounds"]) == 1 for x in items),
      f"accept: 1 round per requirement, {calls['verifier']} Verifier calls on 4 dev requirements")
ok_fp = all(set(x["models"]) <= G0.footprint(x["modules"][0]) for x in items)
check(ok_fp, "every Verifier model is in the footprint of the chosen module")

# 2. validation
v = validate({"chosen_module": "x_module", "verdict": "supported", "models": ["a.b"], "evidence": ["zzz"]},
             ["m1", "m2"], {"m1": ["stock.quant"]}, "### Candidate 1: m1 ...\nAnswer:")
check(v["chosen_module"] is None and v["invalid_module"] == "x_module" and v["verdict"] == "insufficient_evidence",
      "a module that is not a candidate is dropped and the verdict is not accepting")
v = validate({"chosen_module": "m1", "verdict": "Supported", "models": ["stock.quant", "res.partner"],
              "evidence": ["Candidate 1", "made up text"]}, ["m1"], {"m1": ["stock.quant"]},
             "### Candidate 1: m1 ...\nAnswer:")
check(v["models"] == ["stock.quant"] and v["invalid_models"] == ["res.partner"] and v["verdict"] == "supported",
      "models outside the footprint are dropped; verdict is normalised")
check(v["evidence_found"] == 1, "evidence check: 1 of 2 quotes found in the dossier")
(_, _, items_inv, _, _), _, _ = run("invalid")
check(all(x["verification"]["rounds"][0]["invalid_module"] == "not_a_candidate" for x in items_inv),
      "invalid LLM output is recorded and never reaches the answer")
check(all(x["modules"] == x["mapping"]["modules"] for x in items_inv if not x["verification"]["verified"]),
      "without an accepted round the Mapping answer is kept (verified = false)")

# 3. not_supported -> back to Mapping
(_, _, items_r, _, calls_r), log_r, _ = run("reject_first")
x = items_r[0]["verification"]
check(len(x["rounds"]) == 2 and not set(x["rounds"][0]["candidates"]) & set(x["rounds"][1]["candidates"]),
      f"reject: round 2 judges the next candidates {x['rounds'][1]['candidates']}")
check(x["rejected"] == sorted(x["rounds"][0]["candidates"]) and items_r[0]["modules"][0] in x["rounds"][1]["candidates"],
      "the answer comes from round 2; rejected modules are recorded")
(_, _, items_a, _, calls_a), _, _ = run("always_reject")
check(all(len(i["verification"]["rounds"]) == 2 and not i["verification"]["verified"] for i in items_a)
      and calls_a["verifier"] == 8, "always reject: stops after 2 rounds, 8 calls, Mapping answer kept and flagged")

# 4. insufficient evidence -> Collector
(_, _, items_i, _, calls_i), log_i, _ = run("insufficient_first")
q = {i["id"]: i for i in items_i}
x = q["Q08"]["verification"]
oca_cands = [m for m in x["rounds"][0]["candidates"] if G0.modules[m]["source"] == "oca"]
check(any(c["status"] == "cache" and c["issues"] > 0 for c in x["collector"]),
      f"insufficient evidence: Collector read cached issues {[(c['module'], c['status'], c['issues']) for c in x['collector']]}")
check(len(x["rounds"]) == 2 and x["rounds"][1]["candidates"] == x["rounds"][0]["candidates"]
      and x["rounds"][1]["with_issues"], "round 2 judges the same candidates with GitHub issues in the prompt")
check(calls_i["collector"] == 0, "Collector: 0 LLM calls (cached Collector v1 output)")
col = rp.CollectorLink(G0, "data/collected_v1", tmp / "no_oca", lambda: 10)
check(col.issues("stock")["status"] == "no_source" and not col.issues("stock")["issues"]
      and col.issues("stock_request")["status"] == "cache", "Odoo core module stock: no Collector source; "
      "stock_request: cached Collector v1 output")

# 5. low confidence -> Collector before round 1
(_, _, items_t, _, _), log_t, _ = run("accept", ["--tau", "100"])
check(all(i["verification"]["collector"] and all(c["why"] == "low_confidence" for c in i["verification"]["collector"])
          for i in items_t) and log_t[0]["issues"] in (True, False),
      "tau above every score: the Collector is called before round 1 for every requirement")
check(any(r["with_issues"] for i in items_t for r in i["verification"]["rounds"]),
      "issues of OCA candidates appear in round 1 when confidence is low")

# 6. P-V = Mapping; evaluate
margs = ma.build_parser().parse_args(["--lang", "en", "--split", "dev", "--source", "snapshot",
                                      "--out-root", str(tmp / "runs_map"), "--llm-cache", str(tmp / "llm"),
                                      "--cache-dir", str(tmp / "cache")])
_, minfo = ma.run(margs, fake_embed, fake_llm, PG)
same = all(a["modules"] == b["modules"] and a["models"] == b["models"] and a["module_ranking"] == b["module_ranking"]
           for a, b in zip(minfo["items"], items_pv))
check(same, "P-V predictions = Mapping Agent predictions (modules, models, ranking)")
reqs = load_requirements("data/eval/requirements_v0.csv", "dev")
ev_p = evaluate(reqs, {i["id"]: i for i in items})["overall"]
ev_pv = evaluate(reqs, {i["id"]: i for i in items_pv})["overall"]
check(ev_p["n"] == 4 and ev_pv["n"] == 4, f"evaluate (fake): P hit@1 {ev_p['hit@1']} model F1 {ev_p['model_f1']}; "
      f"P-V hit@1 {ev_pv['hit@1']} model F1 {ev_pv['model_f1']}")
check(json.loads((out_p / "predictions.json").read_text())["config"] == "P"
      and json.loads((out_pv / "predictions.json").read_text())["config"] == "P-V", "output files carry P and P-V")

# 7. budget and cache
fake, _ = make_fake("accept")
bargs = rp.build_parser().parse_args(["--lang", "en", "--split", "test", "--source", "snapshot",
                                      "--out-root", str(tmp / "runs_b"), "--llm-cache", str(tmp / "llm_b"),
                                      "--cache-dir", str(tmp / "cache"), "--max-llm-calls", "5",
                                      "--verifier-cache", str(tmp / "ver_b"), "--oca-dir", str(tmp / "no_oca")])
*_, cb = rp.run(bargs, fake_embed, fake_llm, fake, PG)
check(cb["stopped"] and cb["mapping"] + cb["verifier"] == 5, f"budget 5: stopped after {cb['mapping']} Mapping + "
      f"{cb['verifier']} Verifier calls")
for cap in (3, 4):                            # the budget is shared: Mapping never adds a call beyond it
    fake_c, _ = make_fake("accept")
    cargs = rp.build_parser().parse_args(["--lang", "en", "--split", "test", "--source", "snapshot",
                                          "--out-root", str(tmp / f"runs_c{cap}"), "--llm-cache", str(tmp / f"llm_c{cap}"),
                                          "--cache-dir", str(tmp / "cache"), "--max-llm-calls", str(cap),
                                          "--verifier-cache", str(tmp / f"ver_c{cap}"), "--oca-dir", str(tmp / "no_oca")])
    *_, cc = rp.run(cargs, fake_embed, fake_llm, fake_c, PG)
    check(cc["stopped"] and cc["mapping"] + cc["verifier"] == cap,
          f"budget {cap}: exactly {cap} calls ({cc['mapping']} Mapping + {cc['verifier']} Verifier)")
bargs.max_llm_calls = 60
*_, it_b1, _, cb1 = rp.run(bargs, fake_embed, fake_llm, fake, PG)
*_, it_b2, _, cb2 = rp.run(bargs, fake_embed, fake_llm, fake, PG)
check(len(it_b1) == 16 and cb2["mapping"] + cb2["verifier"] == 0, f"resume: 16 requirements; rerun "
      f"{cb2['mapping'] + cb2['verifier']} calls")
check([i["modules"] + i["models"] for i in it_b1] == [i["modules"] + i["models"] for i in it_b2],
      "rerun from cache gives the same answers")

# 7b. transient error (504 deadline): retried once, then the run stops cleanly and resumes from the cache
state = {"n": 0}
fake_ok, _ = make_fake("accept")


def flaky(prompt):
    state["n"] += 1
    if state["n"] in (3, 4):                  # 3rd requirement: first try and the retry both fail
        raise RuntimeError("DeadlineExceeded('504 Deadline expired before operation could complete.')")
    return fake_ok(prompt)


fargs = rp.build_parser().parse_args(["--lang", "en", "--split", "dev", "--source", "snapshot",
                                      "--out-root", str(tmp / "runs_f"), "--llm-cache", str(tmp / "llm"),
                                      "--cache-dir", str(tmp / "cache"), "--verifier-cache", str(tmp / "ver_f"),
                                      "--oca-dir", str(tmp / "no_oca")])
out_f, _, it_f, _, cf = rp.run(fargs, fake_embed, fake_llm, flaky, PG)
saved = json.loads((out_f / "predictions.json").read_text())
check(cf["stopped"] and "TransientLLMError" in cf["stopped"] and len(it_f) == 2 and len(saved["items"]) == 2,
      "504 twice: run stops cleanly, the 2 finished requirements are saved")
_, _, it_f2, _, cf2 = rp.run(fargs, fake_embed, fake_llm, flaky, PG)
check(not cf2["stopped"] and len(it_f2) == 4 and cf2["verifier"] == 2,
      "rerun: 2 requirements from the cache, 2 new calls, 4 requirements saved")
state["n"] = 10
fargs.verifier_cache = str(tmp / "ver_f2")
state2 = {"n": 0}


def flaky_once(prompt):
    state2["n"] += 1
    if state2["n"] == 1:
        raise RuntimeError("ServiceUnavailable('503 The model is overloaded.')")
    return fake_ok(prompt)


_, _, it_f3, _, cf3 = rp.run(fargs, fake_embed, fake_llm, flaky_once, PG)
check(not cf3["stopped"] and len(it_f3) == 4, "503 once: retried and the run completes")

# 8. dry verifier
(_, _, items_d, _, cd), _, dargs = run("accept", ["--dry-verifier"])
pdir = tmp / "prompts" / "en_dev_run1"
check(cd["verifier"] == 0 and len(list(pdir.glob("*.txt"))) == 4 and all(i["modules"] == i["mapping"]["modules"]
      for i in items_d), "--dry-verifier: 0 calls, 4 prompts saved, Mapping answer passed through")

print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
