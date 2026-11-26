"""
Full pipeline P (Mapping -> Verifier, with the agentic loop) and its ablation P-V (Mapping only), in one run.

For each requirement:
  1. Mapping Agent: analysis (1 LLM call, cached in data/mapping/llm), features, decision (same code and cache
     as src/mapping/mapping_agent.py). This answer is written as configuration P-V (no LLM call of its own).
  2. If the best Mapping score is below tau (low confidence), the Mapping Agent asks the Collector for more
     evidence on its top candidates.
  3. Verifier round 1: deterministic KG checks + 1 LLM judgement of the top C candidates.
     - verdict supported / partial  -> accepted: the chosen module and models become the answer.
     - verdict insufficient_evidence -> the Collector adds GitHub issues for the candidates that do not
                                        have them yet, then round 2 on the same candidates.
                                        If the Collector has nothing new: accepted as "weak" when a module
                                        was chosen, otherwise handled as not_supported.
     - verdict not_supported         -> back to the Mapping Agent: the rejected candidates are excluded and
                                        the next C modules of the ranking are judged in round 2.
  4. At most max_rounds (2) rounds. If no round accepts, the Mapping answer is kept and flagged
     verified = false (the BA sees a warning). An empty answer is never returned (D21 scores it 0).

Collector in the loop: GitHub issues are read from data/collected_v1/<module>.json (Collector v1 output,
cache: 0 calls). For an OCA module that was never collected, collect_v1() is called (GitHub + 1 LLM call),
if the OCA repositories are present (--oca-dir). Odoo core modules have no Collector source.
The issues are only shown to the Verifier; they are not written to the KG (D18).

Outputs (same format as the Mapping Agent, scored by src/evaluation/evaluate.py):
  data/runs/pfull_<lang>_<date>_<split>_run<r>/predictions.json   configuration P   (with Verifier)
  data/runs/pv_<lang>_<date>_<split>_run<r>/predictions.json      configuration P-V (Mapping only)
Each P item keeps the Verifier rounds: candidates, verdict, chosen module, models, invalid outputs,
evidence check, Collector calls, LLM cache use.

Usage (in ~/agentic-graphrag, venv activated, Docker services and Ollama running):
  python src/verifier/run_pipeline.py --lang en --split dev --params data/mapping/params_dev_en.json --dry-verifier
        # 0 LLM calls: Mapping from cache, Verifier prompts saved to data/verifier/prompts/ for inspection
  python src/verifier/run_pipeline.py --lang en --split dev --params data/mapping/params_dev_en.json
        # 4 to 8 Verifier calls on the dev split (Mapping analyses are cached)
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/verifier", "src/mapping", "src/baselines", "src/extractor", "src/evaluation", "src/collector"):
    sys.path.insert(0, str(ROOT / sub))
import mapping_agent as ma  # noqa: E402
from analyze import PROMPT_VERSION as MAP_PROMPT, Analyzer, DailyQuotaExceeded as MapQuota  # noqa: E402
from verifier import (ACCEPT, PROMPT_VERSION as VER_PROMPT, VERIFIER_VERSION, DailyQuotaExceeded as VerQuota,  # noqa: E402
                      TransientLLMError, Verifier, build_prompt, dossier, kg_checks, validate)

PIPELINE_VERSION = "pipe-v0.1"


# ============ Collector in the loop ============
class CollectorLink:
    """Evidence for a module from the Collector: cached Collector v1 output, or a new collect_v1() call."""

    def __init__(self, g, collected_dir, oca_dir, budget, allow_new=True):
        self.g, self.dir, self.oca = g, Path(collected_dir), Path(oca_dir)
        self.budget, self.allow_new = budget, allow_new
        self.state = {"llm_calls": 0, "max_llm_calls": 0}
        self.done = {}

    def issues(self, module):
        if module in self.done:
            return self.done[module]
        path = self.dir / f"{module}.json"
        status = "no_source"
        if path.exists():
            status = "cache"
        elif (self.g.modules.get(module, {}).get("source") == "oca" and self.allow_new
              and self.oca.exists() and self.budget() > 0):
            from collect_modules_v1 import collect_v1
            self.state["max_llm_calls"] = self.state["llm_calls"] + 1
            r = collect_v1(module, self.oca, self.dir, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           self.state)
            status = "new" if r.get("status") == "ok" else f"collect_{r.get('status')}"
        items = []
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            for s in rec.get("github", {}).get("selected", []):
                txt = " ".join((s.get("body_clean") or "").split())
                items.append({"number": s["number"], "title": s["title"], "type": s["type"], "state": s["state"],
                              "year": (s.get("created_at") or "")[:4],
                              "text": txt[:500] + (" ..." if len(txt) > 500 else "")})
        self.done[module] = {"status": status, "issues": items}
        return self.done[module]


# ============ the agentic loop for one requirement ============
def verify_one(r, text, analysis, feat, dec0, res, p, ver, col, args):
    g = res.g
    rounds, rejected, evidence = [], set(), {}
    dec = dec0
    collector_calls = []

    def collect(mods, why):
        for m in mods:
            if m not in evidence:
                e = col.issues(m)
                evidence[m] = e
                collector_calls.append({"module": m, "why": why, "status": e["status"], "issues": len(e["issues"])})

    if dec0["low_confidence"]:
        collect([m for m in dec0["module_ranking"] if m not in rejected][:args.candidates], "low_confidence")
    accepted, weak = None, False
    for rnd in range(1, args.max_rounds + 1):
        cands = [m for m in dec["module_ranking"] if m not in rejected][:args.candidates]
        doss, valid = [], {}
        for i, m in enumerate(cands, 1):
            iss = evidence.get(m, {}).get("issues") if m in evidence else None
            text_m, models_m = dossier(res, feat, p, ma.rank_models, m, i, iss)
            doss.append(text_m)
            valid[m] = models_m
        prompt = build_prompt(text, analysis.get("normalized_en", ""), doss,
                              any(evidence.get(m, {}).get("issues") for m in cands))
        raw, cached = ver.judge(r["id"], args.lang, rnd, prompt)
        v = validate(raw, cands, valid, prompt)
        rounds.append({"round": rnd, "candidates": cands, "llm_cached": cached, "prompt_chars": len(prompt),
                       "with_issues": [m for m in cands if evidence.get(m, {}).get("issues")], **v})
        if raw is None:                          # no LLM (--dry-verifier): pass the Mapping answer through
            break
        if v["verdict"] in ACCEPT and v["chosen_module"]:
            accepted = v
            break
        if rnd == args.max_rounds:
            break
        new_source = [m for m in cands if m not in evidence]
        if v["verdict"] == "insufficient_evidence" and new_source:
            collect(new_source, "insufficient_evidence")
            if any(evidence[m]["issues"] for m in new_source):
                continue                          # round 2 on the same candidates with the new evidence
        if v["verdict"] == "insufficient_evidence" and v["chosen_module"]:
            accepted, weak = v, True              # nothing new to collect: keep the choice, flagged weak
            break
        rejected |= set(cands)                    # back to the Mapping Agent without these modules
        dec = ma.decide(feat, p, g, exclude=rejected)
        if dec["low_confidence"]:
            collect([m for m in dec["module_ranking"] if m not in rejected][:args.candidates], "low_confidence")

    if accepted:
        m = accepted["chosen_module"]
        models = accepted["models"] or [x["model"] for x in ma.rank_models(feat, m, p)[:p.m_models]]
        ranking = [m] + [x for x in dec0["module_ranking"] if x != m]
        verified = True
    else:
        m, models, ranking, verified = dec0["modules"][0], dec0["models"], dec0["module_ranking"], False
    rel = {x["model"]: x for x in ma.rank_models(feat, m, p)}
    answer = [{"module": m, "models": [{"model": x, "relations": rel.get(x, {}).get("relations", []),
                                         "code": rel.get(x, {}).get("code")} for x in models],
               "kg_checks": kg_checks(g, m, models), "path": feat["modules"][m]["path"],
               "functions": g.functions_of(m),
               "verdict": accepted["verdict"] if accepted else None,
               "missing": accepted["missing"] if accepted else [],
               "evidence": accepted["evidence"] if accepted else [],
               "reason": accepted["reason"] if accepted else ""}]
    return {"module_ranking": ranking, "modules": [m], "models": models, "answer": answer,
            "verification": {"verified": verified, "weak": weak,
                             "changed_module": m != dec0["modules"][0],
                             "changed_models": models != dec0["models"],
                             "rounds": rounds, "rejected": sorted(rejected), "collector": collector_calls,
                             "llm_calls": sum(1 for x in rounds if not x["llm_cached"])},
            "mapping": {"modules": dec0["modules"], "models": dec0["models"],
                        "low_confidence": dec0["low_confidence"]}}


# ============ run ============
def run(args, embed_fn=None, llm_fn=None, ver_llm_fn=None, pg=None):
    from metrics import load_requirements
    t0 = time.time()
    p = ma.Params(**json.loads(Path(args.params).read_text())["params"]) if args.params else ma.Params()
    if args.tau is not None:
        p.tau = args.tau
    res, embed = ma.load_resources(args.source, args.snapshot, args.semantic_dir, args.collected_dir,
                                   args.cache_dir, embed_fn, pg)
    print(f"Pipeline {PIPELINE_VERSION} = Mapping {ma.MAPPING_VERSION} + Verifier {VERIFIER_VERSION} "
          f"({args.lang}, split {args.split}, run {args.run}, tau {p.tau}): graph from {res.g.source} {res.g.stats()}")
    an = Analyzer("none" if args.dry_llm else args.llm, args.llm_model, args.run, args.llm_cache,
                  args.max_llm_calls, llm_fn)
    budget = lambda: args.max_llm_calls - an.calls - ver.calls - col.state["llm_calls"]  # noqa: E731
    vb = "none" if (args.dry_verifier or args.dry_llm) else args.llm
    prompt_dir = Path(args.prompt_dir) / f"{args.lang}_{args.split}_run{args.run}" if vb == "none" else None
    ver = Verifier(vb, args.llm_model, args.run, args.verifier_cache, budget, ver_llm_fn, prompt_dir)
    col = CollectorLink(res.g, args.collected_dir, args.oca_dir, budget, allow_new=not args.no_new_collect)
    reqs = load_requirements(args.requirements, args.split)
    items_p, items_pv, stopped = [], [], None
    for r in reqs:
        text = r[f"requirement_{args.lang}"]
        try:
            analysis, cached = an.analyze(r["id"], args.lang, text)
            feat = ma.features(res, text, analysis, p)
            dec0 = ma.decide(feat, p, res.g)
            base = {"id": r["id"], "query": text, "analysis": analysis, "llm_cached": cached,
                    "entry_functions": feat["entry_functions"], "entry_models": feat["entry_models"]}
            out = verify_one(r, text, analysis, feat, dec0, res, p, ver, col, args)
        except (MapQuota, VerQuota, TransientLLMError) as e:
            stopped = f"{type(e).__name__}: {e}"
            break
        except Exception as e:  # noqa: BLE001  (any other error: keep what is done, report, stop)
            stopped = f"{type(e).__name__}: {repr(e)[:300]}"
            break
        items_pv.append({**base, **dec0})
        items_p.append({**base, **out})
        v = out["verification"]
        flag = ("no LLM" if vb == "none" else "unverified" if not v["verified"] else "weak" if v["weak"] else
                v["rounds"][-1]["verdict"] if v["rounds"] and v["rounds"][-1]["verdict"] else "no LLM")
        change = " | CHANGED from Mapping" if v["changed_module"] else ""
        print(f"  {r['id']}: {out['modules'][0]} | models: {', '.join(out['models'])} | {flag}, "
              f"{len(v['rounds'])} round(s), {v['llm_calls']} new call(s){change}")
    today = f"{date.today():%Y%m%d}"
    common = {"lang": args.lang, "date": today, "run": args.run, "mapping_version": ma.MAPPING_VERSION,
              "mapping_prompt": MAP_PROMPT, "llm": an.model, "graph": res.g.stats(), "graph_source": res.g.source,
              "params": asdict(p), "params_file": args.params, "requirements_file": args.requirements,
              "split": args.split, "new_embeddings": embed.new, "stopped": stopped}
    tag = "nollm" if args.dry_llm else ""
    out_pv = Path(args.out_root) / f"pv{tag}_{args.lang}_{today}_{args.split}_run{args.run}"
    out_p = Path(args.out_root) / f"pfull{tag}_{args.lang}_{today}_{args.split}_run{args.run}"
    for d, cfg, items, extra in (
            (out_pv, "P-V", items_pv, {"llm_calls_mapping": an.calls}),
            (out_p, "P", items_p, {"pipeline_version": PIPELINE_VERSION, "verifier_version": VERIFIER_VERSION,
                                   "verifier_prompt": VER_PROMPT, "verifier_backend": vb,
                                   "candidates": args.candidates, "max_rounds": args.max_rounds,
                                   "llm_calls_mapping": an.calls, "llm_calls_verifier": ver.calls,
                                   "llm_calls_collector": col.state["llm_calls"]})):
        d.mkdir(parents=True, exist_ok=True)
        info = {"config": cfg, **common, **extra, "seconds": round(time.time() - t0, 1), "items": items}
        (d / "predictions.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(items_p)
    changed = sum(it["verification"]["changed_module"] for it in items_p)
    unver = sum(not it["verification"]["verified"] for it in items_p)
    print(f"New LLM calls: Mapping {an.calls}, Verifier {ver.calls}, Collector {col.state['llm_calls']}; "
          f"{n} requirements; module changed by the Verifier {changed}; unverified {unver}; "
          f"{round(time.time() - t0, 1)} s")
    if prompt_dir:
        print(f"Verifier prompts (no LLM): {prompt_dir}/")
    if stopped:
        print(f"STOPPED early at requirement {len(items_p) + 1}/{len(reqs)}: {stopped}\n  Rerun the same command "
              f"(after the quota resets if the daily quota is exhausted); answers already received are cached and are "
              f"not called again. The files below hold only the finished requirements.")
    print(f"Saved: {out_p}/predictions.json (P), {out_pv}/predictions.json (P-V)")
    print(f"Next: python src/evaluation/evaluate.py {out_p}/predictions.json --split {args.split}")
    return out_p, out_pv, items_p, items_pv, {"mapping": an.calls, "verifier": ver.calls,
                                               "collector": col.state["llm_calls"], "stopped": stopped}


def build_parser():
    ap = ma.build_parser()
    ap.description = "Full pipeline P (Mapping + Verifier + agentic loop) and P-V"
    ap.add_argument("--candidates", type=int, default=3, help="candidates judged per Verifier round")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--tau", type=float, default=None, help="low-confidence threshold (default: from --params)")
    ap.add_argument("--dry-verifier", action="store_true", help="no Verifier LLM call; prompts are saved")
    ap.add_argument("--verifier-cache", default="data/verifier/llm")
    ap.add_argument("--prompt-dir", default="data/verifier/prompts")
    ap.add_argument("--oca-dir", default="oca")
    ap.add_argument("--no-new-collect", action="store_true", help="Collector: cached outputs only")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
