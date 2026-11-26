"""
Freezes the parameters of pipeline P for the official runs on the test split (0 LLM calls, no database).

Input:
  - the Mapping parameters chosen on the dev split by src/mapping/tune.py (data/mapping/params_dev_<lang>.json)
  - a dev run of the Mapping Agent with those parameters (P-V predictions, e.g.
    data/runs/pv_en_20261126_dev_run1/predictions.json), which holds the Mapping score of every module

tau (low-confidence threshold: below it the Mapping Agent asks the Collector for evidence before the Verifier):
  tau = lowest top Mapping score among the dev requirements whose top module is correct (main or accepted),
        rounded down to a multiple of --step (default 0.05).
  So no correct dev answer is flagged as low confidence, and the dev results of P do not change.
  With 4 dev requirements this is a rule of thumb, not a tuned value: it is reported as such.

All other parameters are copied unchanged from the tune.py file.

Usage:
  python src/mapping/finalize_params.py --params data/mapping/params_dev_en.json \
      --dev-run data/runs/pv_en_20261126_dev_run1/predictions.json
  # writes data/mapping/params_final_en.json, used with --params by run_pipeline.py and mapping_agent.py
"""
import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import load_requirements  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--params", required=True, help="output of tune.py")
    ap.add_argument("--dev-run", required=True, help="P-V (Mapping) predictions on the dev split with --params")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tuned = json.loads(Path(args.params).read_text(encoding="utf-8"))
    run = json.loads(Path(args.dev_run).read_text(encoding="utf-8"))
    lang = tuned["lang"]
    if run.get("lang") != lang:
        sys.exit(f"Language mismatch: params {lang}, dev run {run.get('lang')}")
    same = {k: (run["params"].get(k), tuned["params"][k]) for k in tuned["params"] if k != "tau"}
    diff = {k: v for k, v in same.items() if v[0] != v[1]}
    if diff:
        sys.exit(f"The dev run was not made with these parameters: {diff}")

    reqs = {r["id"]: r for r in load_requirements(args.requirements, "dev")}
    rows = []
    for it in run["items"]:
        if it["id"] not in reqs:
            continue
        r = reqs[it["id"]]
        top = it["top_modules"][0]
        second = it["top_modules"][1]["score"] if len(it["top_modules"]) > 1 else None
        ok = top["module"] in {r["main_module"], *r["accepted_modules"]}
        rows.append({"id": it["id"], "top_module": top["module"], "top_score": top["score"],
                     "second_score": second, "correct": ok})
    missing = sorted(set(reqs) - {x["id"] for x in rows})
    if missing:
        sys.exit(f"Dev requirements missing from the dev run: {missing}")
    correct = [x["top_score"] for x in rows if x["correct"]]
    if not correct:
        sys.exit("No correct dev answer: tau cannot be set by this rule")
    tau = math.floor(min(correct) / args.step + 1e-9) * args.step
    tau = round(tau, 4)

    params = {**tuned["params"], "tau": tau}
    out = Path(args.out or f"data/mapping/params_final_{lang}.json")
    info = {"date": f"{date.today():%Y%m%d}", "lang": lang, "status": "frozen for the official runs on the test split",
            "params": params,
            "sources": {"mapping_params": args.params, "dev_run": args.dev_run},
            "tau_rule": f"lowest top Mapping score among correct dev answers, rounded down to a multiple of {args.step}",
            "tau_dev_scores": rows,
            "notes": ["Mapping parameters unchanged from tune.py (12 combinations tie on dev; the one closest to the "
                      "defaults is kept). Changing them would need new Verifier calls on dev without a dev signal.",
                      "With this tau no dev requirement is flagged low_confidence, so the dev results of P and P-V "
                      "(R-34) stay valid.",
                      "Collector evidence in the loop comes from data/collected_v1 (0 LLM calls); run_pipeline.py "
                      "is run with --no-new-collect."]}
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Dev top Mapping scores ({lang}):")
    for x in rows:
        print(f"  {x['id']}: {x['top_module']} {x['top_score']:.4f} (second {x['second_score']:.4f}), "
              f"{'correct' if x['correct'] else 'WRONG'}")
    print(f"tau = {tau} (lowest correct {min(correct):.4f}, rounded down to {args.step}); "
          f"low_confidence on dev: {sum(x['top_score'] < tau for x in rows)}/{len(rows)}")
    print(f"Saved: {out}")
    return info


if __name__ == "__main__":
    main()
