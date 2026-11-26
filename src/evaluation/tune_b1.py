"""
Chooses m (number of models in the answer set) of baseline B1 on the dev split, from B1 runs already made
(0 LLM calls, no database, no embedding): each B1 predictions file keeps the top 10 models of every requirement.

k (modules in the answer set) stays 1 for every configuration (B1, P, P-V): the comparison is made on the same
answer size, and MRR / Hit@k do not depend on k.

Objective: mean model F1 (lenient) on dev, averaged over the languages given. Ties: the m closest to the m of
pipeline P (--p-m, default 4), then the smaller m.

Usage:
  python src/evaluation/tune_b1.py data/runs/b1_en_20261124/predictions.json data/runs/b1_vi_20261124/predictions.json
  # writes data/baselines/params_b1_dev.json; then rerun B1 with --m <chosen m> and score it
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import load_requirements, score_one  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Choose m of B1 on the dev split")
    ap.add_argument("predictions", nargs="+", help="B1 predictions.json files (one per language)")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--grid", default="1,2,3,4,5,6")
    ap.add_argument("--p-m", type=int, default=4, help="m of pipeline P (tie-break)")
    ap.add_argument("--out", default="data/baselines/params_b1_dev.json")
    args = ap.parse_args(argv)

    dev = load_requirements(args.requirements, "dev")
    grid = [int(x) for x in args.grid.split(",")]
    runs = []
    for f in args.predictions:
        run = json.loads(Path(f).read_text(encoding="utf-8"))
        items = {it["id"]: it for it in run["items"]}
        short = [r["id"] for r in dev if len(items[r["id"]].get("top_models", [])) < max(grid)]
        if short:
            sys.exit(f"{f}: fewer than {max(grid)} saved models for {short}")
        runs.append((f, run["lang"], items))

    table = []
    for m in grid:
        row = {"m": m}
        for f, lang, items in runs:
            s = [score_one(r, {"module_ranking": items[r["id"]]["module_ranking"],
                               "modules": items[r["id"]]["module_ranking"][:1],
                               "models": [x["model"] for x in items[r["id"]]["top_models"][:m]]}) for r in dev]
            for key in ("mrr", "model_f1", "model_f1_strict", "model_r"):
                row[f"{lang}_{key}"] = round(sum(x[key] for x in s) / len(s), 4)
        row["objective"] = round(sum(row[f"{lang}_model_f1"] for _, lang, _ in runs) / len(runs), 4)
        table.append(row)
    best = max(table, key=lambda r: (r["objective"], -abs(r["m"] - args.p_m), -r["m"]))
    ties = [r["m"] for r in table if r["objective"] == best["objective"]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {"date": f"{date.today():%Y%m%d}", "split": "dev", "ids": [r["id"] for r in dev],
            "inputs": [f for f, _, _ in runs], "objective": "mean model F1 (lenient) over the languages",
            "tie_break": f"closest to the m of P ({args.p_m}), then smaller m",
            "grid": table, "ties_at_best": ties, "params": {"k": 1, "m": best["m"]}}
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    langs = [lang for _, lang, _ in runs]
    print(f"B1 on dev {info['ids']} ({', '.join(langs)}; 0 LLM calls)")
    for r in table:
        print(f"  m={r['m']}: " + "; ".join(f"{lang} model F1 {r[f'{lang}_model_f1']:.3f} "
                                            f"(strict {r[f'{lang}_model_f1_strict']:.3f})" for lang in langs)
              + f"; objective {r['objective']:.3f}")
    print(f"Chosen: k = 1, m = {best['m']} (ties at best: {ties})")
    print(f"Saved: {out}")
    return info


if __name__ == "__main__":
    main()
