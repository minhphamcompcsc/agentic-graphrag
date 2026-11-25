"""
Chooses the decision parameters of the Mapping Agent on the dev split (4 requirements) by grid search.
Uses only cached LLM analyses (run the agent on the dev split first), so it costs 0 LLM calls.

Objective = mean MRR (module level) + mean model F1 (lenient), both from src/evaluation/metrics.py.
Ties are broken by the smallest distance to the default parameters (to avoid over-fitting 4 requirements).
alpha is fixed to 1: with k_modules = 1 only the ratios beta/alpha and gamma/alpha change the ranking.

Usage:
  python src/mapping/tune.py --lang en            # writes data/mapping/params_dev_en.json
  python src/mapping/mapping_agent.py --lang en --split test --params data/mapping/params_dev_en.json
"""
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_agent import Params, build_parser, decide, features, load_resources  # noqa: E402
from analyze import Analyzer  # noqa: E402
from metrics import load_requirements, score_one  # noqa: E402

GRID = {
    "beta": [0.0, 0.25, 0.5, 1.0, 2.0],
    "gamma": [0.0, 0.25, 0.5, 1.0],
    "cov": ["ochiai", "ratio"],
    "m_models": [2, 3, 4],
    "lam_entry": [0.0, 0.2, 0.5],
    "lam_function": [0.0, 0.1, 0.3],
    "delegated_penalty": [0.0, 0.3],
}


def distance_to_default(p: Params, d: Params):
    out = 0.0
    for k, vals in GRID.items():
        a, b = getattr(p, k), getattr(d, k)
        out += (a != b) if isinstance(a, str) else abs(vals.index(a) - vals.index(b)) / (len(vals) - 1)
    return out


def main(argv=None, embed_fn=None, llm_fn=None, pg=None):
    ap = build_parser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.split = "dev"
    base = Params()
    res, _ = load_resources(args.source, args.snapshot, args.semantic_dir, args.collected_dir, args.cache_dir,
                            embed_fn, pg)
    an = Analyzer("none" if args.dry_llm else args.llm, args.llm_model, args.run, args.llm_cache, 0, llm_fn)
    reqs = load_requirements(args.requirements, "dev")
    feats = []
    for r in reqs:
        text = r[f"requirement_{args.lang}"]
        analysis, _ = an.analyze(r["id"], args.lang, text)       # max_calls = 0: cache only
        feats.append((r, features(res, text, analysis, base)))
    results = []
    for combo in itertools.product(*GRID.values()):
        p = replace(base, **dict(zip(GRID, combo)))
        scores = [score_one(r, decide(f, p, res.g)) for r, f in feats]
        mrr = sum(s["mrr"] for s in scores) / len(scores)
        mf1 = sum(s["model_f1"] for s in scores) / len(scores)
        results.append((round(mrr + mf1, 6), -distance_to_default(p, base), p, mrr, mf1))
    results.sort(key=lambda t: (-t[0], -t[1]))
    best = results[0]
    default_row = next(r for r in results if r[2] == base)
    out = Path(args.out or f"data/mapping/params_dev_{args.lang}{'_nollm' if args.dry_llm else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    info = {"date": f"{date.today():%Y%m%d}", "lang": args.lang, "split": "dev", "ids": [r["id"] for r in reqs],
            "objective": "mean MRR + mean model F1 (lenient)", "grid": GRID, "combinations": len(results),
            "best": {"objective": best[0], "mrr": round(best[3], 4), "model_f1": round(best[4], 4)},
            "default": {"objective": default_row[0], "mrr": round(default_row[3], 4),
                        "model_f1": round(default_row[4], 4)},
            "ties_at_best": sum(1 for r in results if r[0] == best[0]),
            "params": asdict(best[2])}
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Tuned on dev {info['ids']} ({len(results)} combinations, 0 LLM calls)")
    print(f"  default: objective {default_row[0]:.3f} (MRR {default_row[3]:.3f}, model F1 {default_row[4]:.3f})")
    print(f"  best   : objective {best[0]:.3f} (MRR {best[3]:.3f}, model F1 {best[4]:.3f}); "
          f"{info['ties_at_best']} combinations tie, the one closest to the defaults is kept")
    print(f"  params : {json.dumps({k: info['params'][k] for k in GRID})}")
    print(f"Saved: {out}")
    return info


if __name__ == "__main__":
    main()
