"""
Paired comparison of configurations on the same requirements: Wilcoxon signed-rank test (proposal, Section 6).

For every configuration the per-requirement scores are computed with the shared scoring rules
(src/evaluation/metrics.py, D21). A configuration run several times (P, run 1 to 3) is given as several
predictions files: its score on a requirement is the mean over the runs. Each configuration is then compared
with the baseline on the same requirements.

Test: two-sided Wilcoxon signed-rank test on the per-requirement differences (other - baseline).
  - zero differences are dropped (Wilcoxon's method); tied absolute differences get their mean rank
  - p-value: exact, by enumerating all 2^n sign assignments of the ranks (n = non-zero differences <= 20;
    exact also with ties), otherwise the normal approximation with tie correction
  - effect size: matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-), from -1 to 1
  - wins / ties / losses of the configuration against the baseline, per requirement
Primary metric (declared before the test runs): model F1, lenient (the proposal tests F1 per requirement;
the module level is close to the ceiling). The other metrics are secondary and are reported without
correction for multiple comparisons; Holm-adjusted p-values are printed next to them for reference.

Usage:
  python src/evaluation/compare.py --split test --baseline B1 \\
      --config B1=data/runs/b1_en_20261126/predictions.json \\
      --config P-noLLM=data/runs/pnollm_en_20261126_all_run1/predictions.json \\
      --out data/runs/compare/en_test_b1_pnollm.json
  # several runs of one configuration: --config P=run1/predictions.json,run2/predictions.json,run3/predictions.json
"""
import argparse
import itertools
import json
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, load_requirements  # noqa: E402

PRIMARY = "model_f1"
DEFAULT_METRICS = ["model_f1", "model_f1_strict", "mrr", "hit@1", "module_f1"]
EXACT_MAX_N = 20


def ranks_abs(values):
    """Mean ranks (1-based) of |values|, ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon(diffs, eps=1e-12):
    """Two-sided Wilcoxon signed-rank test. Returns a dict (n, W+, W-, p, method, r_rb)."""
    d = [x for x in diffs if abs(x) > eps]
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "p_value": 1.0, "method": "all differences are 0",
                "rank_biserial": 0.0}
    rk = ranks_abs(d)
    w_plus = sum(r for r, x in zip(rk, d) if x > 0)
    w_minus = sum(r for r, x in zip(rk, d) if x < 0)
    total = w_plus + w_minus
    mean = total / 2
    if n <= EXACT_MAX_N:
        obs = abs(w_plus - mean)
        hits = 0
        for signs in itertools.product((0, 1), repeat=n):          # 2^n sign assignments of the ranks
            wp = sum(r for r, s in zip(rk, signs) if s)
            if abs(wp - mean) >= obs - 1e-9:
                hits += 1
        p, method = hits / 2 ** n, f"exact (2^{n} sign assignments)"
    else:
        ties = {}
        for r in rk:
            ties[r] = ties.get(r, 0) + 1
        var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(var) if var > 0 else 0.0
        p, method = math.erfc(max(z, 0) / math.sqrt(2)), "normal approximation (tie and continuity corrected)"
    return {"n_nonzero": n, "w_plus": w_plus, "w_minus": w_minus, "p_value": min(1.0, p), "method": method,
            "rank_biserial": (w_plus - w_minus) / total if total else 0.0}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


def per_requirement(paths, reqs):
    """Mean per-requirement scores over the runs of one configuration."""
    runs, info = [], []
    for p in paths:
        run = json.loads(Path(p).read_text(encoding="utf-8"))
        res = evaluate(reqs, {it["id"]: it for it in run["items"]})
        missing = [r["id"] for r in reqs if r["id"] not in {it["id"] for it in run["items"]}]
        runs.append({q["id"]: q for q in res["per_question"]})
        info.append({"file": str(p), "config": run.get("config"), "lang": run.get("lang"),
                     "run": run.get("run"), "missing": missing, "stopped": run.get("stopped")})
    return runs, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paired Wilcoxon comparison of configurations")
    ap.add_argument("--config", action="append", required=True,
                    help="NAME=predictions.json[,predictions.json...] (several files = several runs, averaged)")
    ap.add_argument("--baseline", required=True, help="name of the baseline configuration")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--allow-missing", action="store_true",
                    help="compare even if a run has no prediction for some requirements (scored as empty)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    reqs = load_requirements(args.requirements, args.split)
    ids = [r["id"] for r in reqs]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    configs = {}
    for c in args.config:
        name, _, files = c.partition("=")
        configs[name] = [f for f in files.split(",") if f]
    if args.baseline not in configs:
        sys.exit(f"--baseline {args.baseline} is not one of {list(configs)}")

    scores, sources = {}, {}
    for name, paths in configs.items():
        runs, info = per_requirement(paths, reqs)
        bad = [i for i in info if i["missing"]]
        if bad and not args.allow_missing:
            sys.exit(f"{name}: requirements without prediction {[(i['file'], i['missing']) for i in bad]} "
                     f"(unfinished run?); use --allow-missing to score them as empty answers")
        scores[name] = {q: {m: sum(r[q][m] for r in runs) / len(runs) for m in metrics} for q in ids}
        sources[name] = info

    base = scores[args.baseline]
    results = []
    for name in configs:
        if name == args.baseline:
            continue
        rows = []
        for m in metrics:
            diffs = [scores[name][q][m] - base[q][m] for q in ids]
            t = wilcoxon(diffs)
            rows.append({"metric": m, "primary": m == PRIMARY,
                         "mean_baseline": sum(base[q][m] for q in ids) / len(ids),
                         "mean_config": sum(scores[name][q][m] for q in ids) / len(ids),
                         "mean_diff": sum(diffs) / len(diffs),
                         "wins": sum(x > 1e-12 for x in diffs), "ties": sum(abs(x) <= 1e-12 for x in diffs),
                         "losses": sum(x < -1e-12 for x in diffs), **t,
                         "per_requirement_diff": {q: round(x, 4) for q, x in zip(ids, diffs)}})
        for row, adj in zip(rows, holm([r["p_value"] for r in rows])):
            row["p_holm"] = adj
        results.append({"config": name, "baseline": args.baseline, "tests": rows})

    print(f"Wilcoxon signed-rank, two-sided, split {args.split}, n = {len(ids)} requirements; "
          f"primary metric: {PRIMARY}")
    for res in results:
        print(f"\n{res['config']} vs {res['baseline']}:")
        print(f"  {'metric':<16} {'base':>6} {'config':>6} {'diff':>7}  W/T/L     n   W+     W-     p        "
              f"p(Holm)  r_rb")
        for r in res["tests"]:
            print(f"  {r['metric'] + (' *' if r['primary'] else ''):<16} {r['mean_baseline']:6.3f} "
                  f"{r['mean_config']:6.3f} {r['mean_diff']:+7.3f}  {r['wins']:>2}/{r['ties']:>2}/{r['losses']:<2}  "
                  f"{r['n_nonzero']:>2}  {r['w_plus']:6.1f} {r['w_minus']:6.1f} {r['p_value']:.4f}   "
                  f"{r['p_holm']:.4f}   {r['rank_biserial']:+.2f}")
    print("\n  * primary metric. W/T/L = requirements where the configuration is better / equal / worse. "
          "n = non-zero differences.\n  p: exact when n <= 20. r_rb: rank-biserial correlation (effect size).")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"date": f"{date.today():%Y%m%d}", "split": args.split, "requirements": ids,
                                   "requirements_file": args.requirements, "primary_metric": PRIMARY,
                                   "test": "Wilcoxon signed-rank, two-sided, zero differences dropped",
                                   "sources": sources, "results": results}, indent=2), encoding="utf-8")
        print(f"Saved: {out}")
    return results


if __name__ == "__main__":
    main()
