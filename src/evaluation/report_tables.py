"""
Tables and figure of the Internship 1 report, generated from the run folders in one command.

Reads data/runs and finds, for each configuration, the latest COMPLETE run (every requirement of the split has a
prediction and the run did not stop early):
  B1       b1_<lang>_<date>/                         (latest date: the m = 4 rerun, D26)
  P-noLLM  pnollm_<lang>_<date>_<all|test>_run1/
  P        pfull_en_<date>_test_run<r>/              (r = 1 to 3; the latest complete folder of each run number)
  P-V      pv_en_<date>_test_run<r>/                 (same folder date and run as P)
  P, P-V on dev: pfull_en_<date>_dev_run1/, pv_en_<date>_dev_run1/ (RQ3 dev columns)
Missing or unfinished P runs are skipped (the tables and the figure are written without them).
Every score is recomputed from predictions.json with the shared scoring rules (src/evaluation/metrics.py, D21);
a configuration with several runs is the mean over the runs. Wilcoxon tests are run with
src/evaluation/compare.py (its JSON outputs are saved in data/runs/compare/report_*.json).

Outputs (--out-dir, default reports/tt1):
  tables/tab_rq1_{vi,en}.tex        RQ1 results on the test split (B1, P-noLLM, P; English and Vietnamese queries)
  tables/tab_wilcoxon_{vi,en}.tex   Wilcoxon signed-rank against B1 (model F1 is the primary metric, D28)
  tables/tab_rq3_{vi,en}.tex        P against P-V (dev and test), Verifier and Collector activity
  tables/tab_p_runs_{vi,en}.tex     P run by run (only when P has runs)
  tables/all_tables_{vi,en}.tex     the tables above in one file
  figures/fig_rq1_bar.pdf           grouped bars, Hit@1 and model F1, English queries (vector PDF, English text)
  figures/fig_rq1_bar_vi.pdf        same for the Vietnamese queries (B1, P-noLLM)
  summary.json                      every number used above, with the source folders
LaTeX follows the report conventions: table[H], small, tabularx, booktabs; Vietnamese tables use the decimal comma.

Usage (0 LLM calls, no database):
  python src/evaluation/report_tables.py
  python src/evaluation/report_tables.py --p data/runs/pfull_en_20261128_test_run1,data/runs/pfull_en_20261130_test_run2
"""
import argparse
import contextlib
import io
import json
import re
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare  # noqa: E402
from metrics import evaluate, load_requirements  # noqa: E402

RUN_RE = re.compile(r"^(?P<cfg>[a-z0-9]+)_(?P<lang>en|vi)_(?P<date>\d{8})(?:_(?P<split>all|dev|test)_run(?P<run>\d+))?$")
COLORS = {"B1": "#257de0", "P-noLLM": "#f47230", "P": "#00b894"}   # categorical slots 1 to 3, validated (dataviz)
WILCOXON_METRICS = ["model_f1", "model_f1_strict", "mrr", "hit@1"]


# ============ finding runs ============
def scan(root):
    out = []
    for d in sorted(Path(root).iterdir()) if Path(root).exists() else []:
        m = RUN_RE.match(d.name)
        if m and (d / "predictions.json").exists():
            out.append({**m.groupdict(), "dir": d})
    return out


def load(d):
    return json.loads((Path(d) / "predictions.json").read_text(encoding="utf-8"))


def complete(d, ids):
    run = load(d)
    have = {it["id"] for it in run["items"]}
    return not run.get("stopped") and set(ids) <= have


def latest(runs, cfg, lang, splits, ids, run=None):
    cands = [r for r in runs if r["cfg"] == cfg and r["lang"] == lang and (r["split"] in splits)
             and (run is None or r["run"] == str(run))]
    for r in sorted(cands, key=lambda r: r["date"], reverse=True):
        if complete(r["dir"], ids):
            return r["dir"]
    return None


def find_p_runs(runs, split, ids):
    """Latest complete pfull folder for each run number, with its pv folder."""
    out = []
    for r in sorted({x["run"] for x in runs if x["cfg"] == "pfull" and x["lang"] == "en" and x["split"] == split},
                    key=int):
        p = latest(runs, "pfull", "en", (split,), ids, r)
        if p is None:
            continue
        pv = Path(str(p).replace("/pfull_", "/pv_").replace("\\pfull_", "\\pv_"))
        out.append((p, pv if (pv / "predictions.json").exists() and complete(pv, ids) else None))
    return out


# ============ scores ============
def scores(dirs, reqs):
    """Mean over runs of the overall metrics, plus the per-run values."""
    per_run = [evaluate(reqs, {it["id"]: it for it in load(d)["items"]})["overall"] for d in dirs]
    keys = [k for k in per_run[0] if k != "n"]
    mean = {k: sum(r[k] for r in per_run) / len(per_run) for k in keys}
    return {"mean": mean, "runs": per_run, "n": per_run[0]["n"], "files": [str(d) for d in dirs]}


def verifier_stats(dirs, ids):
    """Verifier and Collector activity of P (mean over runs of counts per split)."""
    rows = []
    for d in dirs:
        items = [it for it in load(d)["items"] if it["id"] in ids]
        v = [it["verification"] for it in items]
        rows.append({
            "changed_module": sum(x["changed_module"] for x in v),
            "changed_models": sum(x["changed_models"] for x in v),
            "unverified": sum(not x["verified"] for x in v),
            "weak": sum(x["weak"] for x in v),
            "two_rounds": sum(len(x["rounds"]) >= 2 for x in v),
            "calls_per_req": sum(sum(1 for r in x["rounds"] if r.get("verdict")) for x in v) / len(v),
            "collector_reqs": sum(bool(x["collector"]) for x in v),
            "low_confidence": sum(bool(it.get("mapping", {}).get("low_confidence")) for it in items),
            "issues_in_prompt": sum(any(r.get("with_issues") for r in x["rounds"]) for x in v),
            "issues_in_prompt_ids": sorted(it["id"] for it in items
                                           if any(r.get("with_issues") for r in it["verification"]["rounds"]))})
    keys = [k for k in rows[0] if not k.endswith("_ids")]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    mean["issues_in_prompt_ids"] = sorted({q for r in rows for q in r["issues_in_prompt_ids"]})
    return {"mean": mean, "runs": rows}


def run_compare(name, configs, baseline, split, out):
    argv = ["--split", split, "--baseline", baseline, "--out", str(out), "--metrics", ",".join(WILCOXON_METRICS)]
    for cfg, files in configs.items():
        argv += ["--config", f"{cfg}=" + ",".join(str(Path(f) / "predictions.json") for f in files)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = compare.main(argv)
    return {r["config"]: {t["metric"]: t for t in r["tests"]} for r in res}, buf.getvalue()


# ============ number formats ============
def num(x, lang, nd=3):
    """0.625 -> 0,625 (vi); trailing zeros dropped but one decimal kept: 1.0, 0.5."""
    if x is None:
        return "--"
    s = f"{round(x + 0.0, nd):.{nd}f}".rstrip("0")
    s = s + "0" if s.endswith(".") else s
    return s.replace(".", ",") if lang == "vi" else s


def pval(p, lang):
    if p < 0.0001:                      # exact p can be far below the 4-digit display (e.g. 2/2^16)
        s = "$<$0.0001"
    else:
        s = f"{p:.4f}" if p < 0.01 else num(p, "en", 4)
    return s.replace(".", ",") if lang == "vi" else s


def eff(r, lang):
    s = f"{r:.2f}"
    return s.replace(".", ",") if lang == "vi" else s


def count(x, lang):
    return str(int(x)) if abs(x - round(x)) < 1e-9 else num(x, lang, 2)


# ============ LaTeX ============
T = {
    "vi": {"metric": "Chỉ số", "en": "Tiếng Anh", "vi": "Tiếng Việt", "lenient": "nới", "strict": "chặt",
           "rq1_cap": "Kết quả trên tập kiểm thử ({n} câu)", "p_mean": ". P là trung bình {k} lần chạy",
           "cmp": "So sánh", "cfg": "Cấu hình", "wtl": "Thắng/ hoà/ thua", "primary": "chính",
           "wil_cap": "Wilcoxon signed-rank 2 phía, so với B1 trên {n} câu kiểm thử. n: số câu có hiệu khác 0; "
                      "p chính xác; r: rank-biserial",
           "dev": "Tập phát triển ({n} câu)", "test": "Tập kiểm thử ({n} câu)",
           "rq3_cap": "P (Mapping và Verifier) so với P-V (chỉ Mapping){runs}. Số liệu tập phát triển chỉ để minh "
                      "hoạ, không dùng để kết luận",
           "runs_note": "; tập kiểm thử: trung bình {k} lần chạy",
           "changed": "Số câu Verifier đổi module", "changed_models": "Số câu Verifier đổi model",
           "calls": "Số lời gọi Verifier mỗi câu", "two": "Số câu có vòng 2", "unver": "Số câu chưa xác minh",
           "lowconf": "Số câu điểm Mapping dưới $\\tau$", "issues": "Số câu Verifier nhận issue GitHub",
           "wil_pv": "Wilcoxon model F1 nới, P so với P-V (p)",
           "run": "Lần chạy", "mean": "Trung bình", "sd": "Độ lệch chuẩn",
           "runs_cap": "P trên tập kiểm thử theo từng lần chạy (cache LLM riêng mỗi lần)",
           "rows": [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("mrr", "MRR"),
                    ("module_f1", "Module F1 (nới)"), ("module_f1_strict", "Module F1 (chặt)"),
                    ("model_p", "Model Precision (nới)"), ("model_r", "Model Recall"),
                    ("model_f1", "Model F1 (nới)"), ("model_f1_strict", "Model F1 (chặt)")],
           "wm": {"model_f1": "Model F1 (chính)", "model_f1_strict": "Model F1 chặt", "mrr": "MRR",
                  "hit@1": "Hit@1"},
           "langname": {"en": "tiếng Anh", "vi": "tiếng Việt"}},
    "en": {"metric": "Metric", "en": "English queries", "vi": "Vietnamese queries", "lenient": "lenient",
           "strict": "strict", "rq1_cap": "Results on the test split ({n} requirements)",
           "p_mean": ". P is the mean of {k} runs", "cmp": "Comparison", "cfg": "Config.",
           "wtl": "Wins/ ties/ losses", "primary": "primary",
           "wil_cap": "Two-sided Wilcoxon signed-rank test against B1 on {n} test requirements. n: non-zero "
                      "differences; exact p; r: rank-biserial correlation",
           "dev": "Dev split ({n})", "test": "Test split ({n})",
           "rq3_cap": "P (Mapping and Verifier) against P-V (Mapping only){runs}. Dev numbers are illustrative only",
           "runs_note": "; test split: mean of {k} runs",
           "changed": "Requirements where the Verifier changed the module",
           "changed_models": "Requirements where the Verifier changed the models",
           "calls": "Verifier calls per requirement", "two": "Requirements with a second round",
           "unver": "Unverified requirements", "lowconf": "Requirements with Mapping score below $\\tau$",
           "issues": "Requirements with GitHub issues given to the Verifier",
           "wil_pv": "Wilcoxon on lenient model F1, P vs P-V (p)",
           "run": "Run", "mean": "Mean", "sd": "Std. dev.",
           "runs_cap": "P on the test split, run by run (separate LLM cache per run)",
           "rows": [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("mrr", "MRR"),
                    ("module_f1", "Module F1 (lenient)"), ("module_f1_strict", "Module F1 (strict)"),
                    ("model_p", "Model precision (lenient)"), ("model_r", "Model recall"),
                    ("model_f1", "Model F1 (lenient)"), ("model_f1_strict", "Model F1 (strict)")],
           "wm": {"model_f1": "Model F1 (primary)", "model_f1_strict": "Model F1 strict", "mrr": "MRR",
                  "hit@1": "Hit@1"},
           "langname": {"en": "English", "vi": "Vietnamese"}},
}


def table(spec, header, body, caption, label):
    return ("\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabularx}{\\textwidth}{@{}" + spec + "@{}}\n"
            "\\toprule\n" + header + "\\midrule\n" + body + "\\bottomrule\n\\end{tabularx}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def tab_rq1(S, lang):
    t = T[lang]
    groups = [("en", [c for c in ("B1", "P-noLLM", "P") if ("en", c) in S]),
              ("vi", [c for c in ("B1", "P-noLLM", "P") if ("vi", c) in S])]
    groups = [(g, cs) for g, cs in groups if cs]
    cols = [(g, c) for g, cs in groups for c in cs]
    head = " & ".join(f"\\multicolumn{{{len(cs)}}}{{c}}{{\\textbf{{{t[g]}}}}}" for g, cs in groups)
    rules, start = [], 2
    for i, (g, cs) in enumerate(groups):
        end = start + len(cs) - 1
        rules.append(f"\\cmidrule({'lr' if i < len(groups) - 1 else 'l'}){{{start}-{end}}}")
        start = end + 1
    header = (f"& {head} \\\\\n" + "".join(rules) + "\n" + f"\\textbf{{{t['metric']}}} & "
              + " & ".join(f"\\textbf{{{c}}}" for _, c in cols) + " \\\\\n")
    body = ""
    for key, name in t["rows"]:
        bold = key == "model_f1"
        vals = []
        for g, c in cols:
            m = S[(g, c)]["mean"]
            v = f"{num(m[key[0]], lang)} / {num(m[key[1]], lang)}" if isinstance(key, tuple) else num(m[key], lang)
            vals.append(f"\\textbf{{{v}}}" if bold else v)
        body += (f"\\textbf{{{name}}}" if bold else name) + " & " + " & ".join(vals) + " \\\\\n"
    n = next(iter(S.values()))["n"]
    k = len(S[("en", "P")]["runs"]) if ("en", "P") in S else 0
    cap = t["rq1_cap"].format(n=n) + (t["p_mean"].format(k=k) if k else "")
    return table("L" + "P{1.6cm}" * len(cols), header, body, cap, "tab:rq1" + ("_en" if lang == "en" else ""))


def tab_wilcoxon(W, S, lang):
    t = T[lang]
    header = (f"\\textbf{{{t['cmp']}}} & \\textbf{{{t['metric']}}} & \\textbf{{B1}} & \\textbf{{{t['cfg']}}} & "
              f"\\textbf{{{t['wtl']}}} & \\textbf{{n}} & \\textbf{{p}} & \\textbf{{r}} \\\\\n")
    blocks = []
    order = [("en", "P-noLLM"), ("vi", "P-noLLM"), ("en", "P")]
    for (g, c), tests in sorted(W.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 9):
        rows = []
        for i, met in enumerate(WILCOXON_METRICS):
            x = tests[met]
            name = f"\\textbf{{{t['wm'][met]}}}" if met == "model_f1" else t["wm"][met]
            p = pval(x["p_value"], lang)
            p = f"\\textbf{{{p}}}" if met == "model_f1" and x["p_value"] < 0.05 else p
            lead = f"{c}, {t['langname'][g]}" if i == 0 else ""
            rows.append(f"{lead} & {name} & {num(x['mean_baseline'], lang)} & {num(x['mean_config'], lang)} & "
                        f"{x['wins']}/{x['ties']}/{x['losses']} & {x['n_nonzero']} & {p} & "
                        f"{eff(x['rank_biserial'], lang)} \\\\\n")
        blocks.append("".join(rows))
    n = next(iter(S.values()))["n"]
    return table("P{2.3cm}LrrP{1.7cm}rP{1.2cm}r", header, "\\addlinespace\n".join(blocks),
                 t["wil_cap"].format(n=n), "tab:wilcoxon" + ("_en" if lang == "en" else ""))


def tab_rq3(R, lang):
    t = T[lang]
    sets = [s for s in ("dev", "test") if s in R]
    if not sets:
        return ""
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{t[s].format(n=R[s]['P']['n'])}}}}}" for s in sets)
    rules = "".join(f"\\cmidrule({'lr' if i < len(sets) - 1 else 'l'}){{{2 + 2 * i}-{3 + 2 * i}}}"
                    for i in range(len(sets)))
    header = (f"& {head} \\\\\n{rules}\n\\textbf{{{t['metric']}}} & "
              + " & ".join("\\textbf{P-V} & \\textbf{P}" for _ in sets) + " \\\\\n")
    lab = dict(t["rows"])
    rows = [("hit@1", "Hit@1"), ("model_p", lab["model_p"]), ("model_r", lab["model_r"]),
            ("model_f1", lab["model_f1"]), ("model_f1_strict", lab["model_f1_strict"])]
    body = ""
    for key, name in rows:
        vals = []
        for s in sets:
            for c in ("P-V", "P"):
                m = R[s][c]["mean"]
                vals.append(f"{num(m[key[0]], lang)} / {num(m[key[1]], lang)}" if isinstance(key, tuple)
                            else num(m[key], lang))
        body += f"{name} & " + " & ".join(vals) + " \\\\\n"
    act = [("changed_module", "changed"), ("changed_models", "changed_models"), ("calls_per_req", "calls"),
           ("two_rounds", "two"), ("unverified", "unver"), ("low_confidence", "lowconf"),
           ("issues_in_prompt", "issues")]
    body += "\\addlinespace\n"
    for key, lab in act:
        vals = []
        for s in sets:
            vals += ["", count(R[s]["verifier"]["mean"][key], lang) if key != "calls_per_req"
                     else num(R[s]["verifier"]["mean"][key], lang, 2)]
        body += f"{t[lab]} & " + " & ".join(vals) + " \\\\\n"
    if "test" in R and R["test"].get("wilcoxon"):
        w = R["test"]["wilcoxon"]
        vals = []
        for s in sets:
            vals += ["", pval(w["p_value"], lang) if s == "test" else ""]
        body += f"{t['wil_pv']} & " + " & ".join(vals) + " \\\\\n"
    k = len(R["test"]["P"]["runs"]) if "test" in R else 0
    cap = t["rq3_cap"].format(runs=t["runs_note"].format(k=k) if k > 1 else "")
    return table("L" + "P{1.7cm}" * (2 * len(sets)), header, body, cap, "tab:rq3" + ("_en" if lang == "en" else ""))


def tab_p_runs(S, PV, lang):
    if ("en", "P") not in S:
        return ""
    t = T[lang]
    keys = [("hit@1", "Hit@1"), ("mrr", "MRR"), ("model_f1", "Model F1"), ("model_f1_strict", "Model F1 " +
            t["strict"])]
    header = (f"\\textbf{{{t['run']}}} & " + " & ".join(f"\\textbf{{{n}}}" for _, n in keys)
              + " & \\textbf{P-V model F1} \\\\\n")
    runs, pv = S[("en", "P")]["runs"], (PV["runs"] if PV else [])
    body = ""
    for i, r in enumerate(runs):
        body += (f"{i + 1} & " + " & ".join(num(r[k], lang) for k, _ in keys)
                 + f" & {num(pv[i]['model_f1'], lang) if i < len(pv) else '--'} \\\\\n")
    body += "\\midrule\n"
    mean = S[("en", "P")]["mean"]
    body += (f"{t['mean']} & " + " & ".join(num(mean[k], lang) for k, _ in keys)
             + f" & {num(PV['mean']['model_f1'], lang) if PV else '--'} \\\\\n")
    if len(runs) > 1:
        body += (f"{t['sd']} & " + " & ".join(num(statistics.stdev([r[k] for r in runs]), lang) for k, _ in keys)
                 + f" & {num(statistics.stdev([r['model_f1'] for r in pv]), lang) if len(pv) > 1 else '--'} \\\\\n")
    return table("P{2.6cm}" + "L" * (len(keys) + 1), header, body, t["runs_cap"], "tab:p_runs"
                 + ("_en" if lang == "en" else ""))


# ============ figure ============
def figure(S, lang_q, path):
    try:
        import matplotlib
    except ImportError:
        print("  NOTE: matplotlib is not installed (pip install 'matplotlib<3.12'); figures skipped")
        return None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9})
    cfgs = [c for c in ("B1", "P-noLLM", "P") if (lang_q, c) in S]
    if not cfgs:
        return None
    metrics = [("hit@1", "Hit@1 (module)"), ("model_f1", "Model F1 (lenient)")]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        sc = S[(lang_q, c)]
        xs = [j + (i - (len(cfgs) - 1) / 2) * w for j in range(len(metrics))]
        ys = [sc["mean"][k] for k, _ in metrics]
        label = c + (f" (mean of {len(sc['runs'])} runs)" if len(sc["runs"]) > 1 else "")
        ax.bar(xs, ys, width=w - 0.03, color=COLORS[c], label=label, edgecolor="white", linewidth=1)
        hi = ys
        if len(sc["runs"]) > 1:                        # range over runs
            lo = [min(r[k] for r in sc["runs"]) for k, _ in metrics]
            hi = [max(r[k] for r in sc["runs"]) for k, _ in metrics]
            ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)], [b - y for y, b in zip(ys, hi)]],
                        fmt="none", ecolor="#191919", elinewidth=1, capsize=3)
        for x, y, top in zip(xs, ys, hi):
            ax.text(x, top + 0.02, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color="#191919")
    ax.set_xticks(range(len(metrics)), [n for _, n in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Score", color="#666666")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#666666")
    n = next(iter(S.values()))["n"]
    ax.set_title(f"Test split, {n} requirements, {'English' if lang_q == 'en' else 'Vietnamese'} queries",
                 fontsize=9, color="#191919", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cfgs))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ============ main ============
def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the report tables and figure from data/runs")
    ap.add_argument("--runs-root", default="data/runs")
    ap.add_argument("--out-dir", default="reports/tt1")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--compare-dir", default="data/runs/compare")
    for o in ("b1-en", "b1-vi", "pnollm-en", "pnollm-vi", "p-dev", "pv-dev"):
        ap.add_argument(f"--{o}", help="run folder (default: latest complete one)")
    ap.add_argument("--p", help="P test run folders, comma separated (default: latest complete folder per run)")
    args = ap.parse_args(argv)

    reqs_test = load_requirements(args.requirements, "test")
    reqs_dev = load_requirements(args.requirements, "dev")
    ids_t, ids_d = [r["id"] for r in reqs_test], [r["id"] for r in reqs_dev]
    runs = scan(args.runs_root)
    src = {
        ("en", "B1"): args.b1_en or latest(runs, "b1", "en", (None,), ids_t),
        ("vi", "B1"): args.b1_vi or latest(runs, "b1", "vi", (None,), ids_t),
        ("en", "P-noLLM"): args.pnollm_en or latest(runs, "pnollm", "en", ("all", "test"), ids_t),
        ("vi", "P-noLLM"): args.pnollm_vi or latest(runs, "pnollm", "vi", ("all", "test"), ids_t),
    }
    if args.p:
        p_runs = [(Path(d), Path(str(d).replace("pfull_", "pv_"))) for d in args.p.split(",") if d]
        p_runs = [(p, pv if (pv / "predictions.json").exists() else None) for p, pv in p_runs]
    else:
        p_runs = find_p_runs(runs, "test", ids_t)
    S, notes = {}, []
    for key, d in src.items():
        if d:
            S[key] = scores([d], reqs_test)
        else:
            notes.append(f"{key[1]} ({key[0]}): no complete run found, skipped")
    PV = None
    if p_runs:
        S[("en", "P")] = scores([p for p, _ in p_runs], reqs_test)
        pvs = [pv for _, pv in p_runs if pv]
        PV = scores(pvs, reqs_test) if len(pvs) == len(p_runs) else None
    else:
        notes.append("P (en, test): no complete run yet, skipped")

    # Wilcoxon against B1 (compare.py)
    cdir = Path(args.compare_dir)
    W, logs = {}, []
    for lang in ("en", "vi"):
        if (lang, "B1") not in S:
            continue
        configs = {"B1": [src[(lang, "B1")]]}
        if (lang, "P-noLLM") in S:
            configs["P-noLLM"] = [src[(lang, "P-noLLM")]]
        if lang == "en" and ("en", "P") in S:
            configs["P"] = [p for p, _ in p_runs]
        if len(configs) > 1:
            res, log = run_compare(lang, configs, "B1", "test", cdir / f"report_{lang}_test_vs_b1.json")
            logs.append(log)
            for c in configs:
                if c != "B1":
                    W[(lang, c)] = res[c]

    # RQ3: P vs P-V on dev and test
    R = {}
    pdev = args.p_dev or latest(runs, "pfull", "en", ("dev",), ids_d)
    pvdev = args.pv_dev or (Path(str(pdev).replace("pfull_", "pv_")) if pdev else None)
    if pdev and pvdev and (Path(pvdev) / "predictions.json").exists():
        R["dev"] = {"P": scores([pdev], reqs_dev), "P-V": scores([pvdev], reqs_dev),
                    "verifier": verifier_stats([pdev], ids_d)}
    if p_runs and PV:
        res, log = run_compare("pv", {"P-V": [pv for _, pv in p_runs], "P": [p for p, _ in p_runs]}, "P-V", "test",
                               cdir / "report_en_test_p_vs_pv.json")
        logs.append(log)
        R["test"] = {"P": S[("en", "P")], "P-V": PV, "verifier": verifier_stats([p for p, _ in p_runs], ids_t),
                     "wilcoxon": res["P"]["model_f1"]}

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    written = []
    for lang in ("vi", "en"):
        parts = {"rq1": tab_rq1(S, lang), "wilcoxon": tab_wilcoxon(W, S, lang) if W else "",
                 "rq3": tab_rq3(R, lang), "p_runs": tab_p_runs(S, PV, lang)}
        for name, tex in parts.items():
            if tex:
                f = out / "tables" / f"tab_{name}_{lang}.tex"
                f.write_text(tex, encoding="utf-8")
                written.append(str(f))
        allf = out / "tables" / f"all_tables_{lang}.tex"
        allf.write_text("% Generated by src/evaluation/report_tables.py on "
                        f"{date.today():%Y-%m-%d}; do not edit by hand\n\n" + "\n".join(t for t in parts.values() if t),
                        encoding="utf-8")
        written.append(str(allf))
    figs = [figure(S, "en", out / "figures" / "fig_rq1_bar.pdf"), figure(S, "vi", out / "figures" / "fig_rq1_bar_vi.pdf")]
    written += [f for f in figs if f]

    def clean(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    summary = {
        "date": f"{date.today():%Y%m%d}", "split": "test", "n_test": len(ids_t), "n_dev": len(ids_d),
        "sources": {f"{c} ({lang})": s["files"] for (lang, c), s in S.items()},
        "scores": {f"{c} ({lang})": {"mean": clean(s["mean"]), "runs": [clean(r) for r in s["runs"]]}
                   for (lang, c), s in S.items()},
        "p_v_test": {"mean": clean(PV["mean"]), "runs": [clean(r) for r in PV["runs"]], "files": PV["files"]}
        if PV else None,
        "wilcoxon_vs_b1": {f"{c} ({lang})": {m: {k: (round(v, 6) if isinstance(v, float) else v)
                                                for k, v in t.items() if k != "per_requirement_diff"}
                                            for m, t in tests.items()} for (lang, c), tests in W.items()},
        "rq3": {s: {"P": clean(R[s]["P"]["mean"]), "P-V": clean(R[s]["P-V"]["mean"]),
                    "verifier": R[s]["verifier"]["mean"], "files": R[s]["P"]["files"] + R[s]["P-V"]["files"],
                    "wilcoxon_model_f1": ({k: v for k, v in R[s]["wilcoxon"].items() if k != "per_requirement_diff"}
                                          if R[s].get("wilcoxon") else None)} for s in R},
        "notes": notes, "outputs": written}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                 default=lambda o: round(o, 4) if isinstance(o, float) else str(o)),
                                      encoding="utf-8")

    print(f"Report tables from {args.runs_root} (test split, {len(ids_t)} requirements)")
    for (lang, c), s in S.items():
        m = s["mean"]
        print(f"  {c:<8} {lang}: Hit@1 {m['hit@1']:.3f}  MRR {m['mrr']:.3f}  model F1 {m['model_f1']:.3f} "
              f"(strict {m['model_f1_strict']:.3f})  runs {len(s['runs'])}  <- {', '.join(Path(f).name for f in s['files'])}")
    for (lang, c), tests in W.items():
        x = tests["model_f1"]
        print(f"  Wilcoxon {c} vs B1 ({lang}), model F1: {x['wins']}/{x['ties']}/{x['losses']}, "
              f"n {x['n_nonzero']}, p {('< 0.0001' if x['p_value'] < 0.0001 else format(x['p_value'], '.4f'))}, r {x['rank_biserial']:+.2f}")
    for s in R:
        print(f"  RQ3 {s}: P-V model F1 {R[s]['P-V']['mean']['model_f1']:.3f} -> P {R[s]['P']['mean']['model_f1']:.3f}; "
              f"Verifier changed module in {count(R[s]['verifier']['mean']['changed_module'], 'en')} requirement(s)")
    for n in notes:
        print(f"  NOTE: {n}")
    print(f"Written: {len(written)} files in {out}/ (+ summary.json)")
    return summary


if __name__ == "__main__":
    main()
