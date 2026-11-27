"""
Offline test of src/evaluation/report_tables.py (no database, no LLM): python tests/test_report_tables.py

Uses the committed B1 and P-noLLM runs and SYNTHETIC P / P-V test runs written to a temporary folder
(numbers printed here are NOT results). Checks:
  1. without P runs: P is skipped; B1 and P-noLLM numbers equal evaluate.py and the Wilcoxon results of compare.py
  2. with 3 synthetic P runs (+ 1 unfinished run that must be ignored): P column = mean of the 3 complete runs,
     P runs table, RQ3 test columns, Wilcoxon P vs B1 and P vs P-V
  3. LaTeX conventions (booktabs, tabularx, decimal comma in Vietnamese tables, no em or en dash);
     figures are vector PDFs
"""
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/evaluation"))
import os  # noqa: E402
os.chdir(ROOT)
import report_tables as rt  # noqa: E402
from metrics import evaluate, load_requirements  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


tmp = Path(tempfile.mkdtemp())
runs = tmp / "runs"
runs.mkdir()
for d in ("b1_en_20261124", "b1_en_20261126", "b1_vi_20261126", "pnollm_en_20261126_all_run1",
          "pnollm_vi_20261126_all_run1", "pfull_en_20261126_dev_run1", "pv_en_20261126_dev_run1"):
    shutil.copytree(f"data/runs/{d}", runs / d)
reqs = load_requirements("data/eval/requirements_v0.csv", "test")
gold = {r["id"]: r for r in reqs}

print("\n=== report_tables offline checks ===")
# 1. no P
s1 = rt.main(["--runs-root", str(runs), "--out-dir", str(tmp / "out1"), "--compare-dir", str(tmp / "cmp1")])
ev = evaluate(reqs, {it["id"]: it for it in json.load(open("data/runs/b1_en_20261126/predictions.json"))["items"]})
check(abs(s1["scores"]["B1 (en)"]["mean"]["model_f1"] - ev["overall"]["model_f1"]) < 1e-4
      and s1["sources"]["B1 (en)"][0].endswith("b1_en_20261126"), "B1 en: latest run (m = 4), same as evaluate.py")
old = json.load(open("data/runs/compare/en_test_b1_vs_pnollm.json"))
p_old = next(t for t in old["results"][0]["tests"] if t["metric"] == "model_f1")["p_value"]
check(abs(s1["wilcoxon_vs_b1"]["P-noLLM (en)"]["model_f1"]["p_value"] - p_old) < 1e-6,
      f"Wilcoxon P-noLLM vs B1 (en) equals the committed compare.py result (p {p_old:.4f})")
tex = (tmp / "out1/tables/tab_rq1_vi.tex").read_text()
check("P-noLLM" in tex and "\\textbf{P}" not in tex and "0,695" in tex and "0,638" in tex,
      "no P run: P column skipped; Vietnamese numbers 0,695 and 0,638")
check(any("P (en, test)" in n for n in s1["notes"]) and (tmp / "out1/figures/fig_rq1_bar.pdf").exists(),
      "note about the missing P run; figure written without P")

# 2. synthetic P runs
base = json.load(open("data/runs/pnollm_en_20261126_all_run1/predictions.json"))
items = [it for it in base["items"] if it["id"] in gold]


def write(name, its, cfg, stopped=None):
    d = runs / name
    d.mkdir()
    (d / "predictions.json").write_text(json.dumps({"config": cfg, "lang": "en", "stopped": stopped,
                                                     "items": its}), encoding="utf-8")


for r, k in ((1, 4), (2, 6), (3, 8)):
    p_items, pv_items = [], []
    for i, it in enumerate(items):
        pv = copy.deepcopy(it)
        p = copy.deepcopy(it)
        if i < k:                                     # synthetic Verifier: gold models for the first k requirements
            p["models"] = gold[it["id"]]["gold_models"]
        p["verification"] = {"verified": True, "weak": False, "changed_module": False,
                             "changed_models": p["models"] != pv["models"],
                             "rounds": [{"verdict": "supported", "with_issues": ["x"] if i == 0 else []}]
                             + ([{"verdict": "partial", "with_issues": []}] if i == 1 else []),
                             "collector": [{"module": "x"}] if i == 0 else []}
        p["mapping"] = {"low_confidence": i == 0}
        p_items.append(p)
        pv_items.append(pv)
    write(f"pfull_en_2026113{r}_test_run{r}", p_items, "P")
    write(f"pv_en_2026113{r}_test_run{r}", pv_items, "P-V")
write("pfull_en_20261201_test_run1", p_items[:5], "P", stopped="DailyQuotaExceeded")   # unfinished, newer
write("pv_en_20261201_test_run1", pv_items[:5], "P-V", stopped="DailyQuotaExceeded")
s2 = rt.main(["--runs-root", str(runs), "--out-dir", str(tmp / "out2"), "--compare-dir", str(tmp / "cmp2")])
files = [Path(f).name for f in s2["sources"]["P (en)"]]
check(files == ["pfull_en_20261131_test_run1", "pfull_en_20261132_test_run2", "pfull_en_20261133_test_run3"],
      f"P: 3 complete runs, the unfinished newer run 1 is ignored: {files}")
f1 = [evaluate(reqs, {it["id"]: it for it in json.load(open(runs / f / "predictions.json"))["items"]})["overall"]
      ["model_f1"] for f in files]
check(abs(s2["scores"]["P (en)"]["mean"]["model_f1"] - sum(f1) / 3) < 1e-4, "P = mean of the 3 runs")
tex = (tmp / "out2/tables/tab_rq1_en.tex").read_text()
check("\\textbf{P}" in tex and "mean of 3 runs" in tex, "P column and caption in the English RQ1 table")
w = (tmp / "out2/tables/tab_wilcoxon_vi.tex").read_text()
check("P, tiếng Anh" in w and s2["wilcoxon_vs_b1"]["P (en)"]["model_f1"]["p_value"] < 1, "Wilcoxon P vs B1 row")
r3 = (tmp / "out2/tables/tab_rq3_vi.tex").read_text()
v = s2["rq3"]["test"]["verifier"]
check("Tập kiểm thử (16 câu)" in r3 and v["two_rounds"] == 1 and v["issues_in_prompt"] == 1 and v["low_confidence"] == 1
      and s2["rq3"]["test"]["wilcoxon_model_f1"] is not None, "RQ3: test columns, rounds, issues, tau, P vs P-V test")
check((tmp / "out2/tables/tab_p_runs_vi.tex").exists() and "Độ lệch chuẩn" in
      (tmp / "out2/tables/tab_p_runs_vi.tex").read_text(), "P run-by-run table with standard deviation")

# 3. conventions
alltex = "".join(p.read_text() for p in (tmp / "out2/tables").glob("*.tex"))
check(all(k in alltex for k in ("\\toprule", "\\midrule", "\\bottomrule", "tabularx", "table}[H]"))
      and "\u2014" not in alltex and "\u2013" not in alltex, "booktabs, tabularx, [H]; no em or en dash")
vi = (tmp / "out2/tables/tab_rq1_vi.tex").read_text()
check(" 0." not in vi and "0," in vi, "Vietnamese tables use the decimal comma")
pdf = (tmp / "out2/figures/fig_rq1_bar.pdf").read_bytes()
check(pdf.startswith(b"%PDF") and b"/Subtype /Image" not in pdf, "figure is a vector PDF (no raster image)")

print("ALL OK" if not FAILS else f"{len(FAILS)} FAILED")
sys.exit(1 if FAILS else 0)
