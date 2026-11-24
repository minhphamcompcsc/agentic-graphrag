"""Hand-checked cases for the scoring rules in src/evaluation/metrics.py (no database needed).

    python tests/test_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
from metrics import evaluate, load_requirements, score_one  # noqa: E402

REQS = {r["id"]: r for r in load_requirements(Path(__file__).resolve().parents[1] / "data/eval/requirements_v0.csv")}


def close(a, b):
    return abs(a - b) < 1e-9


def check(name, cond):
    print(("OK   " if cond else "FAIL ") + name)
    return cond


ok = True
# Q01: gold = product.template, product.category, stock.location, stock.quant; accepted = product.product
s = score_one(REQS["Q01"], {"module_ranking": ["stock", "stock_no_negative"], "modules": ["stock"],
                            "models": ["product.product", "stock.quant", "res.partner"]})
ok &= check("Q01 first correct rank 2, Hit@1 0, Hit@3 1, MRR 0.5",
            s["first_correct_rank"] == 2 and s["hit@1"] == 0 and s["hit@3"] == 1 and close(s["mrr"], 0.5))
ok &= check("Q01 model R = 1/4 (only stock.quant is gold)", close(s["model_r"], 0.25))
ok &= check("Q01 model P lenient = 2/3 (accepted product.product is not an error)", close(s["model_p"], 2 / 3))
ok &= check("Q01 model P strict = 1/3", close(s["model_p_strict"], 1 / 3))
ok &= check("Q01 module answer 'stock' is wrong: P 0, R 0", s["module_p"] == 0 and s["module_r"] == 0)

# Q02: main = stock_picking_invoice_link, accepted = sale_stock_picking_invoice_link; excluded models listed
s = score_one(REQS["Q02"], {"module_ranking": ["sale_stock_picking_invoice_link", "stock_picking_invoice_link"],
                            "modules": ["sale_stock_picking_invoice_link"],
                            "models": ["stock.picking", "account.bank.statement.line"]})
ok &= check("Q02 accepted module at rank 1 counts as a hit", s["hit@1"] == 1 and close(s["mrr"], 1.0))
ok &= check("Q02 module answer = accepted only: lenient P 1, R 0 (recall on the main module), strict P 0",
            s["module_p"] == 1 and s["module_r"] == 0 and s["module_p_strict"] == 0)
ok &= check("Q02 excluded model counted as an error: model P 1/2, R 1/2",
            close(s["model_p"], 0.5) and close(s["model_r"], 0.5) and s["excluded_predicted"] == ["account.bank.statement.line"])

# Q05: perfect answer
s = score_one(REQS["Q05"], {"module_ranking": ["stock_picking_back2draft"], "modules": ["stock_picking_back2draft"],
                            "models": ["stock.picking"]})
ok &= check("Q05 perfect: module F1 1, model F1 1", close(s["module_f1"], 1) and close(s["model_f1"], 1))

# Empty prediction and missing prediction
s = score_one(REQS["Q04"], {})
ok &= check("Q04 empty prediction: everything 0", s["mrr"] == 0 and s["model_f1"] == 0 and s["module_f1"] == 0)
res = evaluate([REQS["Q04"], REQS["Q05"]], {"Q05": {"modules": ["stock_picking_back2draft"],
                                                    "models": ["stock.picking"]}})
ok &= check("Macro average over 2 requirements (one missing): model F1 0.5", close(res["overall"]["model_f1"], 0.5))

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
