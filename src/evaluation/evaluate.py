"""
Scores a predictions file against the evaluation set (rules in src/evaluation/metrics.py).

Predictions file (written by every configuration, e.g. src/baselines/b1_vector_rag.py):
  {"config": "B1", "lang": "en", "params": {...},
   "items": [{"id": "Q01", "module_ranking": [...], "modules": [...], "models": [...]}, ...]}

Usage:
  python src/evaluation/evaluate.py data/runs/b1_en_20261124/predictions.json
  python src/evaluation/evaluate.py <predictions.json> --split test
Writes metrics_<split>.json and per_question_<split>.csv next to the predictions file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, load_requirements, write_per_question_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--split", default="all", help="all (default), dev or test (column 'split')")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    run = json.loads(pred_path.read_text(encoding="utf-8"))
    reqs = load_requirements(args.requirements, args.split)
    if not reqs:
        sys.exit(f"No requirement with split = {args.split}")
    preds = {it["id"]: it for it in run["items"]}
    missing = [r["id"] for r in reqs if r["id"] not in preds]
    res = evaluate(reqs, preds)

    o = res["overall"]
    print(f"{run.get('config')} ({run.get('lang')}), split {args.split}: {o['n']} requirements"
          + (f", {len(missing)} without prediction: {missing}" if missing else ""))
    print(f"  Module  Hit@1 {o['hit@1']:.3f}  Hit@3 {o['hit@3']:.3f}  Hit@5 {o['hit@5']:.3f}  MRR {o['mrr']:.3f}")
    print(f"  Module  P {o['module_p']:.3f}  R {o['module_r']:.3f}  F1 {o['module_f1']:.3f}"
          f"  (strict P {o['module_p_strict']:.3f}, F1 {o['module_f1_strict']:.3f})")
    print(f"  Model   P {o['model_p']:.3f}  R {o['model_r']:.3f}  F1 {o['model_f1']:.3f}"
          f"  (strict P {o['model_p_strict']:.3f}, F1 {o['model_f1_strict']:.3f})")
    for q in res["per_question"]:
        print(f"  {q['id']}: first correct rank {q['first_correct_rank'] or '-'}, "
              f"model P {q['model_p']:.2f} R {q['model_r']:.2f}"
              + (f", excluded predicted: {q['excluded_predicted']}" if q["excluded_predicted"] else ""))

    out = {"predictions": str(pred_path), "config": run.get("config"), "lang": run.get("lang"),
           "params": run.get("params"), "split": args.split, "requirements_file": args.requirements,
           "missing_predictions": missing, **res}
    mfile = pred_path.with_name(f"metrics_{args.split}.json")
    mfile.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_per_question_csv(res["per_question"], pred_path.with_name(f"per_question_{args.split}.csv"))
    print(f"Saved: {mfile}, per_question_{args.split}.csv")


if __name__ == "__main__":
    main()
