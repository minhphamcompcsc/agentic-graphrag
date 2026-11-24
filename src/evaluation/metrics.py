"""
Scoring rules for the evaluation set (data/eval/requirements_v0.csv), shared by every configuration
(B1, P, P-V). Agreed with the BA on 2026-11-24.

Module level (ranking):
  - A module is "correct" if it is the main module or an accepted module of the requirement.
  - Hit@k = 1 if a correct module is in the top k of the ranking.
  - MRR   = 1 / rank of the first correct module (0 if none).
  - Answer set (the modules the system finally proposes): Recall on the main module only;
    Precision counts accepted modules as not wrong (lenient) or as wrong (strict).

Model level (answer set of models):
  - Recall    = |pred ∩ gold| / |gold|                (gold = gold_models only)
  - Precision = |pred ∩ (gold ∪ accepted)| / |pred|   (lenient: accepted models are not counted as errors;
                                                       excluded models and any other model are errors)
  - Strict precision = |pred ∩ gold| / |pred|         (reported next to the lenient value)
  - An empty prediction gets precision 0 and F1 0.
Scores are averaged over requirements (macro average); per-requirement F1 is kept for the Wilcoxon test.
"""
import csv
from pathlib import Path


def split_list(cell: str) -> list:
    return [x.strip() for x in (cell or "").split(";") if x.strip()]


def load_requirements(path, split="all"):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        if split != "all" and (r.get("split") or "").strip() != split:
            continue
        out.append({
            "id": r["id"],
            "requirement_vi": r["requirement_vi"],
            "requirement_en": r["requirement_en"],
            "main_module": r["main_module"].strip(),
            "accepted_modules": split_list(r["accepted_modules"]),
            "gold_models": split_list(r["gold_models"]),
            "accepted_models": split_list(r["accepted_models"]),
            "excluded_models": split_list(r["excluded_models"]),
            "direct_depends": split_list(r["direct_depends"]),
            "split": (r.get("split") or "").strip(),
        })
    return out


def prf(n_pred, n_good_for_p, n_gold, n_good_for_r):
    p = n_good_for_p / n_pred if n_pred else 0.0
    r = n_good_for_r / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_one(req: dict, pred: dict, ks=(1, 3, 5)) -> dict:
    """pred = {"module_ranking": [...], "modules": [...answer set...], "models": [...answer set...]}"""
    ok_mods = {req["main_module"], *req["accepted_modules"]}
    ranking = pred.get("module_ranking") or pred.get("modules") or []
    rank = next((i + 1 for i, m in enumerate(ranking) if m in ok_mods), None)
    out = {f"hit@{k}": float(rank is not None and rank <= k) for k in ks}
    out["mrr"] = 1.0 / rank if rank else 0.0
    out["first_correct_rank"] = rank

    mods = list(dict.fromkeys(pred.get("modules") or []))
    main = {req["main_module"]}
    p, r, f = prf(len(mods), len(set(mods) & ok_mods), 1, len(set(mods) & main))
    ps, _, fs = prf(len(mods), len(set(mods) & main), 1, len(set(mods) & main))
    out.update(module_p=p, module_r=r, module_f1=f, module_p_strict=ps, module_f1_strict=fs)

    models = list(dict.fromkeys(pred.get("models") or []))
    gold, acc = set(req["gold_models"]), set(req["accepted_models"])
    p, r, f = prf(len(models), len(set(models) & (gold | acc)), len(gold), len(set(models) & gold))
    ps, _, fs = prf(len(models), len(set(models) & gold), len(gold), len(set(models) & gold))
    out.update(model_p=p, model_r=r, model_f1=f, model_p_strict=ps, model_f1_strict=fs)
    out["excluded_predicted"] = sorted(set(models) & set(req["excluded_models"]))
    return out


METRICS = ["hit@1", "hit@3", "hit@5", "mrr",
           "module_p", "module_r", "module_f1", "module_p_strict", "module_f1_strict",
           "model_p", "model_r", "model_f1", "model_p_strict", "model_f1_strict"]


def aggregate(per_question: list) -> dict:
    n = len(per_question)
    return {m: round(sum(q[m] for q in per_question) / n, 4) if n else None for m in METRICS} | {"n": n}


def evaluate(requirements: list, predictions: dict) -> dict:
    """predictions: id -> pred dict. Requirements without a prediction count as empty answers."""
    per_q = []
    for req in requirements:
        s = score_one(req, predictions.get(req["id"], {}))
        per_q.append({"id": req["id"], "split": req["split"], **s})
    return {"overall": aggregate(per_q), "per_question": per_q}


def write_per_question_csv(per_q: list, path: Path):
    cols = ["id", "split", "first_correct_rank"] + METRICS + ["excluded_predicted"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for q in per_q:
            row = {c: q.get(c) for c in cols}
            row["excluded_predicted"] = ";".join(q.get("excluded_predicted") or [])
            for m in METRICS:
                row[m] = round(row[m], 4)
            w.writerow(row)
