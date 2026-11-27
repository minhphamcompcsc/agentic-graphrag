"""
BA demo of the prototype: one requirement in, a readable mapping out.

The demo runs exactly the code of pipeline P (src/verifier/run_pipeline.py: Mapping Agent, Verifier Agent,
agentic loop with the Collector) with the frozen parameters (data/mapping/params_final_en.json, D26).
It does not re-implement any ranking or checking logic; it only resolves the input, routes the LLM caches
and formats the answer for a Business Analyst.

Input:
  a requirement text (Vietnamese or English; the language is detected), or
  --id Q08 to take a requirement of the evaluation set (data/eval/requirements_v0.csv).
  A text that is exactly a requirement of the evaluation set is treated as that --id (cached answers reused).

Output (terminal text by default; --format md or json, --out FILE to save):
  recommended module and other candidates; models and fields where the requirement is configured or where its
  logic lives (with DEFINES, EXTENDS, ADDS_FIELD); modules to install with it; parts of the requirement that
  need customization; the KG path that explains the choice; Verifier verdict, quotes and reason.

LLM calls and caches:
  - Cached answers are read first: the caches of the official runs (data/mapping/llm, data/verifier/llm,
    run --run) and then the demo cache (data/demo/llm). A cached requirement costs 0 calls.
  - New calls are written ONLY to the demo cache, never to the caches of the official runs, so the demo
    cannot change a run of the evaluation.
  - --dry-llm: never call the LLM. Without a cached analysis the Mapping Agent runs without LLM (query text
    only, as P-noLLM) and the Verifier is skipped (the Mapping answer is shown, flagged "not verified").
  - --max-llm-calls N (default 3) caps Mapping + Verifier + Collector calls; when the budget is used up the
    remaining steps are skipped in the same way as with --dry-llm.
  - --no-new-collect: the Collector only reads cached Collector v1 outputs (0 calls).

Usage (in ~/agentic-graphrag, venv activated, Docker services and Ollama running):
  python src/demo/demo.py --id Q08 --dry-llm                       # 0 LLM calls (dev requirement, cached)
  python src/demo/demo.py --id Q08 --format md --out docs/demo/demo_Q08_en.md
  python src/demo/demo.py "Khong cho xuat kho khi ton kho khong du" --dry-llm
  python src/demo/demo.py "Block delivery when stock is not enough" --max-llm-calls 2 --no-new-collect
"""
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/verifier", "src/mapping", "src/baselines", "src/extractor", "src/evaluation", "src/collector"):
    sys.path.insert(0, str(ROOT / sub))
import mapping_agent as ma  # noqa: E402
import run_pipeline as rp  # noqa: E402
from analyze import Analyzer, DailyQuotaExceeded as MapQuota, TransientLLMError as MapTransient  # noqa: E402
from metrics import load_requirements  # noqa: E402
from verifier import DailyQuotaExceeded as VerQuota, TransientLLMError as VerTransient, Verifier  # noqa: E402

DEMO_VERSION = "demo-v0.1"
VI_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
NOISY = ("message_", "activity_", "website_message")
SYSTEM = {"id", "display_name", "create_uid", "create_date", "write_uid", "write_date", "__last_update"}
LLM_ERRORS = (MapQuota, MapTransient, VerQuota, VerTransient)


def squash(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def detect_lang(text):
    return "vi" if VI_CHARS.search(text or "") else "en"


# ============ 1. input ============
def resolve_requirement(args):
    """Returns (id, text, lang, row of the evaluation set or None)."""
    rows = load_requirements(args.requirements, "all")
    if args.id:
        row = next((r for r in rows if r["id"].lower() == args.id.lower()), None)
        if row is None:
            raise SystemExit(f"Unknown id {args.id} (see {args.requirements})")
        lang = "en" if args.lang == "auto" else args.lang
        return row["id"], row[f"requirement_{lang}"], lang, row
    if not args.text:
        raise SystemExit("Give a requirement text or --id (see --help)")
    text = " ".join(args.text).strip()
    for r in rows:                                    # same text as an evaluation requirement: reuse its caches
        for lang in ("en", "vi"):
            if squash(r[f"requirement_{lang}"]) == squash(text) and args.lang in ("auto", lang):
                return r["id"], r[f"requirement_{lang}"], lang, r
    lang = detect_lang(text) if args.lang == "auto" else args.lang
    return "ADHOC_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10], text, lang, None


# ============ 2. LLM routing: official caches first, then the demo cache, new calls to the demo cache ============
class DemoAnalyzer:
    """Mapping analysis: cache of the official run, then the demo cache, then a new call (demo cache) or no LLM."""

    def __init__(self, args, budget, llm_fn=None):
        mk = lambda root, n: Analyzer(args.llm, args.llm_model, args.run, root, n, llm_fn)  # noqa: E731
        self.sources = [("official run cache", mk(args.llm_cache, 0)),
                        ("demo cache", mk(Path(args.demo_cache) / "mapping", 0))]
        self.new = mk(Path(args.demo_cache) / "mapping", 0)
        self.model = self.new.model
        self.args, self.budget, self.calls, self.mode, self.note = args, budget, 0, None, ""

    def analyze(self, qid, lang, text):
        for name, an in self.sources:
            try:
                out, _ = an.analyze(qid, lang, text)       # max_calls = 0: raises if not cached, never calls
                self.mode = f"LLM analysis ({self.model}, {name}, 0 new calls)"
                return out, True
            except MapQuota:
                pass
        reason = ("--dry-llm" if self.args.dry_llm else "LLM budget used up" if self.budget() <= 0 else None)
        if reason is None:
            self.new.max_calls = self.new.calls + 1
            try:
                out, _ = self.new.analyze(qid, lang, text)
                self.calls += 1
                self.mode = f"LLM analysis ({self.model}, new call, saved in the demo cache)"
                return out, False
            except LLM_ERRORS as e:
                reason = f"LLM error: {str(e)[:200]}"
        none = Analyzer("none")
        self.mode = f"no LLM (query text only, as P-noLLM): no cached analysis and {reason}"
        self.note = reason
        return none.analyze(qid, lang, text)[0], True


class DemoVerifier:
    """Same interface as verifier.Verifier.judge(); returns (None, True) when the Verifier is skipped, so that
    run_pipeline.verify_one passes the Mapping answer through (verified = false)."""

    def __init__(self, args, budget, llm_fn=None, enabled=True):
        mk = lambda root, b: Verifier(args.llm, args.llm_model, args.run, root, b, llm_fn)  # noqa: E731
        zero = lambda: 0  # noqa: E731
        self.sources = [("official run cache", mk(args.verifier_cache, zero)),
                        ("demo cache", mk(Path(args.demo_cache) / "verifier", zero))]
        self.new = mk(Path(args.demo_cache) / "verifier", budget)
        self.args, self.budget, self.enabled = args, budget, enabled
        self.calls, self.log = 0, []

    def judge(self, qid, lang, rnd, prompt):
        for name, ver in self.sources:
            try:
                out, _ = ver.judge(qid, lang, rnd, prompt)  # budget 0: raises if not cached, never calls
                self.log.append({"round": rnd, "source": name})
                return out, True
            except VerQuota:
                pass
        reason = ("Mapping ran without LLM" if not self.enabled else "--dry-llm" if self.args.dry_llm else
                  "--dry-verifier" if self.args.dry_verifier else
                  "LLM budget used up" if self.budget() <= 0 else None)
        if reason is None:
            try:
                out, _ = self.new.judge(qid, lang, rnd, prompt)
                self.calls += 1
                self.log.append({"round": rnd, "source": "new call (demo cache)"})
                return out, False
            except LLM_ERRORS as e:
                reason = f"LLM error: {str(e)[:200]}"
        self.log.append({"round": rnd, "source": f"skipped ({reason})"})
        return None, True


class DemoCollector(rp.CollectorLink):
    """Collector of the agentic loop: cached Collector v1 outputs; a new collect_v1() result goes to the demo
    folder instead of data/collected_v1."""

    def __init__(self, g, collected_dir, demo_dir, oca_dir, budget, allow_new):
        super().__init__(g, collected_dir, oca_dir, budget, allow_new)
        self.official, self.demo_dir = Path(collected_dir), Path(demo_dir)

    def issues(self, module):
        if module not in self.done:
            self.dir = self.official if (self.official / f"{module}.json").exists() else self.demo_dir
        return super().issues(module)


# ============ 3. run pipeline P on one requirement ============
def run_demo(args, embed_fn=None, llm_fn=None, ver_llm_fn=None, pg=None):
    t0 = time.time()
    qid, text, lang, row = resolve_requirement(args)
    args.lang = lang                                   # verify_one reads args.lang
    p = ma.Params(**json.loads(Path(args.params).read_text())["params"]) if args.params else ma.Params()
    if args.tau is not None:
        p.tau = args.tau
    res, embed = ma.load_resources(args.source, args.snapshot, args.semantic_dir, args.collected_dir,
                                   args.cache_dir, embed_fn, pg)
    g = res.g
    state = {}
    budget = lambda: args.max_llm_calls - state["an"].calls - state["ver"].calls - state["col"].state["llm_calls"]  # noqa: E731
    an = state["an"] = DemoAnalyzer(args, budget, llm_fn)
    state["ver"] = DemoVerifier(args, budget, ver_llm_fn)
    state["col"] = DemoCollector(g, args.collected_dir, Path(args.demo_cache) / "collected", args.oca_dir, budget,
                                 allow_new=not (args.no_new_collect or args.dry_llm))
    analysis, _ = an.analyze(qid, lang, text)
    state["ver"].enabled = not an.mode.startswith("no LLM")
    feat = ma.features(res, text, analysis, p)
    dec0 = ma.decide(feat, p, g)
    out = rp.verify_one({"id": qid}, text, analysis, feat, dec0, res, p, state["ver"], state["col"], args)
    report = build_report(qid, text, lang, row, analysis, feat, dec0, out, res, p, args, state, embed, t0)
    return report


# ============ 4. report (plain data; rendered as text, Markdown or JSON) ============
def transitive(g, m):
    seen, todo = set(), list(g.deps.get(m, ()))
    while todo:
        d = todo.pop()
        if d not in seen:
            seen.add(d)
            todo += list(g.deps.get(d, ()))
    return seen


def path_text(g, path):
    parts = []
    for x in path:
        if x in g.functions:
            parts.append(f'Function "{g.functions[x]["name"]}" ({g.functions[x]["module"]})')
        elif x.startswith("-["):
            parts.append(x)
        else:
            parts.append(x)
    return " ".join(parts)


def model_rows(g, m, models, feat, p):
    rel = {r["model"]: r for r in ma.rank_models(feat, m, p)}
    fun_fields = {k for f in g.functions_of(m) for k in g.functions[f].get("fields", [])}
    rows = []
    for x in models:
        added = [(f, lab) for f, lab in g.added.get(m, {}).get(x, []) if f not in SYSTEM and not f.startswith(NOISY)]
        named = [f for f, _ in added if f"{x}:{f}" in fun_fields]
        added.sort(key=lambda t: (t[0] not in named, t[0]))
        rows.append({"model": x, "label": g.models.get(x, {}).get("label", ""),
                     "relations": sorted(rel.get(x, {}).get("relations", [])),
                     "code": bool(rel.get(x, {}).get("code")),
                     "fields_added": [{"field": f, "label": lab, "in_function": f in named} for f, lab in added]})
    return rows


def build_report(qid, text, lang, row, analysis, feat, dec0, out, res, p, args, state, embed, t0):
    g = res.g
    m = out["modules"][0]
    v = out["verification"]
    a = out["answer"][0]
    info = g.modules.get(m, {})
    deps = sorted(g.deps.get(m, ()))
    chain = transitive(g, m)
    fits = {}
    for r in v["rounds"]:
        for c in r.get("candidate_checks") or []:
            fits[c.get("module")] = {"fit": c.get("fit"), "reason": c.get("reason")}
    score = {t["module"]: t for t in dec0["top_modules"]}
    others = []
    for x in [y for y in dec0["module_ranking"] if y != m][:args.show_candidates]:
        f = feat["modules"][x]
        others.append({"module": x, "label": g.modules.get(x, {}).get("label", ""),
                       "source": "OCA" if g.modules.get(x, {}).get("source") == "oca" else "Odoo",
                       "score": score.get(x, {}).get("score"), "sim": f["sim"],
                       "cov": f["cov_ochiai"] if p.cov == "ochiai" else f["cov_ratio"], "prox": f["prox"],
                       "verifier": fits.get(x)})
    last = v["rounds"][-1] if v["rounds"] else {}
    if v["verified"]:
        status = "verified (weak: thin evidence)" if v["weak"] else "verified"
    else:
        judged = [x for x in v["rounds"] if x.get("verdict")]
        status = ("not verified: Verifier skipped" if not judged else "not verified: no round accepted"
                  + ("" if last.get("verdict") else " (last round skipped)"))
    rep = {
        "demo_version": DEMO_VERSION, "pipeline_version": rp.PIPELINE_VERSION, "mapping_version": ma.MAPPING_VERSION,
        "verifier_version": rp.VERIFIER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requirement": {"id": qid, "text": text, "lang": lang, "in_evaluation_set": row is not None},
        "run": {"mapping": state["an"].mode,
                "verifier": [x["source"] for x in state["ver"].log] or ["not needed"],
                "new_llm_calls": {"mapping": state["an"].calls, "verifier": state["ver"].calls,
                                  "collector": state["col"].state["llm_calls"]},
                "max_llm_calls": args.max_llm_calls, "dry_llm": args.dry_llm, "cache_run": args.run,
                "params_file": args.params, "tau": p.tau, "graph_source": g.source, "graph": g.stats(),
                "new_embeddings": embed.new, "seconds": round(time.time() - t0, 1)},
        "analysis": analysis,
        "entry_points": {"functions": [{"name": f["name"], "module": f["module"], "score": f["score"]}
                                       for f in feat["entry_functions"][:3]],
                         "models": feat["entry_models"]},
        "recommendation": {
            "module": m, "label": info.get("label", ""), "source": "OCA" if info.get("source") == "oca" else "Odoo",
            "summary": info.get("summary", ""), "mapping_score": score.get(m, {}).get("score"),
            "status": status, "verdict": a.get("verdict"), "changed_by_verifier": v["changed_module"],
            "mapping_choice": dec0["modules"][0], "low_confidence": dec0["low_confidence"],
            "models": model_rows(g, m, out["models"], feat, p),
            "install": {"direct": [{"module": d, "source": "OCA" if g.modules.get(d, {}).get("source") == "oca"
                                    else "Odoo"} for d in deps],
                        "transitive_count": len(chain),
                        "oca_in_chain": sorted(d for d in chain if g.modules.get(d, {}).get("source") == "oca")},
            "functions": [{"name": g.functions[k]["name"], "description": g.functions[k]["description"]}
                          for k in g.functions_of(m)],
            "kg_path": path_text(g, a.get("path") or []),
            "missing": a.get("missing") or [],
            "reason": a.get("reason") or "",
            "kg_checks": a.get("kg_checks")},
        "other_candidates": others,
        "verifier_rounds": [{"round": r["round"], "candidates": r["candidates"], "verdict": r.get("verdict"),
                             "chosen_module": r.get("chosen_module"), "invalid_module": r.get("invalid_module"),
                             "invalid_models": r.get("invalid_models"), "with_issues": r.get("with_issues")}
                            for r in v["rounds"]],
        "rejected": v["rejected"], "collector": v["collector"],
    }
    quotes = a.get("evidence") or []                   # quotes of the accepted round
    acc = next((x for x in v["rounds"] if x.get("chosen_module") == m and x.get("evidence") == quotes), last)
    found = acc.get("evidence_found", 0)
    rep["recommendation"]["evidence"] = quotes
    rep["recommendation"]["evidence_found"] = f"{found}/{len(quotes)}" if quotes else None
    if row is not None and args.show_gold:
        rep["expected"] = {"main_module": row["main_module"], "accepted_modules": row["accepted_modules"],
                           "gold_models": row["gold_models"], "accepted_models": row["accepted_models"]}
    return rep


# ============ 5. rendering ============
def render(rep, md=False):
    L = []
    h = (lambda t: L.extend([f"## {t}", ""])) if md else (lambda t: L.extend(["", t.upper(), "-" * len(t)]))
    c = (lambda s: f"`{s}`") if md else (lambda s: s)
    b = (lambda s: f"**{s}**") if md else (lambda s: s)
    r, q, run = rep["recommendation"], rep["requirement"], rep["run"]
    L.append("# Agentic GraphRAG for Odoo: requirement mapping" if md else
             "AGENTIC GRAPHRAG FOR ODOO: REQUIREMENT MAPPING (prototype demo)")
    if md:
        L.append("")
    li = "- " if md else ""
    L.append(f"{li}Requirement ({q['id']}, {q['lang']}): {q['text']}")
    if rep["analysis"].get("normalized_en") and squash(rep["analysis"]["normalized_en"]) != squash(q["text"]):
        L.append(f"{li}Understood as: {rep['analysis']['normalized_en']}")
    if md:
        L.append("")

    h("Recommended module")
    L.append(f"- {b(c(r['module']))} ({r['source']}, \"{r['label']}\"): {r['status']}"
             + (f", verdict {b(r['verdict'])}" if r["verdict"] else ""))
    if r["summary"]:
        L.append(f"- Summary: {r['summary']}")
    if r["mapping_score"] is not None:
        L.append(f"- Mapping score {r['mapping_score']:.3f} (tau {run['tau']})"
                 + ("; LOW CONFIDENCE, Collector asked for more evidence" if r["low_confidence"] else ""))
    if r["changed_by_verifier"]:
        L.append(f"- The Verifier changed the Mapping choice ({c(r['mapping_choice'])} -> {c(r['module'])})")
    for f in r["functions"]:
        L.append(f"- Function (KG): {f['name']}: {f['description']}")
    if md:
        L.append("")

    h("Where to configure it or where the logic lives (models and fields)")
    if md:
        L += ["| Model | Relation of the module | Fields the module adds |", "|---|---|---|"]
    for x in r["models"]:
        rel = ", ".join(x["relations"]) + ("" if x["code"] else " (delegated through _inherits, no code here)")
        fl = [f"{fd['field']} ({fd['label']})" + (" *" if fd["in_function"] else "") for fd in x["fields_added"][:8]]
        more = f" and {len(x['fields_added']) - 8} more" if len(x["fields_added"]) > 8 else ""
        if md:
            L.append(f"| `{x['model']}` ({x['label']}) | {rel} | {', '.join(fl) + more or '(none)'} |")
        else:
            L.append(f"- {x['model']} ({x['label']}) [{rel}]")
            L.append(f"    fields added: {', '.join(fl) + more if fl else '(none)'}")
    L.append(("" if not md else "\n") + "Relations: DEFINES = the module creates the model; EXTENDS = the module "
             "changes its code; ADDS_FIELD = the module adds fields. * = field named by a business function.")
    if md:
        L.append("")

    h("Modules to install with it")
    ins = r["install"]
    L.append("- Direct dependencies: " + (", ".join(f"{c(d['module'])} ({d['source']})" for d in ins["direct"])
                                       or "none"))
    L.append(f"- Full dependency chain: {ins['transitive_count']} modules"
             + (f"; OCA modules in the chain: {', '.join(c(x) for x in ins['oca_in_chain'])}"
                if ins["oca_in_chain"] else "; all of them Odoo core"))
    if md:
        L.append("")

    h("Not covered (needs customization)")
    if r["missing"]:
        L += [f"- {x}" for x in r["missing"]]
    else:
        L.append("- Nothing reported by the Verifier" if r["verdict"] else "- Unknown (the Verifier did not run)")
    if md:
        L.append("")

    h("Why: path in the knowledge graph")
    L.append(f"- {c(r['kg_path'])} (from the module to the closest entry point of the requirement)"
             if r["kg_path"] else "- No path within the distance limit")
    ep = rep["entry_points"]
    if ep["models"]:
        L.append("- Objects in the requirement linked to models: " + "; ".join(
            f"\"{e['mention']}\" -> {c(e['model']) if e['model'] else 'not linked'} ({e['method']})"
            for e in ep["models"]))
    if ep["functions"]:
        L.append("- Closest business functions: " + "; ".join(
            f"{f['name']} ({c(f['module'])}, {f['score']:.3f})" for f in ep["functions"]))
    if md:
        L.append("")

    h("Verifier")
    if r["verdict"]:
        L.append(f"- Verdict: {b(r['verdict'])}" + (f"; reason: {r['reason']}" if r["reason"] else ""))
        for e in r["evidence"]:
            L.append(f"- Quote: \"{e}\"")
        if r["evidence_found"]:
            L.append(f"- Quotes found word for word in the KG and documentation given to the Verifier: "
                     f"{r['evidence_found']}")
    else:
        L.append("- The Verifier did not run: " + "; ".join(run["verifier"]))
    for rd in rep["verifier_rounds"]:
        L.append(f"- Round {rd['round']}: candidates {', '.join(rd['candidates'])} -> "
                 f"{rd['verdict'] or 'skipped'}" + (f", chose {rd['chosen_module']}" if rd["chosen_module"] else "")
                 + (" (with GitHub issues)" if rd["with_issues"] else ""))
    if rep["rejected"]:
        L.append(f"- Rejected and sent back to the Mapping Agent: {', '.join(rep['rejected'])}")
    for col in rep["collector"]:
        L.append(f"- Collector: {col['module']} ({col['why']}): {col['status']}, {col['issues']} issue(s)")
    if md:
        L.append("")

    h("Other candidate modules")
    if md:
        L += ["| Module | Source | Score | sim | cov | prox | Verifier |", "|---|---|---|---|---|---|---|"]
    for o in rep["other_candidates"]:
        vf = f"{o['verifier']['fit']}: {o['verifier']['reason']}" if o["verifier"] else ""
        sc = f"{o['score']:.3f}" if o["score"] is not None else ""
        if md:
            L.append(f"| `{o['module']}` | {o['source']} | {sc} | {o['sim']:.3f} | {o['cov']:.3f} | "
                     f"{o['prox']:.3f} | {vf} |")
        else:
            L.append(f"- {o['module']} ({o['source']}) score {sc} (sim {o['sim']:.3f}, cov {o['cov']:.3f}, "
                     f"prox {o['prox']:.3f})" + (f"; Verifier: {vf}" if vf else ""))
    if md:
        L.append("")

    if "expected" in rep:
        h("Expected answer (evaluation set)")
        e = rep["expected"]
        L.append(f"- Main module: {c(e['main_module'])}"
                 + (f"; also accepted: {', '.join(e['accepted_modules'])}" if e["accepted_modules"] else ""))
        L.append(f"- Gold models: {', '.join(e['gold_models'])}"
                 + (f"; also accepted: {', '.join(e['accepted_models'])}" if e["accepted_models"] else ""))
        if md:
            L.append("")

    h("Run")
    n = run["new_llm_calls"]
    L.append(f"- Mapping: {run['mapping']}")
    L.append(f"- Verifier: {'; '.join(f'round {i}: {s}' for i, s in enumerate(run['verifier'], 1))}")
    L.append(f"- New LLM calls: Mapping {n['mapping']}, Verifier {n['verifier']}, Collector {n['collector']} "
             f"(limit {run['max_llm_calls']}{', --dry-llm' if run['dry_llm'] else ''})")
    L.append(f"- KG: {run['graph_source']}, {run['graph']['modules']} modules, {run['graph']['models']} models, "
             f"{run['graph']['functions']} functions; parameters {run['params_file']}; {run['seconds']} s")
    L.append(f"- {rep['demo_version']} on {rep['pipeline_version']} ({rep['mapping_version']} + {rep['verifier_version']})")
    return "\n".join(L) + "\n"


def build_parser():
    ap = rp.build_parser()
    ap.description = "BA demo: map one requirement with pipeline P and print a readable answer"
    ap.add_argument("text", nargs="*", help="requirement text (Vietnamese or English)")
    ap.add_argument("--id", help="take the requirement from the evaluation set, e.g. Q08")
    ap.add_argument("--format", choices=["text", "md", "json"], default="text")
    ap.add_argument("--out", help="save the output to this file (the terminal still shows the text version)")
    ap.add_argument("--demo-cache", default="data/demo/llm", help="cache of new LLM calls made by the demo")
    ap.add_argument("--show-candidates", type=int, default=4, help="other candidate modules to show")
    ap.add_argument("--show-gold", action="store_true", help="also show the expected answer (--id only)")
    lang = ap._option_string_actions["--lang"]
    lang.choices, lang.default, lang.help = ["auto", "en", "vi"], "auto", "auto: detected from the text (--id: en)"
    ap._option_string_actions["--dry-llm"].help = ("never call the LLM: cached answers only; otherwise Mapping "
                                                   "without LLM and no Verifier")
    ap.set_defaults(params="data/mapping/params_final_en.json", max_llm_calls=3)
    return ap


def main(argv=None, **fakes):
    args = build_parser().parse_args(argv)
    rep = run_demo(args, **fakes)
    text = render(rep)
    body = {"text": text, "md": render(rep, md=True),
            "json": json.dumps(rep, ensure_ascii=False, indent=2) + "\n"}[args.format]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(text + f"\nSaved ({args.format}): {args.out}")
    else:
        print(body)
    return rep


if __name__ == "__main__":
    main()
