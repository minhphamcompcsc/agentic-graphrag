"""
Verifier Agent (first version ver-v0.1): checks the candidates proposed by the Mapping Agent before the answer
goes to the BA.

Two kinds of checks:
  1. Deterministic checks on the KG (no LLM):
       - the module exists in the KG (installed) and every proposed model is in its footprint
         (DEFINES, EXTENDS, ADDS_FIELD);
       - dependencies: direct DEPENDS_ON (to install with the module) and the size of the transitive chain (R-21).
  2. One LLM judgement per round (structured output, cached): for the top candidates of the Mapping Agent,
     the LLM reads a "dossier" built ONLY from the KG and the module documentation (functions, models touched,
     fields added, dependencies, optional GitHub issues from the Collector) and returns:
       - fit of each candidate (yes, partly, no);
       - chosen_module (one of the candidates or null) and a verdict:
           supported | partial | not_supported | insufficient_evidence;
       - the models of the chosen module where the requirement is configured or where its logic lives
         (copied from the dossier list, validated against the KG);
       - missing parts of the requirement (customization gaps), evidence quotes, reason.
     The output is validated: a module that is not a candidate or a model that is not in the footprint is
     dropped (counted as invalid); evidence quotes are checked against the dossier text (fabrication check).

The agentic loop that uses this class (back to the Mapping Agent, call to the Collector) is in
src/verifier/run_pipeline.py.
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mapping"))
from analyze import NO_OUTPUT_WAIT, NoStructuredOutput, save_failures, unwrap  # noqa: E402  (shared with Mapping)

VERIFIER_VERSION = "ver-v0.1"
PROMPT_VERSION = "ver-v0.1"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_PAUSE = int(os.getenv("LLM_PAUSE", "13"))        # 5 calls/minute on the free tier
VERDICTS = ("supported", "partial", "not_supported", "insufficient_evidence")
ACCEPT = ("supported", "partial")
MAX_MODELS = 4                    # models the Verifier may return
DOSSIER_MODELS = 25               # models listed per candidate (most relevant first, from the Mapping Agent)
DOSSIER_FIELDS = 8                # fields listed per model
DOC_CHARS = 1500                  # module documentation characters per candidate
ISSUE_CHARS = 500                 # characters per GitHub issue from the Collector
NOISY_FIELDS = ("message_", "activity_", "website_message", "__last_update")


class CandidateCheck(BaseModel):
    module: str = Field(description="Candidate module name, copied exactly")
    fit: str = Field(description="'yes', 'partly' or 'no'")
    reason: str = Field(description="One short sentence based on the information given")


class Judgement(BaseModel):
    candidates: List[CandidateCheck] = Field(default_factory=list, description="One entry per candidate")
    chosen_module: Optional[str] = Field(default=None, description="The candidate that best fulfils the "
                                                                   "requirement, or null if none fits")
    verdict: str = Field(description="'supported', 'partial', 'not_supported' or 'insufficient_evidence'")
    models: List[str] = Field(default_factory=list, description="1 to 4 models of the chosen module, copied "
                                                               "exactly from its Models list, most important first")
    missing: List[str] = Field(default_factory=list, description="Parts of the requirement no candidate covers")
    evidence: List[str] = Field(default_factory=list, description="1 to 3 short quotes copied from the "
                                                                 "information given")
    reason: str = Field(default="", description="1 to 3 sentences")


PROMPT = """You are a senior Odoo 18 functional consultant. A search agent mapped the business requirement below
to candidate Odoo modules. Check its answer.

Requirement (written by a business analyst, may be in Vietnamese or English):
<<<
{requirement}
>>>
English restatement by the search agent: {normalized}

Candidate modules, best first. Everything below comes from a knowledge graph built from the installed
Odoo 18 database (models, fields, dependencies) and from the module documentation{issue_note}.

{dossiers}

Answer:
1. candidates: for each candidate, does its functionality fit the requirement? "yes", "partly" or "no",
   with one short reason based on the information above.
2. chosen_module: the candidate that best fulfils the requirement (copy its name), or null if none fits.
3. verdict:
   - "supported": the chosen module covers the requirement;
   - "partial": it covers the main need, but some parts need other settings or customization;
   - "not_supported": no candidate covers the requirement;
   - "insufficient_evidence": a candidate might fit, but the information above is too thin to decide.
4. models: 1 to 4 models of the chosen module, copied exactly from its "Models" list, where this requirement
   is configured or where the module's logic for it lives; most important first. Prefer models where the
   module has code (DEFINES or EXTENDS) or adds the setting fields; avoid models that only receive fields
   through delegation.
5. missing: parts of the requirement that no candidate covers (empty list if none).
6. evidence: 1 to 3 short quotes copied word for word from the information above.
7. reason: 1 to 3 sentences.
Only judge the listed candidates. You may use general Odoo knowledge to interpret them, but do not
invent features that the information above does not support.
"""


class DailyQuotaExceeded(RuntimeError):
    pass


class TransientLLMError(RuntimeError):
    """503, 504 (deadline expired), timeout: the model is overloaded; rerun later (answers are cached)."""


TRANSIENT = ("503", "504", "DeadlineExceeded", "Deadline expired", "DEADLINE_EXCEEDED", "UNAVAILABLE",
             "ServiceUnavailable", "high demand", "overloaded", "timed out", "Timeout", "ReadTimeout")
TRANSIENT_WAIT = int(os.getenv("LLM_TRANSIENT_WAIT", "30"))
ERROR_CHARS = 2000                                   # keep the quota details of Google errors in the logs


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def norm(s: str) -> str:
    s = re.sub(r"[*_`\\#>\"'“”‘’]", "", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


# ============ 1. deterministic checks on the KG ============
def transitive_deps(g, m):
    seen, todo = set(), list(g.deps.get(m, ()))
    while todo:
        d = todo.pop()
        if d not in seen:
            seen.add(d)
            todo += list(g.deps.get(d, ()))
    return seen


def kg_checks(g, module, models):
    fp = g.footprint(module) if module in g.modules else set()
    code = g.code_models(module) if module in g.modules else set()
    return {"module_in_kg": module in g.modules,
            "models_in_footprint": [x for x in models if x in fp],
            "models_not_in_footprint": [x for x in models if x not in fp],
            "models_without_code": [x for x in models if x in fp and x not in code],
            "direct_depends": sorted(g.deps.get(module, ())),
            "transitive_depends": len(transitive_deps(g, module))}


# ============ 2. dossier of one candidate ============
def dossier(res, feat, p, rank_models, module, rank, issues=None):
    """Text the LLM reads about one candidate + the list of models it may return for it."""
    g = res.g
    info = g.modules.get(module, {})
    src = "OCA" if info.get("source") == "oca" else "Odoo"
    lines = [f"### Candidate {rank}: {module} ({src} module, \"{info.get('label', '')}\")",
             f"Direct dependencies: {', '.join(sorted(g.deps.get(module, ()))) or 'none'}"]
    doc = (res.docs.get(module) or "").strip()
    if doc:
        cut = doc[:DOC_CHARS] + (" [...]" if len(doc) > DOC_CHARS else "")
        lines.append("Documentation:\n" + cut)
    funs = g.functions_of(module)
    if funs:
        lines.append("Business functions (knowledge graph):")
        lines += [f"- {g.functions[k]['name']}: {g.functions[k]['description']}" for k in funs]
    rows = rank_models(feat, module, p)
    shown = rows[:DOSSIER_MODELS]
    lines.append(f"Models (relation of the module to the model; fields the module adds){'' if len(rows) <= len(shown) else f', {len(shown)} most relevant of {len(rows)}'}:")
    for r in shown:
        x = r["model"]
        adds = [f for f, _ in g.added.get(module, {}).get(x, []) if not f.startswith(NOISY_FIELDS)
                and f not in ("id", "display_name", "create_uid", "create_date", "write_uid", "write_date")]
        rel = ", ".join(sorted(r["relations"]))
        note = "" if r["code"] else ", delegated: no code here"
        tail = f"; adds {', '.join(adds[:DOSSIER_FIELDS])}{' ...' if len(adds) > DOSSIER_FIELDS else ''}" if adds else ""
        lines.append(f"- {x} ({g.models.get(x, {}).get('label', '')}) [{rel}{note}]{tail}")
    if issues:
        lines.append("GitHub issues selected by the Collector agent (may describe older or newer versions):")
        for it in issues:
            lines.append(f"- #{it['number']} {it['title']} ({it['type']}, {it['state']}, {it['year']}): "
                         f"{it['text']}")
    return "\n".join(lines), [r["model"] for r in shown]


# ============ 3. LLM judgement ============
def make_llm(backend: str, model: str):
    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from analyze import no_hidden_retry       # same fix as the Mapping Agent (no silent 503 retries)
        llm = no_hidden_retry(ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0, timeout=LLM_TIMEOUT))
    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0)
    else:
        raise ValueError(backend)
    return llm.with_structured_output(Judgement, include_raw=True)


class Verifier:
    """judge(qid, lang, round, prompt) -> (Judgement dict, from_cache). Backends: gemini (official runs),
    ollama (prompt development), none (no LLM: the Mapping answer is passed through, prompts are saved)."""

    def __init__(self, backend="gemini", model=None, run=1, cache_root="data/verifier/llm", budget=None,
                 llm_fn=None, prompt_dir=None):
        self.backend = backend
        self.model = model or (os.getenv("GEMINI_MODEL", "gemini-3.6-flash") if backend == "gemini" else
                               os.getenv("OLLAMA_MODEL", "qwen2.5:3b") if backend == "ollama" else "none")
        self.dir = Path(cache_root) / self.model.replace(":", "_").replace("/", "_") / f"run{run}"
        self.calls, self.budget = 0, budget or (lambda: 10 ** 9)
        self._llm, self.llm_fn = None, llm_fn      # llm_fn(prompt) -> dict, used by offline tests
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.failures = []                         # failed structured-output attempts of the current call

    def _invoke(self, prompt: str) -> dict:
        if self.llm_fn:
            return self.llm_fn(prompt)
        if self._llm is None:
            self._llm = make_llm(self.backend, self.model)
        return unwrap(self._llm.invoke(prompt))

    def _call(self, prompt: str) -> dict:
        for attempt in (1, 2):
            try:
                return self._invoke(prompt)
            except NoStructuredOutput as e:
                self.failures.append({"attempt": attempt, "error": str(e), **e.detail})
                if attempt == 1:
                    print(f"    {str(e)[:160]}; trying once more in {NO_OUTPUT_WAIT} s")
                    time.sleep(NO_OUTPUT_WAIT if not self.llm_fn else 0)
                    continue
                raise TransientLLMError(f"{e} (twice; raw answers saved)")
            except Exception as e:  # noqa: BLE001
                msg = repr(e)
                if "PerDay" in msg or "per day" in msg:
                    raise DailyQuotaExceeded(msg[:ERROR_CHARS])
                if attempt == 1 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                    print("    per-minute limit, waiting 60 s and trying once more")
                    time.sleep(60)
                    continue
                if any(t in msg for t in TRANSIENT):
                    if attempt == 1:
                        print(f"    model overloaded or timeout ({msg[:80]}...), waiting {TRANSIENT_WAIT} s "
                              f"and trying once more")
                        time.sleep(TRANSIENT_WAIT if not self.llm_fn else 0)
                        continue
                    raise TransientLLMError(msg[:ERROR_CHARS])
                raise

    def judge(self, qid: str, lang: str, rnd: int, prompt: str):
        if self.prompt_dir:
            self.prompt_dir.mkdir(parents=True, exist_ok=True)
            (self.prompt_dir / f"{lang}_{qid}_round{rnd}.txt").write_text(prompt, encoding="utf-8")
        if self.backend == "none":
            return None, True
        path = self.dir / f"{lang}_{qid}_round{rnd}.json"
        if path.exists():
            c = json.loads(path.read_text(encoding="utf-8"))
            if c["prompt_sha256"] == sha(prompt) and c["prompt_version"] == PROMPT_VERSION:
                return c["output"], True
        if self.budget() <= 0:
            raise DailyQuotaExceeded("--max-llm-calls reached")
        t = time.time()
        self.failures = []
        try:
            out = self._call(prompt)
        except Exception:
            if self.failures:
                save_failures(self.dir / "errors", f"{lang}_{qid}_round{rnd}", self.failures, recovered=False)
            raise
        if self.failures:                          # first attempt failed, the retry worked: keep the evidence
            save_failures(self.dir / "errors", f"{lang}_{qid}_round{rnd}", self.failures, recovered=True)
        self.calls += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "id": qid, "lang": lang, "round": rnd, "backend": self.backend, "llm": self.model, "temperature": 0,
            "prompt_version": PROMPT_VERSION, "verifier_version": VERIFIER_VERSION,
            "called_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t, 1), "prompt_sha256": sha(prompt), "prompt": prompt,
            "output": out}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.llm_fn:
            time.sleep(LLM_PAUSE)
        return out, False


def build_prompt(requirement, normalized, dossiers, with_issues):
    note = " and from GitHub issues selected by a collector agent" if with_issues else ""
    return PROMPT.format(requirement=requirement, normalized=normalized or "(none)", issue_note=note,
                         dossiers="\n\n".join(dossiers))


def validate(raw: dict, candidates: list, valid_models: dict, prompt: str) -> dict:
    """Keeps only what the KG allows; records what was dropped."""
    out = {"verdict": None, "chosen_module": None, "models": [], "invalid_module": None, "invalid_models": [],
           "candidate_checks": [], "missing": [], "evidence": [], "evidence_found": 0, "reason": ""}
    if raw is None:
        return out
    j = Judgement(**raw)
    v = re.sub(r"[\s-]+", "_", (j.verdict or "").strip().lower().strip("\"'."))
    out["verdict"] = v if v in VERDICTS else "insufficient_evidence"
    chosen = (j.chosen_module or "").strip() or None
    if chosen and chosen not in candidates:
        out["invalid_module"], chosen = chosen, None
    out["chosen_module"] = chosen
    if chosen is None and out["verdict"] in ACCEPT:
        out["verdict"] = "insufficient_evidence"       # an accepting verdict needs a valid module
    allowed = valid_models.get(chosen, []) if chosen else []
    for x in j.models:
        x = x.strip()
        if x in allowed and x not in out["models"]:
            out["models"].append(x)
        elif x not in allowed:
            out["invalid_models"].append(x)
    out["models"] = out["models"][:MAX_MODELS]
    out["candidate_checks"] = [c.model_dump() for c in j.candidates]
    out["missing"], out["reason"] = j.missing, j.reason
    out["evidence"] = j.evidence[:3]
    text = norm(prompt.split("Answer:")[0])
    out["evidence_found"] = sum(1 for e in out["evidence"] if norm(e) and norm(e) in text)
    return out
