"""
Requirement analysis: the ONE LLM call the Mapping Agent makes per requirement (D15).

The LLM turns the BA requirement (Vietnamese or English) into:
  normalized_en        the requirement in plain English (module texts are in English)
  low_level_keywords   specific entities: objects, documents, settings, actions, states
  high_level_keywords  broader themes
  objects              business objects + best guess of the Odoo 18 model
This is the 2-level keyword idea of LightRAG (Guo et al., 2024), implemented on our own KG (D13):
low-level keywords look for Function nodes, high-level keywords look for modules.
The LLM is NOT asked for module names, so the module choice comes from the KG and the embeddings.

Every answer is cached in data/mapping/llm/<model>/run<r>/<lang>_<id>.json (input, prompt version, output),
so tuning and reruns cost 0 calls. Backends: gemini (official runs, D12), ollama (prompt development),
none (no LLM: ablation and offline tests).
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

PROMPT_VERSION = "map-v0.1"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_PAUSE = int(os.getenv("LLM_PAUSE", "13"))        # 5 calls/minute on the free tier


class ObjectGuess(BaseModel):
    mention: str = Field(description="Business object exactly as the requirement names it (translated to English)")
    odoo_model: Optional[str] = Field(default=None, description="Best guess of the Odoo 18 technical model, "
                                                                "e.g. 'stock.picking'; null if not confident")


class RequirementAnalysis(BaseModel):
    normalized_en: str = Field(description="The requirement restated in plain English business language, "
                                           "1 to 2 sentences, keeping every condition and exception")
    low_level_keywords: List[str] = Field(description="3 to 8 specific terms: business objects, documents, "
                                                      "settings, actions or states")
    high_level_keywords: List[str] = Field(description="2 to 5 broader business themes")
    objects: List[ObjectGuess] = Field(default_factory=list, description="Business objects involved")


PROMPT = """You are an Odoo 18 functional consultant. A business analyst wrote the requirement below
(it may be in Vietnamese or in English). Analyse it so that it can be searched in a knowledge graph of
Odoo modules, data models and business functions.

Requirement:
<<<
{requirement}
>>>

Fill the fields:
- normalized_en: restate the requirement in plain English business language (1 to 2 sentences).
  Keep every condition and exception.
- low_level_keywords: 3 to 8 specific terms (business objects, documents, settings, actions, states),
  e.g. "cancelled transfer", "draft state", "lot number".
- high_level_keywords: 2 to 5 broader themes, e.g. "transfer management", "inventory control".
- objects: the business objects involved, each with its name as written (in English) and your best
  guess of the Odoo 18 technical model (e.g. "stock.picking"); null if not confident.
Do not name Odoo or OCA modules and do not propose a solution: only describe what is asked.
"""


class DailyQuotaExceeded(RuntimeError):
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def make_llm(backend: str, model: str):
    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0, timeout=LLM_TIMEOUT)
    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0)
    else:
        raise ValueError(backend)
    return llm.with_structured_output(RequirementAnalysis)


class Analyzer:
    """analyze(id, lang, text) -> (RequirementAnalysis as dict, from_cache)."""

    def __init__(self, backend="gemini", model=None, run=1, cache_root="data/mapping/llm",
                 max_calls=40, llm_fn=None):
        self.backend = backend
        self.model = model or (os.getenv("GEMINI_MODEL", "gemini-3.6-flash") if backend == "gemini" else
                               os.getenv("OLLAMA_MODEL", "qwen2.5:3b") if backend == "ollama" else "none")
        self.dir = Path(cache_root) / self.model.replace(":", "_").replace("/", "_") / f"run{run}"
        self.max_calls, self.calls = max_calls, 0
        self._llm, self.llm_fn = None, llm_fn     # llm_fn(prompt) -> dict, used by offline tests

    def _call(self, prompt: str) -> dict:
        if self.llm_fn:
            return self.llm_fn(prompt)
        if self._llm is None:
            self._llm = make_llm(self.backend, self.model)
        for attempt in (1, 2):
            try:
                out = self._llm.invoke(prompt)
                if out is None:
                    raise RuntimeError("LLM returned no structured output")
                return out.model_dump()
            except Exception as e:  # noqa: BLE001
                msg = repr(e)
                if "PerDay" in msg or "per day" in msg:
                    raise DailyQuotaExceeded(msg[:300])
                if attempt == 1 and ("429" in msg or "RESOURCE_EXHAUSTED" in msg):
                    print("    per-minute limit, waiting 60 s and trying once more")
                    time.sleep(60)
                    continue
                raise

    def analyze(self, qid: str, lang: str, text: str):
        if self.backend == "none":
            return {"normalized_en": text if lang == "en" else "", "low_level_keywords": [],
                    "high_level_keywords": [], "objects": []}, True
        path = self.dir / f"{lang}_{qid}.json"
        if path.exists():
            c = json.loads(path.read_text(encoding="utf-8"))
            if c["input_sha256"] == sha(text) and c["prompt_version"] == PROMPT_VERSION:
                return c["output"], True
        if self.calls >= self.max_calls:
            raise DailyQuotaExceeded(f"--max-llm-calls {self.max_calls} reached")
        out = self._call(PROMPT.format(requirement=text))
        self.calls += 1
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "id": qid, "lang": lang, "backend": self.backend, "llm": self.model, "temperature": 0,
            "prompt_version": PROMPT_VERSION, "called_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_sha256": sha(text), "input": text, "output": out}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.llm_fn:
            time.sleep(LLM_PAUSE)
        return out, False
