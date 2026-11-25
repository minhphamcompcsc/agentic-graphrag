"""
Baseline B1: plain vector RAG (no graph, no LLM).

Index (built from the same Odoo database and module texts as the KG):
  - one document per installed module:
      * modules with a Collector file (data/collected_v1/<module>.json): the same text the semantic
        layer reads (build_input_text: name, summary, manifest description, readme sections)
      * other modules: name, summary and description from ir_module_module (HTML and RST marks removed)
  - one document per model: "label (technical name)", the text used for entity linking
Embedding: bge-m3 through Ollama (same model as the KG). Cosine similarity.

For each requirement (requirement_en or requirement_vi):
  - module_ranking = all modules sorted by cosine
  - modules (answer set) = top --k modules
  - models (answer set)  = top --m models
Results go to data/runs/b1_<lang>_<YYYYMMDD>/ (predictions.json, corpus.json); score them with
src/evaluation/evaluate.py. Embeddings are cached in data/cache/ (not committed), so reruns are fast.

Usage (in ~/agentic-graphrag, venv activated):
  python src/baselines/b1_vector_rag.py --lang en
  python src/baselines/b1_vector_rag.py --lang vi
  python src/baselines/b1_vector_rag.py --lang en --k 1 --m 3
"""
import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "extractor"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from extract_semantic import build_input_text  # noqa: E402  (same module text as the semantic layer)
from metrics import load_requirements  # noqa: E402

load_dotenv(override=True)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
PG = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    dbname=os.getenv("PG_DB", "kg_demo"),
    user=os.getenv("PG_USER", "odoo"),
    password=os.getenv("PG_PASSWORD", "odoo"),
)
MAX_CHARS = 4000   # long core descriptions are cut (bge-m3 accepts more, but keeps documents comparable)

# description is a translated field (jsonb) in Odoo 18; the type is checked first to be safe
SQL_DESC_TYPE = """
SELECT data_type FROM information_schema.columns
WHERE table_name = 'ir_module_module' AND column_name = 'description'
"""
SQL_MODULES = """
SELECT m.name,
       COALESCE(m.shortdesc->>'en_US', m.name) AS label,
       COALESCE(m.summary->>'en_US', '')       AS summary,
       {desc}                                  AS description
FROM ir_module_module m
WHERE m.state = 'installed'
ORDER BY m.name
"""
SQL_MODELS = """
SELECT im.model AS name, COALESCE(im.name->>'en_US', im.model) AS label
FROM ir_model im
ORDER BY im.model
"""


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def clean_description(text: str) -> str:
    """Remove HTML tags and RST decoration lines, collapse blank lines."""
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    t = re.sub(r"^[=\-~^*#]{3,}\s*$", "", t, flags=re.M)      # RST title underlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def module_documents(rows, collected_dir: Path):
    docs = []
    for r in rows:
        f = collected_dir / f"{r['name']}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            text, source = build_input_text(rec), str(collected_dir)
        else:
            parts = [f"# {r['label']}"]
            if r["summary"]:
                parts.append(f"## SUMMARY\n{r['summary']}")
            desc = clean_description(r["description"])
            if desc:
                parts.append(f"## DESCRIPTION\n{desc}")
            text, source = "\n\n".join(parts), "ir_module_module"
        docs.append({"name": r["name"], "source": source, "text": text[:MAX_CHARS],
                     "chars": len(text), "truncated": len(text) > MAX_CHARS})
    return docs


class CachedEmbedder:
    """Embeds with Ollama and keeps vectors in data/cache/embeddings_<model>.npz (key = sha of the text)."""

    def __init__(self, cache_dir: Path, embed_fn=None):
        self.path = cache_dir / f"embeddings_{EMBED_MODEL}.npz"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.store = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.store = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.embed_fn = embed_fn
        self.new = 0

    def _backend(self):
        if self.embed_fn is None:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=EMBED_MODEL)
            self.embed_fn = lambda texts: emb.embed_documents(texts)
        return self.embed_fn

    def __call__(self, texts):
        keys = [sha(t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.store]
        for i in range(0, len(todo), 32):
            batch = todo[i:i + 32]
            vecs = np.array(self._backend()([t for _, t in batch]), dtype=np.float32)
            for (k, _), v in zip(batch, vecs):
                self.store[k] = v / np.linalg.norm(v)
            self.new += len(batch)
        if todo:
            np.savez(self.path, keys=np.array(list(self.store.keys())), vecs=np.stack(list(self.store.values())))
        return np.stack([self.store[k] for k in keys])


def read_odoo_rows(pg=None):
    """Installed modules (name, label, summary, description) and models (name, label) from the Odoo database.
    Shared with the Mapping Agent so that sim(q,m) uses exactly the B1 module documents."""
    with psycopg2.connect(**(pg or PG)) as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(SQL_DESC_TYPE)
        row = cur.fetchone()
        desc = {"jsonb": "COALESCE(m.description->>'en_US', '')", None: "''"}.get(
            row and row["data_type"], "COALESCE(m.description::text, '')")
        cur.execute(SQL_MODULES.format(desc=desc))
        mod_rows = [dict(r) for r in cur.fetchall()]
        cur.execute(SQL_MODELS)
        model_rows = [dict(r) for r in cur.fetchall()]
    return mod_rows, model_rows


def run(lang: str, k: int, m: int, requirements: str, collected_dir: Path, out_root: Path,
        cache_dir: Path, embed_fn=None, pg=None):
    t0 = time.time()
    mod_rows, model_rows = read_odoo_rows(pg)

    docs = module_documents(mod_rows, collected_dir)
    model_texts = [f"{r['label']} ({r['name']})" for r in model_rows]
    reqs = load_requirements(requirements)
    queries = [r[f"requirement_{lang}"] for r in reqs]

    embed = CachedEmbedder(cache_dir, embed_fn)
    mod_vecs = embed([d["text"] for d in docs])
    model_vecs = embed(model_texts)
    q_vecs = embed(queries)

    items = []
    for req, qv in zip(reqs, q_vecs):
        ms = mod_vecs @ qv
        order = np.argsort(-ms, kind="stable")
        mo = model_vecs @ qv
        morder = np.argsort(-mo, kind="stable")
        items.append({
            "id": req["id"],
            "query": req[f"requirement_{lang}"],
            "module_ranking": [docs[i]["name"] for i in order],
            "modules": [docs[i]["name"] for i in order[:k]],
            "models": [model_rows[i]["name"] for i in morder[:m]],
            "top_modules": [{"module": docs[i]["name"], "score": round(float(ms[i]), 4)} for i in order[:10]],
            "top_models": [{"model": model_rows[i]["name"], "score": round(float(mo[i]), 4)} for i in morder[:10]],
        })

    today = f"{date.today():%Y%m%d}"
    out_dir = out_root / f"b1_{lang}_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "config": "B1", "lang": lang, "date": today, "embedding": EMBED_MODEL, "llm_calls": 0,
        "params": {"k_modules": k, "m_models": m, "max_chars": MAX_CHARS},
        "corpus": {"modules": len(docs), "models": len(model_rows),
                   "modules_from_collector": sum(d["source"] != "ir_module_module" for d in docs)},
        "requirements_file": requirements, "new_embeddings": embed.new,
        "seconds": round(time.time() - t0, 1), "items": items,
    }
    (out_dir / "predictions.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus = [{"name": d["name"], "source": d["source"], "chars": d["chars"], "truncated": d["truncated"],
               "sha": sha(d["text"])} for d in docs]
    (out_dir / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_dir, run_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "vi"], default="en")
    ap.add_argument("--k", type=int, default=1, help="modules in the answer set (default 1)")
    ap.add_argument("--m", type=int, default=3, help="models in the answer set (default 3)")
    ap.add_argument("--requirements", default="data/eval/requirements_v0.csv")
    ap.add_argument("--collected-dir", default="data/collected_v1")
    ap.add_argument("--out-root", default="data/runs")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    out_dir, info = run(args.lang, args.k, args.m, args.requirements, Path(args.collected_dir),
                        Path(args.out_root), Path(args.cache_dir))
    c = info["corpus"]
    print(f"B1 ({args.lang}): {c['modules']} modules ({c['modules_from_collector']} from the Collector), "
          f"{c['models']} models, {len(info['items'])} requirements; new embeddings {info['new_embeddings']}; "
          f"{info['seconds']} s; 0 LLM calls")
    for it in info["items"]:
        top = ", ".join(f"{t['module']} {t['score']:.3f}" for t in it["top_modules"][:3])
        print(f"  {it['id']}: {top} | models: {', '.join(it['models'])}")
    print(f"Saved: {out_dir}/predictions.json, corpus.json")
    print(f"Next: python src/evaluation/evaluate.py {out_dir}/predictions.json")


if __name__ == "__main__":
    main()
