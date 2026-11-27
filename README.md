# Agentic GraphRAG for Odoo ERP

Master's thesis project (Ho Chi Minh City University of Technology, VNU-HCM), Internship 1 (CO5151).

**Goal:** map business requirements written by a Business Analyst (BA) to Odoo modules and data objects (models, fields), including community modules the system has not seen before, using an agentic GraphRAG pipeline over a 2-layer knowledge graph (KG).

## Research questions

- **RQ1:** Does a 2-layer KG (ORM schema + module text) map requirements more accurately than plain vector RAG?
- **RQ2:** For unseen modules, does an incrementally updated KG get close to a fully rebuilt KG, at a lower cost, and beat a static KG?
- **RQ3:** How much does each agent contribute (Collector coverage, Verifier error reduction)?

## Architecture

Two flows share one KG in Neo4j:

1. **Offline:** sources -> **Collector** -> **Extractor** -> KG
2. **Online:** BA requirement -> **Mapping** -> **Verifier** -> answer for the BA (modules, models, fields, required dependencies, KG paths as evidence)

Agentic loop: the Verifier sends weak answers back to Mapping; Mapping calls the Collector when no candidate is confident enough.

KG layers:

- **Structural layer** (no LLM): Odoo ORM metadata from PostgreSQL (`Module`, `Model`, `Field`; `DEPENDS_ON`, `DEFINES`, `EXTENDS`, `HAS_FIELD`, `ADDS_FIELD`, `RELATES_TO`, `LINKS_TO`).
- **Semantic layer** (LLM): business functions extracted from module documentation with iText2KG components (`Function`, `Concept`; `PROVIDES`, `INVOLVES`, `INVOLVES_FIELD`, `MENTIONS`), anchored on the structural layer.

## Data

- Odoo 18 Community + OCA modules (branch 18.0). No customer data.
- Seen modules (warehouse domain): `stock_no_negative`, `stock_picking_invoice_link`, `stock_picking_auto_create_lot`, `stock_move_location`, `stock_picking_back2draft`, `sale_stock_picking_invoice_link`.
- Unseen module: `stock_request` (OCA/stock-logistics-request).

## Repository layout

| Path | Content |
|---|---|
| `docker-compose.yml` | PostgreSQL 16, Odoo 18.0, Neo4j 5 + APOC |
| `check_env.py`, `src/test_env.py` | Environment checks |
| `src/extractor/extract_schema.py` | Structural layer: Odoo ORM schema -> Neo4j (full or incremental, v0.2) |
| `src/extractor/extract_semantic.py` | Semantic layer: LLM function extraction + entity linking |
| `src/extractor/eval_entity_links.py` | Entity linking accuracy from a labeled CSV |
| `src/collector/collect_modules.py` | Collector v0: manifest + readme |
| `src/collector/collect_modules_v1.py` | Collector v1: + GitHub issues/PRs selected by an LLM |
| `src/baselines/b1_vector_rag.py` | Baseline B1: plain vector RAG over module and model descriptions (no graph, no LLM) |
| `src/mapping/mapping_agent.py` | Mapping Agent (alone = configuration P-V): requirement analysis (1 LLM call), 2-level entry points, score = alpha sim + beta cov + gamma prox, models and KG path for the answer |
| `src/mapping/graph_view.py`, `src/mapping/analyze.py`, `src/mapping/tune.py` | In-memory KG view (Cypher loaders), LLM requirement analysis with cache, parameter search on the dev split |
| `src/verifier/verifier.py` | Verifier Agent: KG checks, candidate dossiers, 1 LLM judgement per round (validated against the KG) |
| `src/verifier/run_pipeline.py` | Full pipeline P (Mapping + Verifier, at most 2 rounds, back to Mapping, Collector in the loop) and P-V in one run |
| `src/demo/demo.py` | BA demo of the prototype: one requirement (Vietnamese or English, or `--id`) through pipeline P with the frozen parameters, readable answer (text, Markdown, JSON) |
| `src/evaluation/metrics.py`, `src/evaluation/evaluate.py` | Scoring rules (Hit@k, MRR, P/R/F1 at module and model level, lenient and strict) and scoring script |
| `src/evaluation/compare.py` | Paired Wilcoxon signed-rank comparison of configurations (exact p-values, effect size) |
| `src/evaluation/tune_b1.py`, `src/mapping/finalize_params.py` | Parameter choice on the dev split before the test runs: m of B1; frozen parameters of P with tau |
| `src/experiments/test_lightrag_neo4j.py` | Experiment: LightRAG with the project's Neo4j |
| `queries/` | Cypher queries for checks and analysis (results written as comments) |
| `data/collected/`, `data/collected_v1/` | Collector outputs |
| `data/semantic/` | Semantic layer LLM cache, entity links, BA reviews |
| `data/snapshots/` | KG statistics and schema snapshot |
| `data/eval/` | Evaluation set (requirements and answers checked against KG metadata) |
| `data/experiments/` | Side experiment results |
| `data/mapping/` | Mapping Agent LLM cache (`llm/<model>/run<r>/`), tuned parameters (`params_dev_en.json`) and frozen parameters for the test runs (`params_final_en.json`) |
| `data/baselines/` | Parameters of the baselines chosen on dev (`params_b1_dev.json`) |
| `data/verifier/` | Verifier LLM cache (`llm/<model>/run<r>/`, prompt included) and prompts of dry runs |
| `data/runs/` | Predictions and scores of each configuration (B1, P, ...) |
| `tests/` | Offline tests (incremental update, Mapping Agent and Verifier against a mock Odoo catalog; scoring rules) |
| `logs/` | Run logs (see `logs/README.md`) |

## Setup (summary)

1. WSL2 Ubuntu 24.04, Python 3.12, Docker Engine; `python -m venv .venv && pip install -r requirements.lock.txt`.
2. Clone OCA repositories (branch 18.0) into `oca/` and start the services: `docker compose up -d`.
3. Create the Odoo database `kg_demo` (no demo data) and install the seen modules.
4. Ollama with `bge-m3` for embeddings; a Gemini API key (`GOOGLE_API_KEY`, `GEMINI_MODEL`) in `.env`.
5. `python check_env.py` should report "Environment ready".

## Pipeline

```bash
python src/extractor/extract_schema.py                  # structural layer
python src/collector/collect_modules.py                 # Collector v0
python src/collector/collect_modules_v1.py --no-llm     # Collector v1 dry run (0 LLM calls)
python src/collector/collect_modules_v1.py              # Collector v1
python src/extractor/extract_semantic.py --dry-run      # semantic layer, no write
python src/extractor/extract_semantic.py                # semantic layer -> Neo4j
python src/extractor/eval_entity_links.py data/semantic/links_YYYYMMDD.csv
```

All LLM results are cached; reruns make no new LLM calls unless `--refresh` options are used.

### Baseline B1 and scoring

```bash
python src/baselines/b1_vector_rag.py --lang en           # 0 LLM calls; embeddings cached in data/cache/
python src/evaluation/evaluate.py data/runs/b1_en_YYYYMMDD/predictions.json
python src/baselines/b1_vector_rag.py --lang vi           # same requirements in Vietnamese
python tests/test_metrics.py                              # hand-checked cases of the scoring rules
```

Scoring rules (`src/evaluation/metrics.py`): a module counts as correct if it is the main or an accepted module
(Hit@k, MRR). Recall uses the gold answers only; precision does not count accepted answers as errors (lenient),
and a strict precision that does count them is reported next to it. Excluded models are errors.

### Mapping Agent

```bash
python src/mapping/mapping_agent.py --lang en --split dev --dry-llm     # plumbing check, 0 LLM calls
python src/mapping/mapping_agent.py --lang en --split dev               # 4 Gemini calls, cached in data/mapping/llm/
python src/mapping/tune.py --lang en                                    # grid search on dev, 0 calls
python src/mapping/mapping_agent.py --lang en --split test --params data/mapping/params_dev_en.json --run 1
python src/evaluation/evaluate.py data/runs/p_en_YYYYMMDD_test_run1/predictions.json --split test
```

For each requirement the LLM (one call) restates it in English and gives low-level keywords, high-level keywords and
business objects with a model guess (no module names). Low-level keywords find Function nodes, objects are linked to
Model nodes; every installed module gets `score = alpha * sim + beta * cov + gamma * prox`:
`sim` = best cosine with the module document (the B1 text) or one of its functions; `cov` = overlap between the
requirement objects and the module footprint (DEFINES, EXTENDS, ADDS_FIELD), Ochiai by default; `prox` = 1 / (1 + d)
with d the shortest KG path to an entry point (technical edges excluded, context fields cost more).
With beta = gamma = 0 and no LLM the module ranking equals B1 (checked in `tests/test_mapping_offline.py`).

Runs of the Mapping Agent alone are written to `data/runs/p_...` (dev runs of 2026-11-25); in the evaluation they are
the Mapping-only configuration (P-V).

### Verifier Agent and agentic loop (configuration P)

```bash
python src/verifier/run_pipeline.py --lang en --split dev --params data/mapping/params_dev_en.json --dry-verifier
#   0 LLM calls: Mapping from cache, Verifier prompts saved to data/verifier/prompts/ for inspection
python src/verifier/run_pipeline.py --lang en --split dev --params data/mapping/params_dev_en.json
#   1 to 2 Verifier calls per requirement (cached in data/verifier/llm/)
python src/evaluation/evaluate.py data/runs/pfull_en_YYYYMMDD_dev_run1/predictions.json --split dev
python src/evaluation/evaluate.py data/runs/pv_en_YYYYMMDD_dev_run1/predictions.json --split dev
```

For each requirement the Verifier receives the top 3 Mapping candidates as dossiers built only from the KG and the
module documentation (business functions, models touched with their relation, fields added, delegated models,
dependencies) and makes one LLM judgement: fit of each candidate, chosen module, verdict (`supported`, `partial`,
`not_supported`, `insufficient_evidence`), the models where the requirement is configured or implemented,
missing parts, evidence quotes. The output is validated against the KG (a module that is not a candidate and models
outside the footprint are dropped; evidence quotes are checked against the dossier). Agentic loop, at most 2 rounds:
`not_supported` sends the Mapping Agent back with the rejected modules excluded; `insufficient_evidence` or a Mapping
score below `tau` calls the Collector (GitHub issues of Collector v1, cached; Odoo core modules have no source).
If no round accepts, the Mapping answer is kept and flagged `verified = false`. The same run writes P (`pfull_...`)
and P-V (`pv_...`, Mapping only, identical to `mapping_agent.py`). Checked offline in `tests/test_verifier_offline.py`.

### BA demo (prototype)

`src/demo/demo.py` runs one requirement through pipeline P (same code as `run_pipeline.py`: Mapping, Verifier,
agentic loop, frozen parameters `data/mapping/params_final_en.json`) and prints an answer for a Business Analyst:
recommended module and other candidates, models and fields where the requirement is configured or where its logic
lives (with `DEFINES`, `EXTENDS`, `ADDS_FIELD`), modules to install with it, parts that need customization, the KG
path behind the choice, and the Verifier verdict with its quotes.

```bash
python src/demo/demo.py --id Q08 --dry-llm                      # 0 LLM calls (dev requirement, cached answers)
python src/demo/demo.py --id Q08 --format md --out docs/demo/demo_Q08_en.md
python src/demo/demo.py "Không cho xuất kho khi tồn kho không đủ" --dry-llm   # language detected (vi)
python src/demo/demo.py "Block delivery when stock is not enough" --max-llm-calls 2 --no-new-collect
```

- Cached answers are read first (caches of the official runs for `--run`, then `data/demo/llm/`); a cached
  requirement costs 0 calls. New calls are saved only in `data/demo/llm/`, never in the caches of the official runs.
- `--dry-llm` never calls the LLM: without a cached analysis the Mapping Agent runs without LLM (as P-noLLM) and the
  Verifier is skipped (answer flagged "not verified"). `--max-llm-calls` (default 3) caps Mapping + Verifier +
  Collector calls in the same way. `--show-gold` adds the expected answer for an `--id`.
- Checked offline in `tests/test_demo_offline.py` (same answer as `run_pipeline.py` from the same caches).

### Frozen parameters and official runs (test split)

Parameters are chosen on the dev split only (0 LLM calls), then frozen before any run of P on the test split:

```bash
python src/evaluation/tune_b1.py data/runs/b1_en_20261124/predictions.json data/runs/b1_vi_20261124/predictions.json
#   B1: k = 1 (as P), m chosen on dev -> data/baselines/params_b1_dev.json (m = 4)
python src/mapping/finalize_params.py --params data/mapping/params_dev_en.json \
    --dev-run data/runs/pv_en_20261126_dev_run1/predictions.json
#   P: Mapping parameters from tune.py + tau (lowest top score of a correct dev answer, rounded down to 0.05)
#   -> data/mapping/params_final_en.json (tau = 1.0)
```

Official runs (D25): P in English, 3 runs on the 16 test requirements (each run has its own LLM caches);
Vietnamese only without LLM (P-noLLM). About 2 Gemini calls per requirement; the free tier allows 20 calls per day,
so a run takes about 2 days: the script stops cleanly at the limit and the same command continues the next day
(finished requirements are read from the caches). Score only the last, complete run directory.

```bash
python src/baselines/b1_vector_rag.py --lang en --k 1 --m 4
python src/mapping/mapping_agent.py --lang vi --split all --dry-llm --params data/mapping/params_final_en.json
python src/verifier/run_pipeline.py --lang en --split test --run 1 --params data/mapping/params_final_en.json \
    --no-new-collect --max-llm-calls 19
python src/evaluation/evaluate.py data/runs/pfull_en_YYYYMMDD_test_run1/predictions.json --split test
```

### Statistical comparison (Wilcoxon signed-rank)

```bash
python src/evaluation/compare.py --split test --baseline B1 \
    --config B1=data/runs/b1_en_20261126/predictions.json \
    --config P-noLLM=data/runs/pnollm_en_20261126_all_run1/predictions.json \
    --config P=<run1>/predictions.json,<run2>/predictions.json,<run3>/predictions.json \
    --out data/runs/compare/en_test.json
python tests/test_compare.py          # exact p-values checked against scipy and a brute-force permutation
```

Per requirement, each configuration gets the scores of `src/evaluation/metrics.py` (mean over its runs).
Every configuration is compared with the baseline: two-sided Wilcoxon signed-rank test on the paired differences
(zero differences dropped, tied ranks averaged), exact p-value for up to 20 non-zero differences, rank-biserial
correlation as effect size, wins / ties / losses. Primary metric, fixed before the test runs: model F1 (lenient).
Other metrics are secondary; Holm-adjusted p-values are shown for reference. With 16 test requirements the tests
are preliminary (Internship 1).

### Incremental update with an unseen module (RQ2)

```bash
python src/extractor/extract_schema.py --check          # graph statistics, structural fingerprint, new modules (no write)
# install the module in Odoo, then:
python src/extractor/extract_schema.py --incremental    # structural layer: only modules installed but not in the graph
python src/extractor/extract_semantic.py --collected-dir data/collected_v1 --modules stock_request --dry-run --catalog neo4j
python src/extractor/extract_semantic.py --collected-dir data/collected_v1 --modules stock_request
python src/extractor/extract_schema.py                  # consistency check: a full run must create 0 nodes and 0 relationships
```

Each structural run writes `data/snapshots/kg_update_YYYYMMDD_<mode>.json` (rows read, nodes and relationships created,
time, graph statistics before and after, SHA-256 fingerprint of the structural layer). The incremental update is
consistent when the full run after it creates nothing and both runs end with the same fingerprint.

## Language

Code, comments, commit messages and documentation are in English. Exceptions:

- `requirement_vi` in `data/eval/requirements_*.csv`: the evaluation set is bilingual (Vietnamese and English requirements).
- Vietnamese test sentences in `check_env.py` and `src/test_env.py`: they check Vietnamese support of the embedding model.
- Vietnamese example requirements in the BA demo section above, in `tests/test_demo_offline.py` (inputs of the language detection test) and the Vietnamese letters used for detection in `src/demo/demo.py`.
- Logs created before 2026-11-24 (see `logs/README.md`).
