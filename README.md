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
| `src/mapping/mapping_agent.py` | Mapping Agent (configuration P): requirement analysis (1 LLM call), 2-level entry points, score = alpha sim + beta cov + gamma prox, models and KG path for the answer |
| `src/mapping/graph_view.py`, `src/mapping/analyze.py`, `src/mapping/tune.py` | In-memory KG view (Cypher loaders), LLM requirement analysis with cache, parameter search on the dev split |
| `src/evaluation/metrics.py`, `src/evaluation/evaluate.py` | Scoring rules (Hit@k, MRR, P/R/F1 at module and model level, lenient and strict) and scoring script |
| `src/experiments/test_lightrag_neo4j.py` | Experiment: LightRAG with the project's Neo4j |
| `queries/` | Cypher queries for checks and analysis (results written as comments) |
| `data/collected/`, `data/collected_v1/` | Collector outputs |
| `data/semantic/` | Semantic layer LLM cache, entity links, BA reviews |
| `data/snapshots/` | KG statistics and schema snapshot |
| `data/eval/` | Evaluation set (requirements and answers checked against KG metadata) |
| `data/experiments/` | Side experiment results |
| `data/mapping/` | Mapping Agent LLM cache (`llm/<model>/run<r>/`) and tuned parameters |
| `data/runs/` | Predictions and scores of each configuration (B1, P, ...) |
| `tests/` | Offline tests (incremental update and Mapping Agent against a mock Odoo catalog; scoring rules) |
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

### Mapping Agent (configuration P)

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
- Logs created before 2026-11-24 (see `logs/README.md`).
