# Verifier Agent data

- `llm/<model>/run<r>/<lang>_<id>_round<k>.json`: one Verifier judgement (round k of requirement id). Each file keeps
  the full prompt (candidate dossiers), its SHA-256, the prompt version (`ver-v0.1`), the model, the time of the call
  and the structured output. A rerun with the same prompt reads the file and makes no call. The prompt depends on the
  Mapping candidates, so changing the Mapping parameters (`--params`) can change the prompt and cause new calls:
  choose the Mapping parameters first.
- `prompts/<lang>_<split>_run<r>/<lang>_<id>_round<k>.txt`: prompts written by `--dry-verifier` runs (no LLM call),
  to inspect what the Verifier would read.

Predictions of the full pipeline are written to `data/runs/pfull_<lang>_<date>_<split>_run<r>/predictions.json`
(configuration P) and `data/runs/pv_<lang>_<date>_<split>_run<r>/predictions.json` (P-V, Mapping only).
Each P item has a `verification` block: rounds (candidates, verdict, chosen module, models, invalid outputs,
evidence found, Collector evidence, cache use), rejected modules, Collector calls, and the Mapping answer it started from.
