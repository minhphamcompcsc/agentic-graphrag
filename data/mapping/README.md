# Mapping Agent data

- `llm/<model>/run<r>/<lang>_<id>.json`: the LLM analysis of each requirement (one call per requirement and run).
  Each file keeps the input text, its SHA-256, the prompt version (`map-v0.1`), the model, the time of the call and the
  structured output. A rerun with the same input and prompt version reads the file and makes no call.
- `params_dev_<lang>.json`: decision parameters chosen by `src/mapping/tune.py` on the dev split
  (Q08, Q11, Q13, Q16), with the grid, the objective and the score of the default parameters.

Predictions are written to `data/runs/p_<lang>_<date>_<split>_run<r>/predictions.json` and scored with
`src/evaluation/evaluate.py`. Runs without LLM (`--dry-llm`) are written to `data/runs/pnollm_...`.
