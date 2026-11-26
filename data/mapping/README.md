# Mapping Agent data

- `llm/<model>/run<r>/<lang>_<id>.json`: the LLM analysis of each requirement (one call per requirement and run).
  Each file keeps the input text, its SHA-256, the prompt version (`map-v0.1`), the model, the time of the call and the
  structured output. A rerun with the same input and prompt version reads the file and makes no call.
- `params_dev_<lang>.json`: decision parameters chosen by `src/mapping/tune.py` on the dev split
  (Q08, Q11, Q13, Q16), with the grid, the objective and the score of the default parameters.

Predictions are written to `data/runs/p_<lang>_<date>_<split>_run<r>/predictions.json` and scored with
`src/evaluation/evaluate.py`. Runs without LLM (`--dry-llm`) are written to `data/runs/pnollm_...`.
- `params_final_en.json` (2026-11-26): parameters frozen for the official runs of P on the test split, written by
  `src/mapping/finalize_params.py`. Mapping parameters are those of `params_dev_en.json` (unchanged); `tau` = 1.0 is
  the lowest top Mapping score among the correct dev answers (1.0269, Q13) rounded down to 0.05, so no dev requirement
  is flagged low confidence and the dev results of P and P-V stay valid. With 4 dev requirements this is a rule of
  thumb, not a tuned value. Below `tau` the pipeline adds the cached GitHub issues of Collector v1 to the Verifier
  input (0 LLM calls with `--no-new-collect`).
