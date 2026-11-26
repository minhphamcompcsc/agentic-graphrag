# Baseline parameters

- `params_b1_dev.json` (2026-11-26): `k` and `m` of baseline B1, chosen on the dev split (Q08, Q11, Q13, Q16) by
  `src/evaluation/tune_b1.py` from the B1 runs of 2026-11-24 (each keeps the top 10 models; 0 LLM calls).
  `k` = 1 for every configuration (B1, P, P-V). Objective: mean model F1 (lenient) over English and Vietnamese.
  English: m = 1 to 3 give 0, m = 4 gives 0.083 (best), m = 5 and 6 give 0.071 and 0.062; Vietnamese: 0 for every m.
  Chosen m = 4, which is also the m of pipeline P. B1 is then rerun with `--k 1 --m 4`; the B1 test scores of
  2026-11-24 (m = 3) are replaced by this run.
