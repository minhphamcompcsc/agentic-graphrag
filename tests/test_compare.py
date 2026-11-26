"""
Checks of the Wilcoxon signed-rank test in src/evaluation/compare.py (no database, no network).
  1. mean ranks with ties
  2. exact p-values equal scipy (method="exact") on 200 random tie-free samples, n = 1 to 15
  3. with ties: the exact sign-flip p-value equals a brute-force permutation of the signed differences
  4. n > 20: the normal approximation is close to scipy (method="approx", correction=True)
  5. hand cases: all zero; all 16 positive (p = 2 / 2^16); Holm adjustment
Usage: python tests/test_compare.py   (scipy is in requirements.lock.txt; checks 2 and 4 need it)
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
from compare import holm, ranks_abs, wilcoxon  # noqa: E402

results = []
results.append(("mean ranks with ties", ranks_abs([0.5, -0.5, 0.25, 1.0]) == [2.5, 2.5, 1.0, 4.0]))

try:
    from scipy.stats import wilcoxon as sp_wilcoxon
except ImportError:
    sp_wilcoxon = None

rng = random.Random(7)
if sp_wilcoxon:
    worst = 0.0
    for _ in range(200):
        n = rng.randint(1, 15)
        d = rng.sample([x / 100 for x in range(-300, 301) if x], n)
        d = [x for i, x in enumerate(d) if abs(x) not in {abs(y) for y in d[:i]}]   # no tied |d|
        ours = wilcoxon(d)["p_value"]
        ref = sp_wilcoxon(d, method="exact").pvalue
        worst = max(worst, abs(ours - ref))
    results.append((f"exact p equals scipy on tie-free samples (max diff {worst:.1e})", worst < 1e-9))
else:
    print("    scipy not installed: check 2 skipped")


def brute_signed_rank(d):
    d = [x for x in d if abs(x) > 1e-12]
    rk = ranks_abs(d)
    obs = sum(r for r, x in zip(rk, d) if x > 0)
    mean = sum(rk) / 2
    hits = sum(1 for s in itertools.product((-1, 1), repeat=len(d))
               if abs(sum(r for r, sign in zip(rk, s) if sign > 0) - mean) >= abs(obs - mean) - 1e-9)
    return hits / 2 ** len(d)


ok = True
for _ in range(50):
    n = rng.randint(2, 12)
    d = [rng.choice([-1, -0.5, -0.25, 0, 0.25, 0.5, 1]) for _ in range(n)]
    if any(abs(x) > 1e-12 for x in d):
        ok &= abs(wilcoxon(d)["p_value"] - brute_signed_rank(d)) < 1e-12
results.append(("with ties: exact p equals brute-force sign permutation", ok))

if sp_wilcoxon:
    worst = 0.0
    for _ in range(50):
        n = rng.randint(21, 40)
        d = [rng.choice([-1, -0.5, -0.25, 0.25, 0.5, 1, 0.75]) for _ in range(n)]
        ours = wilcoxon(d)["p_value"]
        ref = sp_wilcoxon(d, method="approx", correction=True).pvalue
        worst = max(worst, abs(ours - ref))
    results.append((f"n > 20: normal approximation close to scipy (max diff {worst:.1e})", worst < 1e-6))

results.append(("all differences 0: p = 1", wilcoxon([0, 0, 0])["p_value"] == 1.0))
t = wilcoxon([0.5] * 10 + [1.0] * 6)
results.append(("16 positive differences: exact p = 2/2^16, r = 1",
                 abs(t["p_value"] - 2 / 2 ** 16) < 1e-15 and t["rank_biserial"] == 1.0))
results.append(("Holm adjustment", [round(x, 4) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]))

for name, good in results:
    print(f"{'OK  ' if good else 'FAIL'} {name}")
print("ALL OK" if all(g for _, g in results) else "SOME CHECKS FAILED")
sys.exit(0 if all(g for _, g in results) else 1)
