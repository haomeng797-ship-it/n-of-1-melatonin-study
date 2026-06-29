"""
10_equivalence_test.py
Two one-sided tests (TOST) equivalence test for the melatonin mean effect on
daily mood. Complements the between-condition Welch test (Section 3.4) by asking
whether the effect is small enough to be declared equivalent to zero within a
pre-specified bound (SESOI). Reports SESOI = +/-5 points (primary, judged the
smallest practically meaningful day-level shift) and a tighter +/-3 points.

Run from repo root:
    python src/analysis/10_equivalence_test.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"

day  = pd.read_csv(OUT_DIR / "clean_day.csv")
ctrl = day.loc[day.condition == "control",   "mood"].dropna()
mel  = day.loc[day.condition == "melatonin", "mood"].dropna()

n1, n2 = len(ctrl), len(mel)
m1, m2 = ctrl.mean(), mel.mean()
v1, v2 = ctrl.var(ddof=1), mel.var(ddof=1)
diff   = m2 - m1                                  # melatonin - control
se     = np.sqrt(v1 / n1 + v2 / n2)              # Welch standard error
df     = (v1 / n1 + v2 / n2) ** 2 / (
         (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
tval   = diff / se
p_two  = 2 * stats.t.sf(abs(tval), df)
pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
d      = diff / pooled_sd

print("=" * 70)
print("BETWEEN-CONDITION MEAN TEST (daily mood, full 70 days)")
print("=" * 70)
print(f"control:   n={n1}, mean={m1:.2f}, sd={np.sqrt(v1):.2f}")
print(f"melatonin: n={n2}, mean={m2:.2f}, sd={np.sqrt(v2):.2f}")
print(f"diff (mel-ctrl) = {diff:.2f}, Welch t({df:.1f}) = {tval:.2f}, "
      f"p = {p_two:.3f}, d = {d:.2f}")

# The 90% CI of the difference is the interval TOST checks against the bounds.
tcrit = stats.t.ppf(0.95, df)
ci90_lo, ci90_hi = diff - tcrit * se, diff + tcrit * se
print(f"90% CI of difference = [{ci90_lo:.2f}, {ci90_hi:.2f}]")

rows = [
    ("desc.control_n",  n1,    "n control days"),
    ("desc.melatonin_n", n2,   "n melatonin days"),
    ("desc.diff",       diff,  "mean difference (melatonin - control), 0-100 mood"),
    ("desc.welch_t",    tval,  "Welch t for mean difference"),
    ("desc.welch_df",   df,    "Welch df"),
    ("desc.welch_p",    p_two, "two-sided p for mean difference"),
    ("desc.cohens_d",   d,     "Cohen's d (pooled)"),
    ("tost.ci90_lo",    ci90_lo, "lower 90% CI of mean difference"),
    ("tost.ci90_hi",    ci90_hi, "upper 90% CI of mean difference"),
]

print("\n" + "=" * 70)
print("TOST EQUIVALENCE")
print("=" * 70)
for delta in (5.0, 3.0):
    t_lo   = (diff + delta) / se          # H0: diff <= -delta  (upper-tail test)
    p_lo   = stats.t.sf(t_lo, df)
    t_hi   = (diff - delta) / se          # H0: diff >= +delta  (lower-tail test)
    p_hi   = stats.t.cdf(t_hi, df)
    p_tost = max(p_lo, p_hi)
    d_bound = delta / pooled_sd
    equiv  = p_tost < 0.05
    print(f"SESOI +/-{delta:g} (d={d_bound:.2f}): TOST p = {p_tost:.4f} -> "
          f"{'EQUIVALENT' if equiv else 'not equivalent'}")
    tag = f"{delta:g}".replace(".", "")
    rows += [
        (f"tost.sesoi{tag}.delta",      delta,      f"SESOI bound in points, +/-{delta:g}"),
        (f"tost.sesoi{tag}.d_bound",    d_bound,    f"SESOI as Cohen's d, +/-{delta:g}"),
        (f"tost.sesoi{tag}.p",          p_tost,     f"TOST p at +/-{delta:g}"),
        (f"tost.sesoi{tag}.equivalent", int(equiv), f"equivalence established at +/-{delta:g} (1/0)"),
    ]

pd.DataFrame(rows, columns=["key", "value", "description"]).to_csv(
    OUT_DIR / "equivalence_tost.csv", index=False)
print(f"\nSaved equivalence results to {OUT_DIR / 'equivalence_tost.csv'}")
