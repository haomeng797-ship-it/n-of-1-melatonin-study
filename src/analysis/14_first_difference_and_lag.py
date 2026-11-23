"""
14_first_difference_and_lag.py
Two checks added in the revision, both on day-level mood:

  (a) M2 in first differences (Day 18-70): day-to-day change in mood on the
      day-to-day changes in melatonin, agency and metacognition. Any slow
      change in level drops out of a differenced series.
  (b) M1 with a one-day lag of the melatonin indicator, as a check for
      carry-over from the previous assignment.
  (c) The melatonin coefficient, with its 90% interval, from a regression on
      the full 70-day series with lagged mood and a linear time trend, set
      against the equivalence bounds.
  (d) M2 without the Day 44 to Day 45 pair, whose lag spans two calendar days.

Writes outputs/first_difference_and_lag.csv.

Run from repo root:
    python src/analysis/14_first_difference_and_lag.py
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"

day = pd.read_csv(OUT_DIR / "clean_day.csv")
day["mood_lag1"] = day["mood"].shift(1)
day["melatonin_lag1"] = day["melatonin"].shift(1)
sub = day.dropna(subset=["mood", "mood_lag1", "agency", "metacognition"]).reset_index(drop=True)
rows = []

def fit(df, y, cols, label, alpha=0.05):
    m = sm.OLS(df[y], sm.add_constant(df[cols])).fit()
    ci = m.conf_int(alpha=alpha)
    for k in cols:
        rows.append(dict(model=label, term=k, estimate=m.params[k], ci_lo=ci.loc[k, 0],
                         ci_hi=ci.loc[k, 1], p=m.pvalues[k], r2=m.rsquared, n=int(m.nobs)))
    return m

# (a) first differences
d = sub[["study_day", "mood", "melatonin", "agency", "metacognition"]].copy()
for v in ["mood", "melatonin", "agency", "metacognition"]:
    d["d_" + v] = d[v].diff()
d = d.dropna(subset=["d_mood"])
fit(d, "d_mood", ["d_melatonin", "d_agency", "d_metacognition"], "M2_first_difference")

# (b) lagged melatonin
fit(sub, "mood", ["mood_lag1", "melatonin", "melatonin_lag1"], "M1_melatonin_lag")

# (c) model-based interval for the melatonin coefficient, full series (90% CI)
full = day.dropna(subset=["mood_lag1"]).copy()
full["time_c"] = full["study_day"] - full["study_day"].mean()
fit(full, "mood", ["mood_lag1", "melatonin", "time_c"], "full70_AR1_melatonin_time_ci90", alpha=0.10)

# (d) M2 without the pair that spans the 2026-04-03 gap
fit(sub[sub["study_day"] != 45], "mood", ["mood_lag1", "melatonin", "agency", "metacognition"],
    "M2_without_day44_45_pair")

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "first_difference_and_lag.csv", index=False)
print(out.round(4).to_string(index=False))
