"""
12_time_trend_sensitivity.py
Sensitivity analysis: refit M0-M2 (Day 18-70) and the full-sample daily AR(1)
with a linear term for study day (centered). Writes outputs/time_trend_sensitivity.csv.

Run from repo root:
    python src/analysis/12_time_trend_sensitivity.py
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"

day = pd.read_csv(OUT_DIR / "clean_day.csv")
day["mood_lag1"] = day["mood"].shift(1)
sub = day.dropna(subset=["mood", "mood_lag1", "agency", "metacognition"]).reset_index(drop=True)
sub["time_c"] = sub["study_day"] - sub["study_day"].mean()
rows = []

def fit(df, y, cols, label):
    m = sm.OLS(df[y], sm.add_constant(df[cols])).fit()
    ci = m.conf_int()
    for k in cols:
        rows.append(dict(model=label, term=k, estimate=m.params[k], ci_lo=ci.loc[k, 0],
                         ci_hi=ci.loc[k, 1], p=m.pvalues[k], r2=m.rsquared, n=int(m.nobs)))
    return m

base = ["mood_lag1", "melatonin", "agency", "metacognition"]
m1 = fit(sub, "mood", base[:2], "M1")
m2 = fit(sub, "mood", base, "M2")
fit(sub, "mood", ["mood_lag1", "time_c"], "M0_time")
m1t = fit(sub, "mood", base[:2] + ["time_c"], "M1_time")
m2t = fit(sub, "mood", base + ["time_c"], "M2_time")
rows.append(dict(model="internal_block_dR2", term="no_time", estimate=m2.rsquared - m1.rsquared))
rows.append(dict(model="internal_block_dR2", term="with_time", estimate=m2t.rsquared - m1t.rsquared))
for v in ["mood", "agency", "metacognition"]:
    fit(sub, v, ["time_c"], f"trend_{v}_day18_70")

full = day.dropna(subset=["mood_lag1"]).copy()
full["time_c"] = full["study_day"] - full["study_day"].mean()
fit(full, "mood", ["mood_lag1"], "full70_AR1")
fit(full, "mood", ["mood_lag1", "time_c"], "full70_AR1_time")

out = pd.DataFrame(rows)
out.to_csv(OUT_DIR / "time_trend_sensitivity.csv", index=False)
print(out.round(4).to_string(index=False))
