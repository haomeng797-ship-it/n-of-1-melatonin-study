"""
15_lagged_model_and_split.py
Two checks on the Day 18-70 window, reported in Appendix C of the manuscript:

  (a) Lagged model of the three daily series: mood, agency and metacognition,
      each regressed on the previous day's values of all three (a first-order
      vector autoregression fit equation by equation), plus the same check on
      consecutive prompts for the mood equation.
  (b) M2 refit on the days before and after the interim analysis (Day 44),
      with a test of whether the coefficients differ, a second split at
      Day 40, and the melatonin-minus-control difference in each part.

Also counts missed prompts by condition.

Writes outputs/lagged_model.csv and outputs/split_at_interim.csv.

Run from repo root:
    python src/analysis/15_lagged_model_and_split.py
"""
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"
V = ["mood", "agency", "metacognition"]
L = [v + "_lag1" for v in V]

day = pd.read_csv(OUT_DIR / "clean_day.csv")
obs = pd.read_csv(OUT_DIR / "clean_obs.csv", parse_dates=["datetime"])

def rows_from(m, label, terms):
    ci = m.conf_int()
    return [dict(model=label, term=k, estimate=m.params[k], ci_lo=ci.loc[k, 0], ci_hi=ci.loc[k, 1],
                 p=m.pvalues[k], r2=m.rsquared, n=int(m.nobs)) for k in terms]

# ---------- (a) lagged model ----------
d = day.copy()
for v in V:
    d[v + "_lag1"] = d[v].shift(1)
var = d.dropna(subset=V + L)
out_a, resid = [], {}
for v in V:
    m = sm.OLS(var[v], sm.add_constant(var[L])).fit()
    out_a += rows_from(m, f"day_VAR1_{v}", L)
    resid[v] = m.resid
    if v == "mood":
        f = m.f_test("agency_lag1 = 0, metacognition_lag1 = 0")
        out_a.append(dict(model="day_VAR1_mood", term="joint_F_internal_lags", estimate=float(f.fvalue),
                          p=float(f.pvalue), n=int(m.nobs)))
rc = pd.DataFrame(resid).corr()
out_a.append(dict(model="day_VAR1_residual_corr", term="mood~agency", estimate=rc.loc["mood", "agency"]))
out_a.append(dict(model="day_VAR1_residual_corr", term="mood~metacognition", estimate=rc.loc["mood", "metacognition"]))

o = obs.dropna(subset=["agency"]).sort_values("datetime").reset_index(drop=True)
for v in V:
    o[v + "_lag1"] = o[v].shift(1)
o = o.dropna(subset=L)
out_a += rows_from(sm.OLS(o["mood"], sm.add_constant(o[L])).fit(), "prompt_lag1_mood", L)
pd.DataFrame(out_a).to_csv(OUT_DIR / "lagged_model.csv", index=False)

# ---------- (b) split at the interim analysis ----------
d["mood_lag1"] = d["mood"].shift(1)
sub = d.dropna(subset=["mood_lag1", "agency", "metacognition"])
X = ["mood_lag1", "melatonin", "agency", "metacognition"]
out_b = []
for label, lo, hi in [("days_18_44", 18, 44), ("days_45_70", 45, 70), ("days_18_40", 18, 40), ("days_41_70", 41, 70)]:
    s = sub[(sub.study_day >= lo) & (sub.study_day <= hi)]
    m = sm.OLS(s["mood"], sm.add_constant(s[X])).fit()
    out_b += rows_from(m, f"M2_{label}", ["agency", "metacognition", "melatonin"])
    out_b.append(dict(model=f"M2_{label}", term="mood_sd", estimate=s["mood"].std(), n=len(s)))
for cut in (45, 41):
    s = sub.assign(late=(sub.study_day >= cut).astype(int))
    full = smf.ols("mood ~ (mood_lag1 + melatonin + agency + metacognition) * late", s).fit()
    base = smf.ols("mood ~ mood_lag1 + melatonin + agency + metacognition", s).fit()
    a = anova_lm(base, full)
    out_b.append(dict(model=f"coefficients_differ_cut_day{cut}", term="F", estimate=a["F"].iloc[1],
                      p=a["Pr(>F)"].iloc[1], n=int(full.nobs)))
for label, lo, hi in [("days_1_44", 1, 44), ("days_45_70", 45, 70)]:
    s = day[(day.study_day >= lo) & (day.study_day <= hi)]
    c, t = s[s.melatonin == 0].mood, s[s.melatonin == 1].mood
    w = stats.ttest_ind(t, c, equal_var=False)
    out_b.append(dict(model=f"melatonin_minus_control_{label}", term="difference", estimate=t.mean() - c.mean(),
                      p=w.pvalue, n=len(s)))
missed = (3 - day.groupby("condition")["n_obs"].sum() / day.groupby("condition")["n_obs"].count()) \
    * day.groupby("condition")["n_obs"].count()
for cond, k in missed.items():
    out_b.append(dict(model="missed_prompts", term=cond, estimate=k))
pd.DataFrame(out_b).to_csv(OUT_DIR / "split_at_interim.csv", index=False)

print(pd.DataFrame(out_a).round(3).to_string(index=False))
print(pd.DataFrame(out_b).round(3).to_string(index=False))
