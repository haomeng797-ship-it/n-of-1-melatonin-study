"""
02_ar1_models.py
Three nested AR(1)-with-exogenous-inputs specifications, fit on day-level mood:

  M0:  mood_t = phi*mood_{t-1}                                  [baseline]
  M1:  + b1 * melatonin_t                                        [+external]
  M2:  + b2 * agency_t + b3 * metacognition_t                    [+internal]

All three are fit on the Day 18-70 subsample (n=53) so they share data
and incremental R^2 is comparable. Writes coefficient tables and delta-R^2
to outputs/.

Run from repo root:
    python src/analysis/02_ar1_models.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"

day = pd.read_csv(OUT_DIR / "clean_day.csv")
day["mood_lag1"] = day["mood"].shift(1)
day["mood_lag2"] = day["mood"].shift(2)

sub = day.dropna(subset=["mood", "mood_lag1", "agency", "metacognition", "melatonin"]).copy()
sub = sub.reset_index(drop=True)
print(f"Nested-model subsample n = {len(sub)} days "
      f"(study_day {int(sub['study_day'].min())} - {int(sub['study_day'].max())})")

def fit(y, X, label):
    X = sm.add_constant(X)
    m = sm.OLS(y, X).fit()
    ci = m.conf_int(alpha=0.05)
    out = pd.DataFrame({
        "model": label, "term": m.params.index,
        "estimate": m.params.values, "std_err": m.bse.values,
        "t": m.tvalues, "p": m.pvalues,
        "ci_lo": ci[0].values, "ci_hi": ci[1].values,
    })
    return m, out

y = sub["mood"].values
m0, t0 = fit(y, sub[["mood_lag1"]], "M0_AR1")
m1, t1 = fit(y, sub[["mood_lag1", "melatonin"]], "M1_AR1+mel")
m2, t2 = fit(y, sub[["mood_lag1", "melatonin", "agency", "metacognition"]], "M2_full")

# Incremental R^2 via drop-one
def _drop1_R2(full_model, full_X_cols, drop_col):
    keep = [c for c in full_X_cols if c != drop_col]
    sub_m = sm.OLS(y, sm.add_constant(sub[keep])).fit()
    return full_model.rsquared - sub_m.rsquared

dR2 = {
    "melatonin":     _drop1_R2(m2, ["mood_lag1","melatonin","agency","metacognition"], "melatonin"),
    "agency":        _drop1_R2(m2, ["mood_lag1","melatonin","agency","metacognition"], "agency"),
    "metacognition": _drop1_R2(m2, ["mood_lag1","melatonin","agency","metacognition"], "metacognition"),
    "internal_block (agency + metacognition)": m2.rsquared - m1.rsquared,
}

# Save
pd.concat([t0, t1, t2], ignore_index=True).to_csv(OUT_DIR / "ar1_coefficients.csv", index=False)
pd.DataFrame({
    "model":  ["M0_AR1", "M1_AR1+mel", "M2_full"],
    "n":      [int(m0.nobs)]*3,
    "R2":     [m0.rsquared, m1.rsquared, m2.rsquared],
    "adj_R2": [m0.rsquared_adj, m1.rsquared_adj, m2.rsquared_adj],
}).to_csv(OUT_DIR / "ar1_fit_table.csv", index=False)
pd.DataFrame([{"predictor": k, "deltaR2": v} for k, v in dR2.items()]).to_csv(OUT_DIR / "delta_r2.csv", index=False)

print("\n=== M0 baseline AR(1) ===")
print(t0.to_string(index=False))
print(f"R^2 = {m0.rsquared:.3f}")

print("\n=== M1 + melatonin ===")
print(t1.to_string(index=False))
print(f"R^2 = {m1.rsquared:.3f}")

print("\n=== M2 full ===")
print(t2.to_string(index=False))
print(f"R^2 = {m2.rsquared:.3f}, adj R^2 = {m2.rsquared_adj:.3f}")

print("\n=== Incremental Delta R^2 (drop-one within M2) ===")
for k, v in dR2.items():
    print(f"  {k:48s} {v:.4f}")

# ---------------------------------------------------------------------------
# Lag-order sensitivity on the SAME Day 18-70 window as M0-M2.
#
# The series-level check in 11_ar_order_check.R answers a different question,
# namely which lag order describes the whole 70-day mood series. The primary
# models are estimated on Day 18-70, so the lag order that justifies them has
# to be evaluated on that window too. Every fit below uses the same OLS
# machinery and the same 53 rows, so the AIC/BIC values are comparable to each
# other and to M0-M2 above. The second lag is available for the whole window
# because mood is observed on all 70 days (Day 18 takes Day 16 as its lag-2).
# ---------------------------------------------------------------------------
assert sub["mood_lag2"].notna().all(), "mood_lag2 missing inside the Day 18-70 window"

m0_ar2, t0b = fit(y, sub[["mood_lag1", "mood_lag2"]], "M0_AR2")
m2_ar2, t2b = fit(
    y, sub[["mood_lag1", "mood_lag2", "melatonin", "agency", "metacognition"]], "M2_AR2"
)

pd.concat([t0b, t2b], ignore_index=True).to_csv(
    OUT_DIR / "ar_order_same_window.csv", index=False)

pd.DataFrame({
    "model": ["M0_AR1", "M0_AR2", "M2_AR1", "M2_AR2"],
    "n":     [int(m.nobs)     for m in (m0, m0_ar2, m2, m2_ar2)],
    "R2":    [m.rsquared      for m in (m0, m0_ar2, m2, m2_ar2)],
    "adj_R2":[m.rsquared_adj  for m in (m0, m0_ar2, m2, m2_ar2)],
    "aic":   [m.aic           for m in (m0, m0_ar2, m2, m2_ar2)],
    "bic":   [m.bic           for m in (m0, m0_ar2, m2, m2_ar2)],
}).to_csv(OUT_DIR / "ar_order_fit_comparison.csv", index=False)

print("\n=== Lag-order sensitivity, same Day 18-70 window ===")
print("Outcome-only AR(2):")
print(t0b.to_string(index=False))
print("\nM2 + second lag:")
print(t2b.to_string(index=False))
print(f"\nM2   AR(1): AIC = {m2.aic:.2f}, BIC = {m2.bic:.2f}, R2 = {m2.rsquared:.3f}")
print(f"M2   AR(2): AIC = {m2_ar2.aic:.2f}, BIC = {m2_ar2.bic:.2f}, R2 = {m2_ar2.rsquared:.3f}")

# Residual diagnostics and collinearity, persisted so that every diagnostic
# quoted in Section 3.2 and the Table 1 note can be checked from outputs/.
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
lb = acorr_ljungbox(m2.resid, lags=[1, 5, 10], return_df=True)
sw_W, sw_p = stats.shapiro(m2.resid)
print(f"\nM2 residuals — Ljung-Box p (lags 1,5,10): {lb['lb_pvalue'].values}")
print(f"M2 residuals — Shapiro-Wilk W = {sw_W:.3f}, p = {sw_p:.3f}")

M2_X = ["mood_lag1", "melatonin", "agency", "metacognition"]
_Xc = sm.add_constant(sub[M2_X])
vifs = {c: variance_inflation_factor(_Xc.values, i)
        for i, c in enumerate(_Xc.columns) if c != "const"}
_Z = (sub[M2_X] - sub[M2_X].mean()) / sub[M2_X].std()
cond_number = float(np.linalg.cond(_Z.values))

diag = [("resid.ljungbox_p_lag1",  float(lb["lb_pvalue"].iloc[0]), "M2 residuals, Ljung-Box p at lag 1"),
        ("resid.ljungbox_p_lag5",  float(lb["lb_pvalue"].iloc[1]), "M2 residuals, Ljung-Box p at lag 5"),
        ("resid.ljungbox_p_lag10", float(lb["lb_pvalue"].iloc[2]), "M2 residuals, Ljung-Box p at lag 10"),
        ("resid.shapiro_W", float(sw_W), "M2 residuals, Shapiro-Wilk W"),
        ("resid.shapiro_p", float(sw_p), "M2 residuals, Shapiro-Wilk p"),
        ("collin.condition_number", cond_number,
         "Condition number of the standardized M2 predictor matrix")]
diag += [(f"collin.vif_{k}", float(v), f"Variance inflation factor for {k} in M2")
         for k, v in vifs.items()]
pd.DataFrame(diag, columns=["key", "value", "description"]).to_csv(
    OUT_DIR / "m2_diagnostics.csv", index=False)

print("\nM2 collinearity — VIF: "
      + ", ".join(f"{k} {v:.3f}" for k, v in vifs.items())
      + f" | condition number {cond_number:.3f}")
