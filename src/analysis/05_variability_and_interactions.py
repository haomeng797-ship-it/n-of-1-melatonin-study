"""
05_variability_and_interactions.py
(A) Affective variability by condition: SD, MSSD, Levene, and an AR(1)
    with heteroskedastic residuals.
(B) Agency x melatonin interaction (robustness check).
(C) State-dependent phi: mood_lag1 x melatonin (sensitivity, not headline).

Run from repo root:
    python src/analysis/05_variability_and_interactions.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"

day = pd.read_csv(OUT_DIR / "clean_day.csv")
obs = pd.read_csv(OUT_DIR / "clean_obs.csv")

print("=" * 70)
print("(A) AFFECTIVE VARIABILITY BY CONDITION")
print("=" * 70)

mel_day  = day.loc[day.condition == "melatonin", "mood"].dropna()
ctrl_day = day.loc[day.condition == "control",   "mood"].dropna()
print(f"\nDay-level SD: control={ctrl_day.std():.3f}, melatonin={mel_day.std():.3f}")
lev_W, lev_p = stats.levene(mel_day, ctrl_day, center="median")
print(f"  Brown-Forsythe Levene: W = {lev_W:.3f}, p = {lev_p:.3f}")

mel_obs  = obs.loc[obs.condition == "melatonin", "mood"].dropna()
ctrl_obs = obs.loc[obs.condition == "control",   "mood"].dropna()
print(f"\nObs-level SD: control={ctrl_obs.std():.3f}, melatonin={mel_obs.std():.3f}")
lev2_W, lev2_p = stats.levene(mel_obs, ctrl_obs, center="median")
print(f"  Brown-Forsythe Levene: W = {lev2_W:.3f}, p = {lev2_p:.3f}")

# MSSD within same-condition runs
def cond_runs_mssd(df, cond):
    runs, cur = [], []
    for _, row in df.iterrows():
        if row["condition"] == cond:
            cur.append(row["mood"])
        else:
            if len(cur) >= 2:
                runs.append(np.mean(np.diff(cur) ** 2))
            cur = []
    if len(cur) >= 2:
        runs.append(np.mean(np.diff(cur) ** 2))
    return runs

obs_sorted = obs.copy()
obs_sorted["datetime"] = pd.to_datetime(obs_sorted["datetime"])
obs_sorted = obs_sorted.sort_values("datetime").reset_index(drop=True)
ctrl_runs = cond_runs_mssd(obs_sorted, "control")
mel_runs  = cond_runs_mssd(obs_sorted, "melatonin")
print(f"\nMSSD (within-run, mean): control={np.mean(ctrl_runs):.2f}, melatonin={np.mean(mel_runs):.2f}")
mssd_t, mssd_p = stats.ttest_ind(mel_runs, ctrl_runs, equal_var=False)
print(f"  Welch t on run-MSSD: t = {mssd_t:.3f}, p = {mssd_p:.3f}")

# Heteroskedastic regression on M2 residuals
day["mood_lag1"] = day["mood"].shift(1)
sub = day.dropna(subset=["mood", "mood_lag1", "agency", "metacognition", "melatonin"]).copy()
y, X = sub["mood"].values, sm.add_constant(sub[["mood_lag1", "melatonin", "agency", "metacognition"]])
m_full = sm.OLS(y, X).fit()
sub["log_sq_resid"] = np.log(m_full.resid ** 2 + 1e-6)
het = sm.OLS(sub["log_sq_resid"], sm.add_constant(sub[["melatonin"]])).fit()
print("\nLog-variance regression on melatonin indicator:")
print(f"  alpha_1 = {het.params['melatonin']:.3f}, 95% CI "
      f"[{het.conf_int().loc['melatonin',0]:.3f}, {het.conf_int().loc['melatonin',1]:.3f}], "
      f"p = {het.pvalues['melatonin']:.3f}")

print("\n" + "=" * 70)
print("(B) AGENCY x MELATONIN INTERACTION")
print("=" * 70)
sub["agency_c"] = sub["agency"] - sub["agency"].mean()
sub["meta_c"]   = sub["metacognition"] - sub["metacognition"].mean()
sub["ag_x_mel"] = sub["agency_c"] * sub["melatonin"]
X_int = sm.add_constant(sub[["mood_lag1", "agency_c", "meta_c", "melatonin", "ag_x_mel"]])
m_int = sm.OLS(sub["mood"].values, X_int).fit()
ci = m_int.conf_int(alpha=0.05)
print(f"n = {int(m_int.nobs)}, R^2 = {m_int.rsquared:.3f}")
print(f"agency × melatonin: beta = {m_int.params['ag_x_mel']:.3f}, 95% CI "
      f"[{ci.loc['ag_x_mel',0]:.3f}, {ci.loc['ag_x_mel',1]:.3f}], "
      f"p = {m_int.pvalues['ag_x_mel']:.3f}")

print("\n" + "=" * 70)
print("(C) STATE-DEPENDENT PHI (sensitivity)")
print("=" * 70)
sub["lag1_x_mel"] = sub["mood_lag1"] * sub["melatonin"]
X_sd = sm.add_constant(sub[["mood_lag1", "melatonin", "lag1_x_mel"]])
m_sd = sm.OLS(sub["mood"].values, X_sd).fit()
ci2 = m_sd.conf_int(alpha=0.05)
phi_ctrl = m_sd.params["mood_lag1"]
phi_mel  = m_sd.params["mood_lag1"] + m_sd.params["lag1_x_mel"]
print(f"phi (control)   = {phi_ctrl:.3f}")
print(f"phi (melatonin) = {phi_mel:.3f}")
print(f"interaction beta = {m_sd.params['lag1_x_mel']:.3f}, 95% CI "
      f"[{ci2.loc['lag1_x_mel',0]:.3f}, {ci2.loc['lag1_x_mel',1]:.3f}], "
      f"p = {m_sd.pvalues['lag1_x_mel']:.3f}")

# ---- Persist the headline numbers so they are reproducible from outputs/ ----
rows = [
    ("var.day_sd_control",        ctrl_day.std(),                      "Day-level SD of daily mood (control)"),
    ("var.day_sd_melatonin",      mel_day.std(),                       "Day-level SD of daily mood (melatonin)"),
    ("var.day_levene_W",          lev_W,                               "Day-level Brown-Forsythe Levene W"),
    ("var.day_levene_p",          lev_p,                               "Day-level Brown-Forsythe Levene p"),
    ("var.obs_sd_control",        ctrl_obs.std(),                      "Obs-level SD of mood (control)"),
    ("var.obs_sd_melatonin",      mel_obs.std(),                       "Obs-level SD of mood (melatonin)"),
    ("var.obs_levene_W",          lev2_W,                              "Obs-level Brown-Forsythe Levene W"),
    ("var.obs_levene_p",          lev2_p,                              "Obs-level Brown-Forsythe Levene p"),
    ("var.mssd_control_mean",     float(np.mean(ctrl_runs)),           "Mean MSSD across control runs"),
    ("var.mssd_melatonin_mean",   float(np.mean(mel_runs)),            "Mean MSSD across melatonin runs"),
    ("var.mssd_welch_t",          mssd_t,                              "Welch t for MSSD (mel vs control)"),
    ("var.mssd_welch_p",          mssd_p,                              "Welch p for MSSD"),
    ("var.het_alpha1",            het.params["melatonin"],             "Log-variance regression: melatonin coef"),
    ("var.het_alpha1_ci_lo",      het.conf_int().loc["melatonin", 0],  "95% CI lower for log-variance coef"),
    ("var.het_alpha1_ci_hi",      het.conf_int().loc["melatonin", 1],  "95% CI upper for log-variance coef"),
    ("var.het_alpha1_p",          het.pvalues["melatonin"],            "p for log-variance coef"),
    ("interaction.ag_x_mel.beta", m_int.params["ag_x_mel"],            "agency_centered x melatonin interaction beta"),
    ("interaction.ag_x_mel.ci_lo", ci.loc["ag_x_mel", 0],              "95% CI lower for agency x mel interaction"),
    ("interaction.ag_x_mel.ci_hi", ci.loc["ag_x_mel", 1],              "95% CI upper for agency x mel interaction"),
    ("interaction.ag_x_mel.p",    m_int.pvalues["ag_x_mel"],           "p for agency x mel interaction"),
    ("statedep.phi_control",      phi_ctrl,                            "AR(1) phi for control days"),
    ("statedep.phi_melatonin",    phi_mel,                             "AR(1) phi for melatonin days"),
    ("statedep.lag_x_mel.beta",   m_sd.params["lag1_x_mel"],           "lag1 x melatonin interaction beta"),
    ("statedep.lag_x_mel.ci_lo",  ci2.loc["lag1_x_mel", 0],            "95% CI lower for lag1 x mel interaction"),
    ("statedep.lag_x_mel.ci_hi",  ci2.loc["lag1_x_mel", 1],            "95% CI upper for lag1 x mel interaction"),
    ("statedep.lag_x_mel.p",      m_sd.pvalues["lag1_x_mel"],          "p for lag1 x mel interaction"),
]
pd.DataFrame(rows, columns=["key", "value", "description"]).to_csv(
    OUT_DIR / "variability_table.csv", index=False)
print(f"\nSaved variability/interaction results to {OUT_DIR / 'variability_table.csv'}")
