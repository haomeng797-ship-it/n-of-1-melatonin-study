"""
08_results_table.py
Assemble outputs/results_table.csv — the single master table holding every
number reported in the manuscript — from the cleaned data and the per-script
output CSVs. This makes the master table fully reproducible (previously it was
a hand-maintained artifact not produced by any script).

Run from repo root, LAST, after 01-05:
    python src/analysis/08_results_table.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"

day  = pd.read_csv(OUT_DIR / "clean_day.csv")
obs  = pd.read_csv(OUT_DIR / "clean_obs.csv")
fit  = pd.read_csv(OUT_DIR / "ar1_fit_table.csv").set_index("model")
dR2  = pd.read_csv(OUT_DIR / "delta_r2.csv").set_index("predictor")["deltaR2"]
coef = pd.read_csv(OUT_DIR / "ar1_coefficients.csv")
phl  = pd.read_csv(OUT_DIR / "phi_halflife.csv").set_index("level")
kal  = pd.read_csv(OUT_DIR / "kalman_fit_summary.csv").iloc[0]
meta = pd.read_csv(OUT_DIR / "metacontrol_table.csv").set_index("term")
var  = pd.read_csv(OUT_DIR / "variability_table.csv").set_index("key")["value"]

m2 = coef[coef["model"] == "M2_full"].set_index("term")
m0_phi = float(coef[(coef.model == "M0_AR1") & (coef.term == "mood_lag1")]["estimate"].iloc[0])

ctrl = day[day.condition == "control"]
mel  = day[day.condition == "melatonin"]

# Primary between-condition contrast on daily mood (Welch t + pooled-SD Cohen's d)
delta   = mel["mood"].mean() - ctrl["mood"].mean()
welch   = stats.ttest_ind(mel["mood"], ctrl["mood"], equal_var=False)
pooled  = np.sqrt((ctrl["mood"].std(ddof=1) ** 2 + mel["mood"].std(ddof=1) ** 2) / 2)
cohens_d = delta / pooled

half_obs_hours = float(str(phl.loc["obs", "physical_unit"]).split()[0])

def ci(term, frame):
    return f"[{frame.loc[term, 'ci_lo']:.3f}, {frame.loc[term, 'ci_hi']:.3f}]"

excluded = pd.read_csv(OUT_DIR / "excluded_observations.csv")

rows = [
    ("compliance.raw_log_records", len(obs) + len(excluded),
     "Records in the archived raw EMA log"),
    ("compliance.excluded_records", len(excluded),
     "Unscheduled records excluded from analysis"),
    ("compliance.planned_obs", 3 * len(day), "Planned EMA pings (70 days x 3)"),
    ("compliance.analytic_obs", len(obs), "Scheduled observations analysed"),
    ("compliance.rate_pct", round(len(obs) / (3 * len(day)) * 100, 2),
     "Scheduled-response compliance (%)"),
    ("compliance.days_covered", len(day), "Study days with at least one mood ping"),
    ("compliance.days_with_agency_metacog", int(day["agency"].notna().sum()),
     "Days with agency + metacog (added Day 18)"),

    ("desc.control.n_days", len(ctrl), "control: number of days"),
    ("desc.control.mood.mean", round(ctrl["mood"].mean(), 2), "control: mean of daily mood"),
    ("desc.control.mood.sd", round(ctrl["mood"].std(), 2), "control: SD of daily mood"),
    ("desc.control.agency.mean", round(ctrl["agency"].mean(), 2), "control: mean of daily agency"),
    ("desc.control.agency.sd", round(ctrl["agency"].std(), 2), "control: SD of daily agency"),
    ("desc.control.metacognition.mean", round(ctrl["metacognition"].mean(), 2), "control: mean of daily metacognition"),
    ("desc.control.metacognition.sd", round(ctrl["metacognition"].std(), 2), "control: SD of daily metacognition"),
    ("desc.melatonin.n_days", len(mel), "melatonin: number of days"),
    ("desc.melatonin.mood.mean", round(mel["mood"].mean(), 2), "melatonin: mean of daily mood"),
    ("desc.melatonin.mood.sd", round(mel["mood"].std(), 2), "melatonin: SD of daily mood"),
    ("desc.melatonin.agency.mean", round(mel["agency"].mean(), 2), "melatonin: mean of daily agency"),
    ("desc.melatonin.agency.sd", round(mel["agency"].std(), 2), "melatonin: SD of daily agency"),
    ("desc.melatonin.metacognition.mean", round(mel["metacognition"].mean(), 2), "melatonin: mean of daily metacognition"),
    ("desc.melatonin.metacognition.sd", round(mel["metacognition"].std(), 2), "melatonin: SD of daily metacognition"),

    ("primary.delta_daily_mood", round(delta, 2), "Daily-mood difference: melatonin minus control"),
    ("primary.welch_t", round(welch.statistic, 3), "Welch t-statistic for daily mood"),
    ("primary.welch_p", round(welch.pvalue, 3), "Welch p-value (two-sided)"),
    ("primary.cohens_d", round(cohens_d, 3), "Cohen's d for daily-mood between-condition contrast"),

    ("ar1.phi_obs", round(phl.loc["obs", "phi"], 3), "Obs-level AR(1) phi (between EMA pings, ~6h apart)"),
    ("ar1.phi_day", round(m0_phi, 3), "Day-level AR(1) phi (M0 baseline, Day 18-70 subsample)"),
    ("ar1.halflife_obs_steps", round(phl.loc["obs", "half_life_steps"], 2), "Obs-level half-life (in EMA steps)"),
    ("ar1.halflife_obs_hours", round(half_obs_hours, 1), "Obs-level half-life (hours)"),
    ("ar1.halflife_day", round(np.log(0.5) / np.log(m0_phi), 2), "Day-level half-life (days, from M0 phi)"),

    ("M2.mood_lag1.beta", round(m2.loc["mood_lag1", "estimate"], 3), "M2 beta for mood_lag1"),
    ("M2.mood_lag1.ci", ci("mood_lag1", m2), "M2 95% CI for mood_lag1"),
    ("M2.mood_lag1.p", round(m2.loc["mood_lag1", "p"], 4), "M2 p for mood_lag1"),
    ("M2.melatonin.beta", round(m2.loc["melatonin", "estimate"], 3), "M2 beta for melatonin"),
    ("M2.melatonin.ci", ci("melatonin", m2), "M2 95% CI for melatonin"),
    ("M2.melatonin.p", round(m2.loc["melatonin", "p"], 4), "M2 p for melatonin"),
    ("M2.agency.beta", round(m2.loc["agency", "estimate"], 3), "M2 beta for agency"),
    ("M2.agency.ci", ci("agency", m2), "M2 95% CI for agency"),
    ("M2.agency.p", round(m2.loc["agency", "p"], 4), "M2 p for agency"),
    ("M2.metacognition.beta", round(m2.loc["metacognition", "estimate"], 3), "M2 beta for metacognition"),
    ("M2.metacognition.ci", ci("metacognition", m2), "M2 95% CI for metacognition"),
    ("M2.metacognition.p", round(m2.loc["metacognition", "p"], 4), "M2 p for metacognition"),

    ("fit.M0_AR1.R2", round(fit.loc["M0_AR1", "R2"], 3), "R^2 for M0_AR1"),
    ("fit.M0_AR1.adj_R2", round(fit.loc["M0_AR1", "adj_R2"], 3), "Adjusted R^2 for M0_AR1"),
    ("fit.M1_AR1+mel.R2", round(fit.loc["M1_AR1+mel", "R2"], 3), "R^2 for M1_AR1+mel"),
    ("fit.M1_AR1+mel.adj_R2", round(fit.loc["M1_AR1+mel", "adj_R2"], 3), "Adjusted R^2 for M1_AR1+mel"),
    ("fit.M2_full.R2", round(fit.loc["M2_full", "R2"], 3), "R^2 for M2_full"),
    ("fit.M2_full.adj_R2", round(fit.loc["M2_full", "adj_R2"], 3), "Adjusted R^2 for M2_full"),

    ("deltaR2.melatonin", round(dR2["melatonin"], 4), "Incremental R^2: melatonin (drop-one)"),
    ("deltaR2.agency", round(dR2["agency"], 4), "Incremental R^2: agency (drop-one)"),
    ("deltaR2.metacognition", round(dR2["metacognition"], 4), "Incremental R^2: metacognition (drop-one)"),
    ("deltaR2.internal", round(dR2["internal_block (agency + metacognition)"], 4),
     "Incremental R^2: internal block (agency+metacog joint)"),

    ("kalman.sigma_v2", round(kal["sigma_v2"], 3), "Local-level state-space: sigma_v2"),
    ("kalman.sigma_w2", round(kal["sigma_w2"], 3), "Local-level state-space: sigma_w2"),
    ("kalman.snr", round(kal["snr"], 3), "Local-level state-space: snr"),
    ("kalman.loglik", round(kal["loglik"], 3), "Local-level state-space: loglik"),
    ("kalman.AIC", round(kal["AIC"], 3), "Local-level state-space: AIC"),

    ("metacontrol.beta", round(meta.loc["mel_x_meta", "estimate"], 3), "Metacontrol interaction (melatonin x metacog_centered)"),
    ("metacontrol.ci", f"[{meta.loc['mel_x_meta', 'ci_lo']:.3f}, {meta.loc['mel_x_meta', 'ci_hi']:.3f}]",
     "95% CI for metacontrol interaction"),
    ("metacontrol.p", round(meta.loc["mel_x_meta", "p"], 3), "p-value for metacontrol interaction"),

    ("var.day_sd_control", round(var["var.day_sd_control"], 3), "Day-level SD of daily mood (control)"),
    ("var.day_sd_melatonin", round(var["var.day_sd_melatonin"], 3), "Day-level SD of daily mood (melatonin)"),
    ("var.day_levene_W", round(var["var.day_levene_W"], 3), "Day-level Brown-Forsythe Levene W"),
    ("var.day_levene_p", round(var["var.day_levene_p"], 3), "Day-level Brown-Forsythe Levene p"),
    ("var.obs_sd_control", round(var["var.obs_sd_control"], 3), "Obs-level SD of mood (control)"),
    ("var.obs_sd_melatonin", round(var["var.obs_sd_melatonin"], 3), "Obs-level SD of mood (melatonin)"),
    ("var.obs_levene_W", round(var["var.obs_levene_W"], 3), "Obs-level Brown-Forsythe Levene W"),
    ("var.obs_levene_p", round(var["var.obs_levene_p"], 3), "Obs-level Brown-Forsythe Levene p"),
    ("var.mssd_control_mean", round(var["var.mssd_control_mean"], 2), "Mean MSSD across control runs"),
    ("var.mssd_melatonin_mean", round(var["var.mssd_melatonin_mean"], 2), "Mean MSSD across melatonin runs"),
    ("var.mssd_welch_t", round(var["var.mssd_welch_t"], 3), "Welch t for MSSD (mel vs control)"),
    ("var.mssd_welch_p", round(var["var.mssd_welch_p"], 3), "Welch p for MSSD"),
    ("var.het_alpha1", round(var["var.het_alpha1"], 3), "Log-variance regression: melatonin coef (negative = mel reduces resid var)"),
    ("var.het_alpha1_ci", f"[{var['var.het_alpha1_ci_lo']:.3f}, {var['var.het_alpha1_ci_hi']:.3f}]",
     "95% CI for log-variance coef"),
    ("var.het_alpha1_p", round(var["var.het_alpha1_p"], 3), "p for log-variance coef"),

    ("interaction.ag_x_mel.beta", round(var["interaction.ag_x_mel.beta"], 3), "agency_centered x melatonin interaction beta"),
    ("interaction.ag_x_mel.ci", f"[{var['interaction.ag_x_mel.ci_lo']:.3f}, {var['interaction.ag_x_mel.ci_hi']:.3f}]",
     "95% CI for agency x mel interaction"),
    ("interaction.ag_x_mel.p", round(var["interaction.ag_x_mel.p"], 3), "p for agency x mel interaction"),

    ("statedep.phi_control", round(var["statedep.phi_control"], 3), "AR(1) phi for control days"),
    ("statedep.phi_melatonin", round(var["statedep.phi_melatonin"], 3), "AR(1) phi for melatonin days"),
    ("statedep.lag_x_mel.beta", round(var["statedep.lag_x_mel.beta"], 3), "lag1 x melatonin interaction beta"),
    ("statedep.lag_x_mel.p", round(var["statedep.lag_x_mel.p"], 3), "p for lag1 x mel interaction"),
]

pd.DataFrame(rows, columns=["key", "value", "description"]).to_csv(
    OUT_DIR / "results_table.csv", index=False)
print(f"Wrote {len(rows)} rows to {OUT_DIR / 'results_table.csv'}")
