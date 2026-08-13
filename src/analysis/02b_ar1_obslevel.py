"""
02b_ar1_obslevel.py
Obs-level AR(1) on momentary mood (scheduled EMA pings, timestamp-ordered).
Also computes the full-sample day-level AR(1) coefficient (n = 70).

Run from repo root:
    python src/analysis/02b_ar1_obslevel.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"

# Obs-level AR(1)
obs = pd.read_csv(OUT_DIR / "clean_obs.csv")
obs["datetime"] = pd.to_datetime(obs["datetime"])
obs = obs.sort_values("datetime").reset_index(drop=True)
obs["mood_lag1"] = obs["mood"].shift(1)

sub_o = obs.dropna(subset=["mood", "mood_lag1"]).copy()
m_o = sm.OLS(sub_o["mood"], sm.add_constant(sub_o[["mood_lag1"]])).fit()
phi_obs = m_o.params["mood_lag1"]
gaps_h = sub_o["datetime"].diff().dt.total_seconds().div(3600).dropna()
median_gap_h = gaps_h.median()

print(f"Obs-level AR(1)  n = {int(m_o.nobs)}")
print(f"  phi = {phi_obs:.3f}, 95% CI [{m_o.conf_int().loc['mood_lag1',0]:.3f}, "
      f"{m_o.conf_int().loc['mood_lag1',1]:.3f}], p = {m_o.pvalues['mood_lag1']:.4f}")
print(f"  Median between-prompt interval: {median_gap_h:.2f} h")

# Full-sample day-level AR(1)
day = pd.read_csv(OUT_DIR / "clean_day.csv").sort_values("study_day").reset_index(drop=True)
day["mood_lag1"] = day["mood"].shift(1)
sub_d = day.dropna(subset=["mood", "mood_lag1"])
m_d = sm.OLS(sub_d["mood"], sm.add_constant(sub_d[["mood_lag1"]])).fit()
phi_day_full = m_d.params["mood_lag1"]
# Lag-1 autocorrelation (Pearson r) for comparison
r_day = sub_d["mood"].corr(sub_d["mood_lag1"])
print(f"\nDay-level AR(1) on full 70-day sample, n = {int(m_d.nobs)}")
print(f"  OLS phi = {phi_day_full:.3f}; lag-1 Pearson r = {r_day:.3f}")

# Half-lives
half_obs = np.log(0.5) / np.log(phi_obs)
half_day = np.log(0.5) / np.log(phi_day_full)
print(f"\nHalf-lives:")
print(f"  obs-level: {half_obs:.2f} steps ~ {half_obs*median_gap_h:.1f} hours")
print(f"  day-level (full sample): {half_day:.2f} days")

# Save for downstream
pd.DataFrame({
    "level":           ["obs", "day"],
    "phi":             [phi_obs, phi_day_full],
    "half_life_steps": [half_obs, half_day],
    "physical_unit":   [f"{half_obs*median_gap_h:.1f} hours", f"{half_day:.2f} days"],
    "step_hours":      [median_gap_h, 24.0],
    "n_lag1_pairs":    [int(m_o.nobs), int(m_d.nobs)],
    "ci_lo":           [m_o.conf_int().loc["mood_lag1", 0], m_d.conf_int().loc["mood_lag1", 0]],
    "ci_hi":           [m_o.conf_int().loc["mood_lag1", 1], m_d.conf_int().loc["mood_lag1", 1]],
    "p":               [m_o.pvalues["mood_lag1"], m_d.pvalues["mood_lag1"]],
}).to_csv(OUT_DIR / "phi_halflife.csv", index=False)
