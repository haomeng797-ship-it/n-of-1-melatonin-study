"""
06_make_figures.py
Regenerate all 7 figures from the analysis outputs.

Run from repo root (after running 01-05 first):
    python src/analysis/06_make_figures.py

Palette (matches paper figures):
  CONTROL  #4C78A8  — control condition
  ACTIVE   #E45756  — melatonin (active)
  NEUTRAL  #54595F  — secondary elements
  ACCENT   #F2B807  — half-life markers
  METACOG  #DFB55A  — metacognition bar (Option A — softened ochre)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "outputs"
FIG_DIR   = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

day = pd.read_csv(OUT_DIR / "clean_day.csv")
obs = pd.read_csv(OUT_DIR / "clean_obs.csv")
obs["datetime"] = pd.to_datetime(obs["datetime"])
day["date"]     = pd.to_datetime(day["date"])

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})
CONTROL = "#4C78A8"
ACTIVE  = "#E45756"
NEUTRAL = "#54595F"
ACCENT  = "#F2B807"
METACOG = "#DFB55A"

# ---- Figure 1: Daily mood timeseries ----
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
ax.plot(day["study_day"], day["mood"], color=NEUTRAL, linewidth=0.9, alpha=0.5)
for cond, c, lbl in [("control", CONTROL, "Control"),
                     ("melatonin", ACTIVE,  "Active (melatonin)")]:
    m = day["condition"] == cond
    ax.scatter(day.loc[m, "study_day"], day.loc[m, "mood"],
               s=44, color=c, alpha=0.85, edgecolor="white",
               linewidth=0.6, label=lbl, zorder=3)
ax.axvline(18, linestyle=":", color=NEUTRAL, alpha=0.7, linewidth=1)
ax.text(18.5, ax.get_ylim()[1] - 1.2,
        "Day 18: agency &\nmetacognition introduced",
        fontsize=8.5, color=NEUTRAL, va="top")
ax.set_xlabel("Study Day")
ax.set_ylabel("Daily Mood (mean of 3× daily EMA)")
ax.set_title("Daily Mood Over the 70-Day N-of-1 Study", fontsize=13, pad=12)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_daily_mood_timeseries.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 2: Momentary mood across all EMA observations ----
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
for cond, c, lbl in [("control", CONTROL, "Control"),
                     ("melatonin", ACTIVE,  "Active (melatonin)")]:
    m = obs["condition"] == cond
    ax.scatter(obs.loc[m, "datetime"], obs.loc[m, "mood"],
               s=20, color=c, alpha=0.7, edgecolor="white",
               linewidth=0.35, label=lbl, zorder=3)
roll = day.set_index("date")["mood"].rolling(7, center=True, min_periods=3).mean()
ax.plot(roll.index, roll.values, color=NEUTRAL, linewidth=1.8,
        label="7-day rolling mean", zorder=4)
ax.set_xlabel("Date")
ax.set_ylabel("Momentary Mood")
ax.set_title("Momentary Mood Across All EMA Observations", fontsize=13, pad=12)
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_ema_timeseries.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 3: Condition comparison boxplots ----
fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), dpi=120)
for ax, var, lbl in zip(axes, ["mood", "agency", "metacognition"],
                        ["Daily Mood", "Daily Agency", "Daily Metacognition"]):
    ctrl = day.loc[day.condition == "control",   var].dropna().values
    mel  = day.loc[day.condition == "melatonin", var].dropna().values
    bp = ax.boxplot([ctrl, mel], positions=[0, 1], widths=0.55,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], [CONTROL, ACTIVE]):
        patch.set_facecolor(c); patch.set_alpha(0.5); patch.set_edgecolor(c)
    for i, (d_, c) in enumerate([(ctrl, CONTROL), (mel, ACTIVE)]):
        jitter = np.random.RandomState(42 + i).uniform(-0.08, 0.08, size=len(d_))
        ax.scatter([i]*len(d_) + jitter, d_, s=14, color=c, alpha=0.6,
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "Active"])
    ax.set_title(lbl, fontsize=11.5)
    ax.grid(axis="y", alpha=0.25)
    delta = mel.mean() - ctrl.mean()
    ax.text(0.5, ax.get_ylim()[1] - 0.5, f"Δ = {delta:+.2f}",
            ha="center", va="top", fontsize=9.5, color=NEUTRAL, fontstyle="italic")
axes[0].set_ylabel("Score (0–100)")
fig.suptitle("Active vs. Control Condition Comparison (Day-Level Means)",
             fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_condition_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 4: Day-level lag-1 autocorrelation ----
d = day.dropna(subset=["mood"]).sort_values("study_day").reset_index(drop=True)
d["mood_lag1"] = d["mood"].shift(1)
sub = d.dropna(subset=["mood_lag1"])
m = sm.OLS(sub["mood"], sm.add_constant(sub["mood_lag1"])).fit()
phi = m.params["mood_lag1"]
fig, ax = plt.subplots(figsize=(6.5, 6), dpi=120)
ax.scatter(sub["mood_lag1"], sub["mood"], s=46, color=NEUTRAL,
           alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
xx = np.linspace(sub["mood_lag1"].min() - 2, sub["mood_lag1"].max() + 2, 100)
ax.plot(xx, m.params["const"] + phi * xx, color=ACTIVE, linewidth=2.4,
        label=f"AR(1) fit:  φ ≈ {phi:.3f}", zorder=4)
ax.plot(xx, xx, linestyle="--", color=NEUTRAL, alpha=0.6, linewidth=1, label="y = x")
ax.set_xlabel("Mood (Day t − 1)")
ax.set_ylabel("Mood (Day t)")
ax.set_title("Day-Level Lag-1 Autocorrelation\n(Approximate Emotional Inertia)",
             fontsize=12, pad=12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_lag1_autocorrelation.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 5: Kalman latent state ----
ssm = pd.read_csv(OUT_DIR / "kalman_trajectory.csv")
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
for cond, c in [("control", CONTROL), ("melatonin", ACTIVE)]:
    m = ssm["condition"] == cond
    ax.scatter(ssm.loc[m, "study_day"], ssm.loc[m, "y"],
               s=24, color=c, alpha=0.75, edgecolor="white",
               linewidth=0.5, label=f"daily mood ({cond})", zorder=3)
ax.fill_between(ssm["study_day"], ssm["ci_lo"], ssm["ci_hi"],
                color=NEUTRAL, alpha=0.15, label="95% CI", zorder=1)
ax.plot(ssm["study_day"], ssm["smoothed"], color=NEUTRAL, linewidth=2,
        label="Kalman-smoothed latent state", zorder=2)
ax.axvline(18, linestyle=":", color=NEUTRAL, alpha=0.7, linewidth=1)
ax.text(18.5, ax.get_ylim()[1] - 1, "Day 18:\nagency + metacog\nadded",
        fontsize=8, color=NEUTRAL, va="top")
ax.set_xlabel("Study day"); ax.set_ylabel("Mood (0–100)")
ax.set_title("Kalman-smoothed latent affective state, 70-day window",
             fontsize=12, loc="left")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_kalman_latent_state.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 6: IRF decay ----
# Use the full-sample day-level phi from fig4 above
phi_obs = 0.231  # from 02b_ar1_obslevel.py
phi_day = phi
median_gap_h = 6.17
h = np.arange(0, 8)
irf_obs = phi_obs ** h
irf_day = phi_day ** h
half_obs = np.log(0.5) / np.log(phi_obs)
half_day = np.log(0.5) / np.log(phi_day)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
for ax, (h_, irf, p_, half, color, xlbl, hlbl, title) in zip(axes, [
    (h, irf_obs, phi_obs, half_obs, ACTIVE, "Lag (EMA pings, ~6 h apart)",
     f"half-life = {half_obs:.2f} steps\n(~{half_obs*median_gap_h:.1f} hours)",
     "(a) Obs-level IRF (within-day)"),
    (h, irf_day, phi_day, half_day, CONTROL, "Lag (days)",
     f"half-life = {half_day:.2f} days", "(b) Day-level IRF (across-day)"),
]):
    ax.plot(h_, irf, "o-", color=color, linewidth=2, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6, label=f"φ = {p_:.3f}")
    ax.axhline(0.5, color=NEUTRAL, linestyle=":", alpha=0.6)
    ax.axvline(half, color=ACCENT, linestyle="--", alpha=0.7, label=hlbl)
    ax.set_xlabel(xlbl); ax.set_ylabel("Impulse response")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9, framealpha=0.9)
plt.suptitle("Impulse response and half-life of an affective perturbation",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_irf_decay.png", dpi=120, bbox_inches="tight")
plt.close()

# ---- Figure 7: Incremental R^2 ----
dR2 = pd.read_csv(OUT_DIR / "delta_r2.csv").set_index("predictor")
values = [dR2.loc["melatonin",     "deltaR2"] * 100,
          dR2.loc["agency",        "deltaR2"] * 100,
          dR2.loc["metacognition", "deltaR2"] * 100]
labels = ["Melatonin\n(external probe)", "Agency\n(internal)", "Metacognition\n(internal)"]
colors = [ACTIVE, CONTROL, METACOG]
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=120)
bars = ax.bar(labels, values, color=colors, width=0.38, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.45,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=9.5, color="#333333")
ax.set_ylabel("Incremental R² (drop-one, %)", fontsize=10)
ax.set_ylim(0, max(values) * 1.16)
ax.set_title("Variance in daily mood explained by each predictor,\n"
             "over and above the AR(1) baseline", fontsize=10.5, loc="left", color="#222222")
ax.grid(axis="y", alpha=0.18, zorder=0)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.tick_params(axis="both", labelsize=9.5, length=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_delta_r2.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"All 7 figures saved to {FIG_DIR}/")
