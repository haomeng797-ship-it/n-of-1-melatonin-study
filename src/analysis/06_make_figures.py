"""
06_make_figures.py
Regenerate the six manuscript figures (Figures 1-3 and A.1-A.3) from the
analysis outputs. Every number drawn is already in outputs/.

Run from repo root (after running 01-05 and 10 first):
    python src/analysis/06_make_figures.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as sm
REPO = Path(__file__).resolve().parents[2]; FIG = REPO / "figures"; OUT = REPO / "outputs"; DPI = 400
day = pd.read_csv(OUT / "clean_day.csv"); obs = pd.read_csv(OUT / "clean_obs.csv")
obs["datetime"] = pd.to_datetime(obs["datetime"]); day["date"] = pd.to_datetime(day["date"])

# Black and white throughout. Line weights: frame and ticks thinnest, data a little heavier, fitted lines heaviest.
K = "#000000"; FR = 0.35; LW = 0.5; FIT = 0.9
DOT = (0, (1, 2.5)); DASH = (0, (4, 2.5))     # dotted = guides, dashed = confidence limits
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "legend.handletextpad": 0.3, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
    "text.color": K, "axes.labelcolor": K, "axes.edgecolor": K, "xtick.color": K, "ytick.color": K,
    "axes.linewidth": FR, "xtick.major.width": FR, "ytick.major.width": FR,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False, "axes.labelpad": 3, "figure.facecolor": "white", "savefig.facecolor": "white"})
C = dict(facecolor="white", edgecolor=K)   # control day: open circle
M = dict(facecolor=K, edgecolor=K)         # melatonin day: filled circle
def save(f, n): f.savefig(FIG / n, dpi=DPI, facecolor="white"); plt.close(f)
def panel(ax, s, dx=-0.02): ax.text(dx, 1.06, s, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="right")
def limits(ax, x, lo_, hi_, lw=0.55):
    for y in (lo_, hi_): ax.plot(x, y, color=K, lw=lw, ls=DASH, zorder=2)
def hnd(ms=3.8): return [Line2D([], [], marker="o", ls="", ms=ms, mfc="white", mec=K, mew=LW, label="Control day"),
                         Line2D([], [], marker="o", ls="", ms=ms, mfc=K, mec=K, mew=LW, label="Melatonin day")]
def dayax(ax): ax.set_xlim(-0.5, 71.5); ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70]); ax.set_xlabel("Study day")

# Figure 1: all 193 ratings behind the 70 daily means
t0 = day["date"].min(); obs["sd"] = (obs["datetime"] - t0) / pd.Timedelta(days=1) + 0.5
f, ax = plt.subplots(figsize=(7.5, 3.3), dpi=DPI, layout="constrained")
ax.scatter(obs["sd"], obs["mood"], s=6, color=K, marker="+", linewidth=0.3, zorder=1)
ax.plot(day["study_day"], day["mood"], color=K, lw=FR, zorder=2)
for c, st in [("control", C), ("melatonin", M)]:
    m = day["condition"] == c; ax.scatter(day.loc[m, "study_day"], day.loc[m, "mood"], s=15, linewidth=LW, zorder=3, **st)
ax.axvline(17.5, color=K, lw=0.4, ls=DOT, zorder=1)
ax.text(9, 1.01, "Mood only", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
ax.text(44.5, 1.01, "Mood, agency and metacognition", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
dayax(ax); ax.set_ylabel("Mood (0–100)"); ax.set_ylim(46, 93)
ax.legend(handles=hnd() + [Line2D([], [], marker="+", ls="", ms=2.8, mec=K, mew=0.3, label="Single rating")], loc="lower right", ncol=3, columnspacing=1.2)
save(f, "fig1_mood_series.png")

# Figure 2: explained variance, by model and by predictor
fit = pd.read_csv(OUT / "ar1_fit_table.csv"); d = pd.read_csv(OUT / "delta_r2.csv").set_index("predictor")
f, (a, b) = plt.subplots(1, 2, figsize=(7.5, 1.9), dpi=DPI, layout="constrained", gridspec_kw={"wspace": 0.12})
def rows(ax, vals, labs, fmt, xmax, xl):
    y = np.arange(len(vals))[::-1]
    ax.hlines(y, 0, vals, color=K, lw=LW, zorder=2); ax.scatter(vals, y, s=16, color=K, zorder=3, clip_on=False)
    for yy, v in zip(y, vals): ax.text(v + xmax * 0.025, yy, fmt(v), va="center", ha="left", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.set_ylim(-0.6, len(vals) - 0.4); ax.set_xlim(-0.04 * xmax, xmax); ax.set_xlabel(xl)
    ax.tick_params(axis="y", length=0, pad=5)
rows(a, fit["R2"].values * 100, ["M0: carryover", "M1: + melatonin", "M2: + agency, metacognition"], lambda v: f"{v:.1f}", 80, "Variance in daily mood explained (%)")
rows(b, [d.loc[k, "deltaR2"] * 100 for k in ["melatonin", "agency", "metacognition"]], ["Melatonin", "Agency", "Metacognition"], lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}", 25, "Unique variance in M2 (%)")
panel(a, "A", -0.42); panel(b, "B", -0.22)
save(f, "fig2_variance.png")

# Figure 3: condition comparison and the equivalence test
tost = pd.read_csv(OUT / "equivalence_tost.csv").set_index("key")["value"]
f, axes = plt.subplots(1, 4, figsize=(7.5, 2.9), dpi=DPI, layout="constrained", gridspec_kw={"width_ratios": [1.12, 1, 1, 1.25]})
for ax, var, lb, pl in zip(axes[:3], ["mood", "agency", "metacognition"], ["Mood", "Agency", "Metacognition"], "ABC"):
    xs = [day.loc[day.condition == c_, var].dropna().values for c_ in ("control", "melatonin")]
    bp = ax.boxplot(xs, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False, showcaps=False)
    for i in range(2):
        bp["boxes"][i].set(facecolor="none", edgecolor=K, linewidth=LW); bp["medians"][i].set(color=K, linewidth=FIT)
        for w in bp["whiskers"][2 * i:2 * i + 2]: w.set(color=K, linewidth=LW)
    for i, (x, st) in enumerate(zip(xs, [C, M])):
        j = np.random.RandomState(42 + i).uniform(-0.14, 0.14, size=len(x))
        ax.scatter(np.full(len(x), i) + j, x, s=7, linewidth=0.45, zorder=3, **st)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "Melatonin"]); ax.set_xlim(-0.6, 1.6); ax.tick_params(axis="x", length=0)
    ax.set_ylim(58, 88); ax.set_yticks(range(60, 90, 5))   # one scale for the three panels
    if ax is not axes[0]: ax.set_yticklabels([])
    ax.set_title(lb, pad=4); panel(ax, pl, -0.12 if ax is axes[0] else -0.04)
axes[0].set_ylabel("Daily mean (0–100)")
ax = axes[3]
for bnd, ls in [(5, DASH), (3, DOT)]:
    for s_ in (-1, 1): ax.axvline(s_ * bnd, color=K, lw=LW, ls=ls, zorder=1)
ax.axvline(0, color=K, lw=LW, zorder=1)
ax.errorbar([tost["desc.diff"]], [0], xerr=[[tost["desc.diff"] - tost["tost.ci90_lo"]], [tost["tost.ci90_hi"] - tost["desc.diff"]]], fmt="o", ms=4.5, mfc=K, mec=K, ecolor=K, elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)
ax.set_xlim(-6.5, 6.5); ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xticks([-5, -3, 0, 3, 5])
ax.set_xlabel("Melatonin − control, 90% CI"); ax.set_title("Equivalence test for mood", pad=4); panel(ax, "D", -0.02)
save(f, "fig3_condition.png")

# Figure A.1: lag-1 scatter with the OLS line and its 95% confidence limits
s = day.dropna(subset=["mood"]).sort_values("study_day").reset_index(drop=True); s["l"] = s["mood"].shift(1); s = s.dropna(subset=["l"])
mm = sm.OLS(s["mood"], sm.add_constant(s["l"])).fit(); phi = mm.params["l"]
lo, hi = 57, 89; xx = np.linspace(lo, hi, 100); pr = mm.get_prediction(sm.add_constant(pd.Series(xx, name="l"))).summary_frame()
f, ax = plt.subplots(figsize=(4.7, 4.7), dpi=DPI, layout="constrained")
limits(ax, xx, pr["mean_ci_lower"], pr["mean_ci_upper"])
ax.plot(xx, xx, ls=DOT, color=K, lw=LW, zorder=2)
ax.plot(xx, pr["mean"], color=K, lw=FIT, zorder=3)
ax.scatter(s["l"], s["mood"], s=14, facecolor="white", edgecolor=K, linewidth=LW, zorder=4)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.set_xticks(range(60, 90, 5)); ax.set_yticks(range(60, 90, 5))
ax.set_xlabel("Mood, day t − 1"); ax.set_ylabel("Mood, day t")
ax.legend(handles=[Line2D([], [], color=K, lw=FIT, label=f"AR(1) fit, φ = {phi:.2f}"), Line2D([], [], color=K, lw=LW, ls=DOT, label="y = x"),
                   Line2D([], [], color=K, lw=0.55, ls=DASH, label="95% CI")], loc="upper left")
save(f, "figA1_lag1.png")

# Figure A.2: impulse response, with the responses implied by the 95% limits of phi
ph = pd.read_csv(OUT / "phi_halflife.csv", index_col=0); hh = np.linspace(0, 7, 141); hi_ = np.arange(0, 8)
f, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), dpi=DPI, layout="constrained")
for ax, lv, xl, tt, pl in zip(axes, ["obs", "day"], ["Lag (prompts, about 6 h apart)", "Lag (days)"], ["Observation level", "Daily level"], "AB"):
    p_, pl_, ph_ = ph.loc[lv, "phi"], ph.loc[lv, "ci_lo"], ph.loc[lv, "ci_hi"]; hl = np.log(0.5) / np.log(p_)
    limits(ax, hh, pl_ ** hh, ph_ ** hh)
    ax.plot(hh, p_ ** hh, color=K, lw=FIT, zorder=3); ax.scatter(hi_, p_ ** hi_, s=12, facecolor="white", edgecolor=K, linewidth=LW, zorder=4, clip_on=False)
    ax.plot([-0.3, hl, hl], [0.5, 0.5, -0.045], color=K, lw=0.4, ls=DOT, zorder=2)
    ax.text(0.98, 0.96, f"φ = {p_:.2f} [{pl_:.2f}, {ph_:.2f}]\nhalf-life " + (f"{hl:.2f} prompts ({hl * ph.loc[lv, 'step_hours']:.1f} h)" if lv == "obs" else f"{hl:.2f} days"),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, linespacing=1.45)
    ax.set_xlabel(xl); ax.set_title(tt, pad=4); ax.set_ylim(-0.045, 1.045); ax.set_xlim(-0.3, 7.3); ax.set_xticks(hi_); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    panel(ax, pl, -0.07 if lv == "obs" else -0.03)
axes[0].set_ylabel("Impulse response"); axes[1].set_yticklabels([])
save(f, "figA2_irf.png")

# Figure A.3: smoothed latent state
k = pd.read_csv(OUT / "kalman_trajectory.csv")
f, ax = plt.subplots(figsize=(7.5, 3.1), dpi=DPI, layout="constrained")
limits(ax, k["study_day"], k["ci_lo"], k["ci_hi"], lw=0.4)
ax.plot(k["study_day"], k["smoothed"], color=K, lw=0.6, zorder=2)
for c, st in [("control", C), ("melatonin", M)]:
    m = k["condition"] == c; ax.scatter(k.loc[m, "study_day"], k.loc[m, "y"], s=11, linewidth=LW, zorder=3, **st)
dayax(ax); ax.set_ylabel("Daily mean mood (0–100)")
ax.legend(handles=hnd(3.4) + [Line2D([], [], color=K, lw=0.6, label="Smoothed latent state"), Line2D([], [], color=K, lw=0.4, ls=DASH, label="95% CI")], loc="lower right", ncol=2, columnspacing=1.0)
save(f, "figA3_kalman.png")
print("six figures written to", FIG)
