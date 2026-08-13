"""
01_data_prep.py
Load cleaned EMA dataset and produce obs-level and day-level frames.

The repository ships with the cleaned dataset (data/miura_ema_70day.csv).
This script verifies and re-derives the day-level aggregates for downstream
analyses. It also records the Day-18 protocol amendment (agency and
metacognition items added on 2026-03-07) by reporting when those columns
first contain values.

Run from repo root:
    python src/analysis/01_data_prep.py
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR  = REPO_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

obs_raw = pd.read_csv(DATA_DIR / "miura_ema_70day.csv")
obs_raw["datetime"] = pd.to_datetime(obs_raw["datetime"])
obs_raw["date"]     = pd.to_datetime(obs_raw["date"])

# --------------------------------------------------------------------------
# Analysis inclusion rule
#
# The archived raw log holds 195 records. The randomized schedule fixed 70
# study days of three planned prompts each, so 210 planned slots in total.
# Study day 44 carries pings on two adjacent calendar dates: its three planned
# prompts were completed on 2026-04-02, and two further records were entered on
# 2026-04-03. Those two fill none of the 210 planned slots, and keeping them
# would leave day 44 as the only day-level mean averaged over five pings drawn
# from two calendar dates. They are excluded from the analytic dataset and
# written out separately; data/miura_ema_70day.csv is left intact as the
# archival record of everything that was logged.
# --------------------------------------------------------------------------
unscheduled = (obs_raw["study_day"] == 44) & (obs_raw["date"] == pd.Timestamp("2026-04-03"))
assert unscheduled.sum() == 2, \
    f"inclusion rule expected exactly 2 unscheduled records, found {int(unscheduled.sum())}"

excluded = obs_raw.loc[unscheduled].copy()
obs      = obs_raw.loc[~unscheduled].copy().reset_index(drop=True)
excluded.to_csv(OUT_DIR / "excluded_observations.csv", index=False)

assert len(obs_raw) == 195, f"raw log should hold 195 records, found {len(obs_raw)}"
assert len(obs)     == 193, f"analytic set should hold 193 records, found {len(obs)}"
assert (obs.groupby("study_day")["date"].nunique() == 1).all(), \
    "a study day still spans more than one calendar date"

print(f"Raw log        = {len(obs_raw)} records (archived in full)")
print(f"  excluded     = {len(excluded)} unscheduled records on "
      f"{excluded['date'].dt.date.iloc[0]} (study day 44)")
print(f"Analytic set   = {len(obs)} scheduled observations")

# Day-18 protocol amendment marker
first_agency_day = int(obs.dropna(subset=["agency"]).iloc[0]["study_day"])
first_agency_date = obs[obs.study_day == first_agency_day]["date"].iloc[0].date()
print(f"First day with agency/metacognition: Day {first_agency_day} ({first_agency_date})")

# Day-level frame: one row per study day. After the inclusion rule above every
# study day falls on a single calendar date (asserted), so the daily mean is a
# mean over that day's scheduled pings and the day-level series stays at n=70,
# which keeps the downstream lag-1 (AR(1)) structure well defined. Condition and
# melatonin are constant within a study day (asserted below).
assert (obs.groupby("study_day")[["condition", "melatonin"]].nunique() == 1).all().all(), \
    "condition/melatonin is not constant within a study_day"

day = (obs.groupby("study_day", as_index=False)
          .agg(date=("date", "min"),
               condition=("condition", "first"),
               melatonin=("melatonin", "first"),
               mood=("mood", "mean"),
               agency=("agency", "mean"),
               metacognition=("metacognition", "mean"),
               n_obs=("mood", "size"))
          .sort_values("study_day").reset_index(drop=True))
day = day[["date", "study_day", "condition", "melatonin",
           "mood", "agency", "metacognition", "n_obs"]]

# Save for downstream scripts. The day-level file shipped in data/ is a derived
# artifact and is refreshed here so it can never drift from the analytic set.
obs.to_csv(OUT_DIR / "clean_obs.csv", index=False)
day.to_csv(OUT_DIR / "clean_day.csv", index=False)
day.to_csv(DATA_DIR / "miura_ema_70day_daily.csv", index=False)

n_planned = 3 * len(day)
print(f"\nObs-level n   = {len(obs)} (planned {n_planned}, "
      f"scheduled-response compliance = {len(obs)/n_planned*100:.1f}%)")
print(f"Day-level n   = {len(day)} study days")
print(f"  control     = {(day.condition == 'control').sum()}")
print(f"  melatonin   = {(day.condition == 'melatonin').sum()}")
print(f"  days with agency/metacog = {day['agency'].notna().sum()}")

counts = day["n_obs"].value_counts().sort_index(ascending=False)
print("  observations per study day: "
      + ", ".join(f"{n} day(s) with {k}" for k, n in counts.items()))
