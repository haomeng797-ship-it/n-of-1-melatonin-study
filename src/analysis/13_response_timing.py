"""
13_response_timing.py
How close each scheduled response fell to its nominal prompt time
(10:00, 16:00, 22:00). Writes outputs/response_timing.csv (Section 2.4).

Run from repo root:
    python src/analysis/13_response_timing.py
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs"

obs = pd.read_csv(OUT_DIR / "clean_obs.csv", parse_dates=["datetime"])
hours = obs["datetime"].dt.hour + obs["datetime"].dt.minute / 60
# distance in minutes to the nearest nominal prompt time
gap = pd.concat([(hours - h).abs() for h in (10, 16, 22)], axis=1).min(axis=1) * 60

out = pd.DataFrame({
    "key": ["n", "within_15_min", "within_60_min", "beyond_60_min", "max_gap_hours"],
    "value": [len(gap), int((gap <= 15).sum()), int((gap <= 60).sum()),
              int((gap > 60).sum()), round(gap.max() / 60, 2)],
})
out.to_csv(OUT_DIR / "response_timing.csv", index=False)
print(out.to_string(index=False))
