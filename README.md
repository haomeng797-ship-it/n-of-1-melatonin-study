# Characterizing Daily Affective Dynamics in a Single Individual

*A control-theoretic N-of-1 study with ecological momentary assessment.*

A 70-day single-subject self-experiment that tests whether nightly melatonin predicts
daily affective state, modeling the individual as a dynamical system and treating
melatonin as a probe of that system rather than as a candidate treatment. Internal
behavioral drivers (perceived agency and metacognitive awareness) jointly explained
51.5% of daily-mood variance beyond an AR(1) baseline, whereas the melatonin probe
added 0.01% and produced only a non-significant trend toward lower day-to-day
variability (α₁ = −1.11, *p* = .071), pointing to a stabilizing rather than a mood-lifting effect.

**Author:** Miura Meng · meng10@upenn.edu · ORCID [0009-0004-1522-1997](https://orcid.org/0009-0004-1522-1997)

**Keywords:** N-of-1; ecological momentary assessment; dynamical systems; autoregressive
model; idiographic; personal science; intensive longitudinal data; affect dynamics

## Manuscript

All versions are in [`paper/`](paper/) (latest: v9).

## Repository layout

- `randomization/`: pre-registered 70-day schedule (`schedule.json`) and protocol
- `data/`: cleaned EMA, observation-level (`miura_ema_70day.csv`, n = 195) and daily (`miura_ema_70day_daily.csv`, n = 70)
- `src/data_logger.py`: iOS Shortcuts entry validation
- `src/analysis/`: analysis scripts `01`–`09` (run order below)
- `outputs/`: cleaned frames, model tables, and `results_table.csv` (every number reported in the paper)
- `figures/`: figures 1–7
- `paper/`: manuscript versions

## Reproducing the analysis

Python 3.10+ (`pip install -r requirements.txt`), run from the repository root:

```bash
python src/analysis/01_data_prep.py                    # clean and verify the dataset
python src/analysis/02_ar1_models.py                   # nested AR(1) models + incremental R²
python src/analysis/02b_ar1_obslevel.py                # observation-level AR(1) + half-life
python src/analysis/03_state_space.py                  # local-level Kalman filter
python src/analysis/04_metacontrol.py                  # melatonin × metacognition interaction
python src/analysis/05_variability_and_interactions.py # variability, interactions, state-dependence
python src/analysis/08_results_table.py                # assemble results_table.csv
python src/analysis/06_make_figures.py                 # regenerate figures 1–7
```

Two supplementary robustness analyses run in R (≥ 4.1):

```r
source("src/analysis/07_bayesian_robustness.R")    # Bayesian M2 (brms)        -> Table 2 / §3.6
source("src/analysis/09_multilevel_robustness.R")  # mixed-effects (lme4)      -> Table 3 / §3.7
```

Every number in the manuscript can be recovered from `outputs/`. The fig1–fig4 PNGs are
the author's polished originals; re-running `06_make_figures.py` reproduces the same
content in matplotlib with minor stylistic differences.

## Method in brief

A 70-day alternating-treatment design randomized each day to melatonin or control (no
run longer than two consecutive days). EMA was collected three times daily (~10:00,
16:00, 22:00) via iOS Shortcuts: mood, agency, and metacognition on 0–100 sliders plus
a melatonin indicator. Agency and metacognition were added on Day 18, so analyses using
them cover Days 18–70. The final sample is 195 observations across 70 study days (92.9%
compliance), with 35 control and 35 melatonin days.

## Availability and citation

Released under [CC-BY 4.0](LICENSE).

> Meng, M. (2026). *Characterizing daily affective dynamics in a single individual:
> A control-theoretic N-of-1 study with ecological momentary assessment.* Manuscript in
> preparation. https://github.com/haomeng797-ship-it/N-of-1-Melatonin-Study
