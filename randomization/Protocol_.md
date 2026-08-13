# Data Collection Protocol

iOS Shortcuts pipeline for the 70-day N-of-1 study.

## 1. Overview

Automated ecological momentary assessment (EMA) data collection via iOS
Shortcuts, feeding into a CSV for downstream analysis. The pipeline was
developed independently because standard EMA platforms (Qualtrics, REDCap)
introduced too much per-entry friction for a high-frequency, three-times-daily
protocol over a 70-day window. iOS Shortcuts was used as a conditional
programming environment, with control flow (variables, conditionals,
dictionaries, file I/O) treated as equivalent to any lightweight scripting
language.

## 2. Measurement

**Frequency:** Three times per day at approximately 10:00, 16:00, and 22:00.

| Variable        | Prompt                              | Scale     |
|-----------------|-------------------------------------|-----------|
| `mood`          | Current emotional valence           | 0–100     |
| `agency`        | Task progress / sense of forward motion | 0–100 |
| `metacognition` | Awareness of current internal state | 0–100     |
| `melatonin_taken` | Melatonin taken tonight?          | 0 / 1     |
| `override_reason` | Deviation note                    | free text |

Single-item Likert sliders were chosen to minimize response burden. Each
prompt takes approximately 8–12 seconds to complete.

## 3. Output

Each Shortcut run appends one row, with an ISO-8601 timestamp, to a CSV on the
device (`Miura_Data.csv`); this raw log is cleaned and stored in the repo as
`data/miura_ema_70day.csv`, which is the archival record of every record that
was entered (195). The dataset the analyses run on is derived from it by the
inclusion rule in Section 7 and written to `outputs/clean_obs.csv` (193):

```
timestamp,                  mood, agency, metacog, mel, override
2026-03-07T10:00:00-05:00,  72,   65,     80,      1,   N/A
2026-03-07T16:00:00-05:00,  68,   70,     75,      1,   N/A
2026-03-07T22:00:00-05:00,  75,   60,     82,      0,   late dinner
```

## 4. Validation

A Python validation layer (`src/data_logger.py`) is run after each new entry.
It performs three checks: (1) **range check**, verifying that mood, agency,
and metacognition fall within the 0–100 interval; (2) **completeness check**,
verifying that no required field is missing; (3) **duplicate check**, verifying
that no two entries share an identical timestamp.

Invocation: `python src/data_logger.py validate`

## 5. Schedule

The 70-day randomization schedule (`randomization/schedule.json`) was
generated before data collection and was not accessible to the participant
during EMA logging. The schedule produces 35 active and 35 control days with
no runs longer than two consecutive days in either condition. Each study day is
a calendar date. The log covers 71 calendar dates because of the two extra
records described in Section 7; after the inclusion rule is applied, the 70
study days map one-to-one onto 70 calendar dates.

## 6. Day-18 Protocol Amendment

The `agency` and `metacognition` items were added on Day 18 (2026-03-07)
following an interim protocol review. Analyses involving these variables are
correspondingly restricted to Days 18–70.

## 7. Analysis Inclusion Rule

The randomized schedule fixes 70 study days of three prompts each, so 210
planned prompt slots. Two records were entered on 2026-04-03, after the three
prompts for that study day had already been completed on 2026-04-02. They fill
none of the 210 planned slots, and keeping them would leave that one study day
as the only day-level mean averaged over five pings drawn from two calendar
dates.

Those two records are therefore excluded from the analytic dataset. They remain
in `data/miura_ema_70day.csv` and are written out separately to
`outputs/excluded_observations.csv`. The rule is applied in
`src/analysis/01_data_prep.py`, which asserts that exactly two records match it,
that the analytic set holds 193 records, and that every study day then falls on
a single calendar date. All downstream scripts read the analytic set rather than
the archive, so the rule cannot be bypassed.
