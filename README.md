# F1 Strategy Champion Analysis

> *How do podium finishers differ from the rest of the field in tire strategy and pit timing, and what should Ferrari change?*

## Overview

A Formula 1 analysis of the 2022 to 2024 ground effect era, built from Ferrari's point of view: where does Ferrari finish behind its own race pace, and is that a pit timing problem, a tire problem or something else?

Race data comes from the [FastF1](https://github.com/theOehrly/Fast-F1) library (official F1 timing, cached locally) and runs through a six-phase pipeline: Ask, Prepare, Process, Analyze, Share, Act.

## Tools

| Tool | Purpose |
|---|---|
| Python (FastF1, pandas, NumPy, SciPy, scikit-learn, Matplotlib) | Ingestion, cleaning, analysis, model, charts |
| SQLite | Raw and cleaned tables, single source of truth |
| Tableau | Strategy dashboard |

## Data

- 68 races across 2022 (22), 2023 (22) and 2024 (24)
- Full-time drivers only (at least 50% of races in a season)
- 72,189 laps loaded, 54,528 after cleaning: null lap times (1,154), pit in/out laps (4,470), wet compounds (4,384), laps outside 110% of the fastest lap (3,999) and DNF/DSQ drivers (3,654) removed

## Method

- **Strategy delta:** a driver's race-pace rank minus finish position. Positive means strategy gained places beyond raw speed.
- **Tire degradation:** OLS slope of lap time vs. lap in stint, per driver per stint.
- **Pit timing:** first pit lap and pit timing as a share of race distance (early under 33%, mid, late), taken from laps *before* cleaning so pit stops aren't filtered out.
- **Finish position model:** Random Forest on 1,095 driver-race records with 7 features.

## Key findings

| Metric | Podium (P1-P3) | Ferrari | Field |
|---|---|---|---|
| Avg strategy delta (positions) | **+0.40** | **-0.12** | -0.14 |
| Median first pit lap | 17 | 18 | Top 10: 17, P11+: 13 |
| Pit stops in the early window | 35.6% | 36.4% | 39.2% |
| Lap share on SOFT / MEDIUM / HARD | 12.8% / 41.1% / 46.1% | 11.6% / 38.7% / 49.7% | |

- Ferrari loses about half a position per race to strategy compared with podium finishers (-0.12 vs. +0.40).
- **Pit timing doesn't explain the gap.** Ferrari's first stop comes within one lap of podium finishers, and it pits early slightly more often.
- **The difference shows up in tire mix.** Ferrari runs more laps on HARD and fewer on MEDIUM than podium finishers.
- The finish position model explains 73% of the variance (R² = 0.73) with an average error of 2.7 places.
- Strategy delta varies most at the Austrian Grand Prix.

## Recommendations

1. **Look past pit timing.** Ferrari's first stops already match podium finishers, so moving them earlier is unlikely to recover the gap.
2. **Test more MEDIUM running.** Ferrari's heavier HARD usage is the clearest strategy difference in the data. A next step is comparing strategy delta for Ferrari stints by compound sequence.
3. **Prioritize high-variance circuits** like Austria, where strategy choices move results the most.

## Run it

```bash
pip install -r requirements.txt
python python/F1_Strategy_Champion_Python.py
```

The first run downloads every race through FastF1 and takes a while. Later runs use the local cache.

## Change log

- **2026-10-19:** Fixed a bug where pit stops were detected after cleaning had already removed every pit in/out lap, which left the pit timing analysis and the finish position model empty. Findings above come from the fixed pipeline, verified with three identical runs.
