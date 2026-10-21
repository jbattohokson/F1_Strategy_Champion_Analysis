# F1 Strategy Champion Analysis

## The problem

Ferrari has had a front-running car in the 2022–2024 ground-effect era but hasn't converted it into results. **Is Ferrari losing positions to race strategy (pit timing, tire choice), or is the gap something else?**

## Approach

Race data from [FastF1](https://github.com/theOehrly/Fast-F1) (official F1 timing), run through a six-phase pipeline (Ask, Prepare, Process, Analyze, Share, Act):

- **68 races:** 2022 (22), 2023 (22), 2024 (24). Full-time drivers only (≥50% of a season's races).
- **72,189 laps loaded, 54,528 after cleaning.** Removed: null lap times (1,154), pit in/out laps (4,470), wet compounds (4,384), laps outside 110% of the fastest lap (3,999), DNF/DSQ drivers (3,654).
- **Strategy delta** = race-pace rank (from average clean lap time) − finish position. Positive = finished ahead of raw pace.
- **Pace-adjusted delta:** each team's delta minus the average delta that *other* teams' cars got at the same pace rank, with 95% CIs from a bootstrap that resamples whole races.
- **Pit timing** comes from the uncleaned laps (cleaning removes every pit lap). **Tire degradation** is the OLS slope of lap time vs. lap in stint.
- **Finish-position model:** Random Forest, evaluated with race-grouped cross-validation against a pace + grid baseline.

## Findings

1. **The raw comparison was misleading.** On raw strategy delta, podium finishers gain +0.40 positions per race and Ferrari loses 0.12, which looks like half a position lost to strategy. But the metric has a ceiling (the fastest car can only lose places: pace rank 1 averages −0.52), and "podium" is selected for finishing well. Front-runners score low by construction.
2. **Adjusted for pace, Ferrari converts as well as the field.** Compared with other teams' cars at the same pace rank, Ferrari gains **+0.33 positions per race (95% CI −0.05 to +0.67)**. Only Red Bull (+0.54) and Mercedes (+0.50) are clearly above zero; Haas (−0.63) and Kick Sauber (−0.78) are clearly below.
3. **The gap is pace.** Ferrari's cars average pace rank 4.1, against Red Bull's 3.2.
4. **Pit timing already matches the front.** Median first stop: Ferrari lap 18, podium finishers 17. Early-window share (<33% of race distance): Ferrari 36.4%, podium 35.6%, field 39.2%.
5. **Tire mix matches the field.** HARD share of laps: Ferrari 49.7%, rest of field 49.6%. Podium finishers (excluding Ferrari) run 43.1%, less than everyone, which more likely follows from running at the front than causes it.
6. **Strategy features explain little of finishing order.** Holding out whole races, a Random Forest on race pace + grid position scores R² 0.683. Adding pit timing, compound and degradation features raises it to 0.707 (+0.02). The original random-split R² of 0.73 was flattered because drivers from the same race landed in both training and test sets.
7. **Finishing order strays most from pace** at Austria, Emilia Romagna and Monaco (std dev of strategy delta 3.7, 3.5, 3.1).

## Recommendations

1. **Invest in race pace, not strategy changes.** Strategy is converting the pace Ferrari has at or slightly above the field average.
2. **Don't move pit stops earlier or switch compound mix.** Neither differs from the teams Ferrari is trying to beat.
3. **Benchmark conversion against Red Bull and Mercedes,** the only teams clearly beating the pace-adjusted baseline. Their edge over Ferrari (~0.2 positions/race) is within noise here; a per-race comparison of their stop decisions under safety cars is the next test.
4. **Weight pre-race planning toward high-variance circuits** (Austria, Imola, Monaco), using each circuit's own history.

## Run it

```bash
pip install -r requirements.txt
python python/F1_Strategy_Champion_Python.py            # reads the committed python/raw_data/ CSVs
python python/F1_Strategy_Champion_Python.py --refresh  # re-download everything through FastF1 (slow)
python -m pytest tests/
```

Outputs go to `python/tableau_export/` (Tableau CSVs, including `team_pace_adjusted_delta.csv` and `predictive_model_grouped_cv.csv`) and `python/charts/` (`chart9_team_pace_adjusted_delta.png` is the key chart).

## Tools

| Tool | Purpose |
|---|---|
| Python (FastF1, pandas, NumPy, SciPy, scikit-learn, Matplotlib) | Ingestion, cleaning, analysis, model, charts |
| SQLite | Raw and cleaned tables, single source of truth |
| Tableau | Strategy dashboard |

## Limitations

- Pace rank comes from clean-lap averages, which exclude safety-car laps and pit laps, so strategy delta also absorbs reliability, incidents and penalties.
- Degradation slopes don't remove fuel burn-off, so they understate tire wear (HARD's median slope is ~0).
- The Tableau workbook was built on version one and still shows the raw podium comparison.

## Change log

- **2026-10-21 audit:**
  - Added the pace-adjusted delta with race-clustered CIs. It overturns the "Ferrari loses half a position to strategy" headline (see findings 1–2).
  - Compound comparisons now exclude Ferrari from the podium and field groups; the old podium group included Ferrari's own podiums.
  - The model now also reports race-grouped CV against a pace + grid baseline.
  - Recommendations are computed from the data. Before, the script told Ferrari to "commit to earlier stops to match podium" while printing that it already pits early more often, and the SOFT stint lengths and "SOFT → HARD" advice were fixed text.
  - The script now runs offline from the committed `raw_data/` CSVs; before, it always called the FastF1 API even though `.gitignore` says the CSVs are committed for that purpose.
  - Model failures now raise instead of printing "skipped" and carrying on. Two stale charts were removed.
  - Tests added (`tests/test_strategy.py`): cleaning rules, pit detection from raw laps, the ceiling effect on synthetic data with a known answer, a planted strategy edge, a team with exclusive pace ranks, and the timedelta round-trip of the raw CSVs.
- **2026-10-19:** Fixed pit stops being detected after cleaning had removed every pit in/out lap, which had left the pit timing analysis and the model empty.
