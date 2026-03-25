# F1 Strategy Champion Analysis

> *"How do podium finishers differ from midfield in tire strategy and pace management, and what should Ferrari change?"*

---

## Overview

A multi-season Formula 1 analysis covering the **2022–2025 ground effect era**, examining how championship-contending drivers differ from the midfield in tire compound selection, pit stop timing, and lap-time degradation. The project is benchmarked from **Ferrari's perspective**: where does raw pace go unrealized, and what strategic adjustments could recover positions?

Pulls live race data from the [FastF1 API](https://github.com/theOehrly/Fast-F1) and runs a full 6-phase analytics pipeline (Ask → Prepare → Process → Analyze → Share → Act).

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python** (FastF1, pandas, NumPy, SciPy, Matplotlib) | Data ingestion, cleaning, analysis, charting |
| **FastF1 API** | Live F1 lap and telemetry data (cached locally) |
| **Tableau** | Interactive strategy dashboard |

---

## Data Source

- **FastF1 Python library**, pulls official F1 timing data for every race session
- Seasons: 2022, 2023, 2024
- Scope: Full-time drivers only (≥50% of races per season); dry-compound laps only (SOFT, MEDIUM, HARD); finished drivers only (DNF/DSQ removed)

---

## Methodology

### Phase 1 — Ask
Define the business question and scope: podium (P1–P3) vs. Top10 (P4–P10) vs. Back (P11+), Ferrari as benchmark team.

### Phase 2 — Prepare
Pull all race sessions via FastF1 for 2022–2024. Filter to full-time drivers (the majority of races per season) to exclude one-off entries that would skew the data.

### Phase 3 — Process
- Remove null lap times, pit in/out laps, wet-weather compound laps
- Apply 110% fastest lap filter to remove safety car and anomaly laps
- Z-score normalize lap times within each race for cross-circuit comparison
- Compute `lap_in_stint` (lap number within each tire stint)

### Phase 4 — Analyze
- **Tire degradation slope**, OLS regression of lap time vs. lap-in-stint per driver per stint
- **Strategy delta**, pace rank minus finish position (positive = strategy gained positions)
- **Pit window analysis**, classify pit stop timing as early (<33% race distance), mid, or late
- **Ferrari benchmark**, compare Ferrari's metrics against podium finishers and the full field

### Phase 5 — Share
Export 7 CSVs to `tableau_export/` for Tableau dashboard use, plus Ferrari-specific exports.

### Phase 6 — Act
Data-driven recommendations on pit timing, compound selection, and circuit-specific strategy planning.

---

## Key Findings

| Metric | Podium Drivers | Ferrari | Field Avg |
|---|---|---|---|
| Strategy delta (avg positions gained) | See dashboard | See dashboard | Baseline |
| Early pit share (<33% race distance) | Higher | See dashboard | — |
| SOFT compound lap % | See dashboard | See dashboard | — |

*Full quantified findings in the Tableau dashboard and PDF reports.*

---

## Recommendations

1. **Pit Timing**, Commit to early undercut window (laps 12–18) on high-degradation circuits; data shows podium finishers are more aggressive here
2. **Compound Selection**, Calibrate SOFT stint length by circuit degradation rate: 12–14 laps on high-deg circuits, 18–22 on low-deg circuits
3. **Circuit Focus**, Prioritize strategy planning resources on high-variance circuits (Monaco, Hungary, Austria) where strategy delta is largest

---

## Repository Contents

```
├── F1_Strategy_Champion_V3.py           # Full Python pipeline (production version)
├── F1_Strategy_Champion_Analysis.twbx   # Tableau interactive dashboard
├── F1_Strategy_Tableau_Dashboard_1.pdf  # Dashboard export — tire strategy timeline
├── F1_Strategy_Tableau_Dashboard_2.pdf  # Dashboard export — degradation curves
├── F1_Strategy_Tableau_Dashboard_3.pdf  # Dashboard export — Ferrari benchmark
├── F1_Strategy_Champion_Analysis_V2.pdf # Full written analysis report
└── Python_F1_Charts.pdf                 # Python-generated chart exports
```

---

## How to Run

```bash
# 1. Install dependencies
pip install fastf1 pandas numpy scipy matplotlib

# 2. Run the pipeline (first run will download and cache F1 data — takes ~20 min)
python F1_Strategy_Champion_V3.py
```

On first run, FastF1 downloads and caches session data to `f1_cache/`. Subsequent runs load from cache and are much faster. Tableau CSVs are exported to `tableau_export/` and charts to `charts/`.

---

*Analysis covers the 2022–2025 F1 ground effect regulation era.*
