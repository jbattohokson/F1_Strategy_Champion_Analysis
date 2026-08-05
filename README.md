# Podium finishers don't just drive faster.
### They pit smarter. The data shows by how much.

> **[Live Report](https://jbattohokson.github.io/F1_Strategy_Champion_Analysis/F1_Strategy_Champion_Analysis.html)** | **[Tableau Dashboard](https://public.tableau.com/app/profile/julian.batto.hokson/viz/F1_Strategy_Champion_Analysis_V2/Dashboard2)** | [GitHub Repo](https://github.com/jbattohokson/F1_Strategy_Champion_Analysis)

---

## Executive Summary

Raw car speed determines a starting range of possible race results. Strategy determines where within that range a team actually finishes. This analysis quantifies that gap across 69 races in the 2022–2024 Ground Effect era, comparing podium finishers (P1–P3) against mid-field and back-of-grid competitors on three measurable dimensions: pit stop timing, tire compound selection, and stint length management.

Podium finishers gain an average of +0.66 positions over their raw pace ranking through strategy execution alone. Ferrari's average strategy delta across the same period is −0.03. That 0.69-position gap, sustained across a 24-race season, translates to an estimated 72–120 championship points left on the table. All three levers that drive the gap — pit window timing, compound sequencing, and circuit-specific pre-race preparation — are correctable without a car development cycle.

| Metric | Value |
|--------|-------|
| Avg positions gained by podium finishers (strategy) | +0.66 |
| Ferrari avg strategy delta (2022–2024) | −0.03 |
| Position gap vs. podium-tier execution | 0.69 |
| Est. championship points left on table per season | 72–120 |
| Podium pit window on high-deg circuits | Laps 12–18 |
| Peak medium tire degradation rate (2023) | 0.0302 sec/lap |

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python (pandas, NumPy, SciPy, Matplotlib) | Data processing, OLS regression, degradation slope analysis |
| FastF1 API | Lap times, pit stops, tire compounds, weather, telemetry |
| SQL | Aggregation and analytical queries |
| Tableau | Strategy delta visualization, compound usage charts |

---

## The Strategy Delta: Positions gained over raw pace

Strategy Delta = average finish position minus raw pace rank. A positive delta means a driver consistently finishes ahead of where their qualifying pace alone would place them.

A driver who qualifies 7th and finishes 5th has a strategy delta of +2. A driver who qualifies 3rd and finishes 5th has a delta of −2. The delta strips out car performance and isolates the value added — or destroyed — by in-race decisions. Across 69 races, podium-tier drivers average +0.66. Ferrari averages −0.03. That difference compounds every race weekend.

| Position Group | Avg Strategy Delta | Primary Mechanism |
|---------------|-------------------|-------------------|
| Podium (P1–P3) | +0.66 | Early pit windows, compound optionality |
| Top-10 (P4–P10) | +0.12 | Reactive strategy, matched pace |
| Back-of-grid (P11+) | −0.06 | Track position defense, tire conservation |
| Ferrari (2022–2024) | −0.03 | Mid-window pits, HARD compound reliance |

---

## Findings: What the data shows separates podium strategy from field-average execution

### Pit window timing is the single largest strategic separator
Podium finishers consistently initiate their first pit stop between laps 12 and 18 on high and medium-degradation circuits, triggering an undercut window where fresh tires produce faster laps than rivals still on older rubber. Ferrari's pit timing in the analysis period centers on laps 18–22 — the reactive zone, where teams concede the undercut rather than create it. On high-degradation circuits (Hungary, Austria, Suzuka), targeting lap 14 for the first pit yields an estimated 0.3–0.5 position gain per stop — 0.6–1.0 positions per race, compounding to 14–24 additional championship points over 24 races.

### Tire degradation rates vary enough across circuits to require pre-race protocols, not race-day decisions
Medium tire degradation ranged from 0.0049 sec/lap in 2022 to 0.0302 sec/lap in 2023 — a sixfold variance in a single compound across seasons. Soft tires on high-degradation circuits degrade at −0.0814 sec/lap; hard tires on low-degradation circuits hold nearly flat. That variance is circuit-specific and predictable, which means circuit-specific degradation rates should be pre-loaded into the race weekend strategy brief — not calculated race-day.

| Degradation Level | Medium Tire Rate | Optimal Pit Window |
|------------------|-----------------|-------------------|
| High-deg | 0.04–0.05 sec/lap | Laps 12–16 |
| Medium-deg | 0.02–0.03 sec/lap | Laps 16–20 |
| Low-deg | ≤0.01 sec/lap | Laps 20–24 |

### Compound sequencing matters — but it is downstream of pit timing
Podium finishers use SOFT compounds more frequently in opening stints (12.8% of race laps vs. 11.5% for top-10 drivers), with SOFT-MEDIUM-HARD sequences delivering early-race pace, undercut optionality on laps 12–14, and durability to close. Ferrari's data shows over-reliance on SOFT-HARD combinations that skip the MEDIUM stint and reduce late-race flexibility. Medium tires at 18–22 lap stints reduce pace loss by 0.02–0.04 sec/lap versus extended HARD stints on high-degradation circuits.

### Ferrari's 0.69-position strategy gap costs an estimated 72 to 120 championship points per season
The F1 points table is non-linear: a 0.69-position improvement on average across a 24-race season translates to roughly 3–5 additional points per race where strategy is the binding constraint. At 72–120 points over a season, the gap represents the difference between a mid-table constructor and a championship contender in most competitive years — and none of the three levers identified require car development.

---

## Recommendations: Three levers, ranked by implementation speed

### Commit to laps 12–16 on high-degradation circuits
No car development required. Lock lap 14 as the first pit trigger on high-deg circuits (Hungary, Austria, Suzuka, Monaco, Singapore) in the pre-race strategy brief. Expected gain: 0.3–0.5 positions per pit stop.

### Build a circuit strategy card system
One page per race weekend, pre-loaded with historical degradation rate, optimal pit window, compound sequence, and expected position gain range. Reduces race-day strategy variance from reactive to pre-planned. Data source: FastF1 historical pull already in the analysis pipeline.

### Default to SOFT-MEDIUM-HARD on most circuits
Introduce the MEDIUM stint (18–22 laps) into the base compound sequence. SOFT-HARD combinations remove late-race flexibility. Reserve SOFT-HARD for Monza, Spa, and other low-degradation venues where track position defense outweighs undercut creation.

### Shift lap delta comparison to track average, not teammate
Comparing a driver's lap time against their teammate's masks whether the strategy is working relative to the field. Track average comparison identifies undercut opportunities earlier and enables pit scenario simulation every 3–5 laps instead of once pre-race.

---

## Data Scope and Methodology

| Dimension | Value |
|-----------|-------|
| Seasons | 2022, 2023, 2024 (Ground Effect Era) |
| Races analyzed | 69 (post-exclusion: wet conditions, DNFs, red flags removed) |
| Data source | FastF1 API — lap times, pit stops, tire compounds, weather, telemetry |
| Records processed | 156,847 lap records before cleaning |
| Exclusions applied | Null lap times (3,247), pit laps (8,934), wet compounds (2,156), 110% rule outliers (1,829), DNF/DSQ (8,431) |
| Key metrics | Strategy Delta, Degradation Slope (OLS per stint), Compound Usage Share |
| Normalization | Z-score per race to account for circuit-specific pace variance |

FastF1 relies on official FIA broadcast timing. Pit detection has inherent latency of ±1 lap. Safety car and red flag laps are excluded from degradation curve analysis. The strategy delta metric uses qualifying pace rank as the baseline — not championship standing — so it isolates in-race execution from car performance.
