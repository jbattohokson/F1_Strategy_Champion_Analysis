#F1 GROUND EFFECT ERA — TIRE STRATEGY & PACE ANALYSIS (V2)
#2022–2025 · Phase roadmap: Ask → Prepare → Process → Analyze → Share → Act
#What separates a podium finisher from P8? This script pulls race data from
#f1_cache, cleans it, loads it into SQL, and builds metrics for pit windows,
#compound choice, and strategy delta. Analysis from Ferrari's perspective
#vs full-time drivers only.

import os
import sqlite3
import time
import fastf1
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#CONFIG
#All relevant FastF1 data is read from f1_cache (no API fetch if already cached)
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
CACHE_DIR = os.path.join(_SCRIPT_DIR, "f1_cache")
RAW_DIR = os.path.join(_SCRIPT_DIR, "raw_data")
OUTPUT_DB = os.path.join(_SCRIPT_DIR, "f1_strategy.db")
CHARTS_DIR = os.path.join(_SCRIPT_DIR, "charts")
SEASONS = [2022, 2023, 2024]
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
POSITION_GROUP_LABELS = ["Podium", "Top10", "Back"]   #P1-P3, P4-P10, P11+
TEAM_OF_INTEREST = "Ferrari"
MAJORITY_THRESHOLD = 0.5   #Fraction (0.5 = 50%) of races a driver must enter per season to be included
API_CALL_DELAY = 5         #Seconds to wait between race loads — prevents hitting FastF1's 500 calls/hr limit

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


#PHASE 1 — ASK (define the problem)
#Business question: How do race-winning drivers and mid-field drivers use tire
#strategy and pace management differently in the ground effect era (2022–2025),
#and what should Ferrari change to close the gap between raw pace and results?

#Scope: All 20 drivers per race; we keep only drivers who participated in a
#majority of races in each season (so reserve/one-off entries don't skew the
#picture). Analysis is from Ferrari's perspective: benchmark the field and
#recommend where Ferrari can gain through strategy.

#Measure: delta_to_leader, stint_length, lap_time_degradation_slope,
#position_change_after_pit, pit window timing.

#PHASE 2 — PREPARE (pull data from f1_cache, store raw, explore schema)

def prepare_season_data(seasons):
    """Pull full season race data from FastF1 (reads from f1_cache). Save raw laps and results as CSV."""
    all_laps = []
    all_results = []
    all_weather = []

    for season in seasons:
        schedule = fastf1.get_event_schedule(season)
        for rnd in schedule["RoundNumber"]:
            try:
                session = fastf1.get_session(season, rnd, "R")
                session.load()
                race_name = session.event["EventName"]
                print(f"  Loading {season} {race_name}")

                laps = session.laps.copy()
                laps["season"] = season
                laps["race"] = race_name

                results = session.results.copy()
                results["season"] = season
                results["race"] = race_name

                weather = session.weather_data.copy()
                weather["season"] = season
                weather["race"] = race_name

                all_laps.append(laps)
                all_results.append(results)
                all_weather.append(weather)

                #Pause between race loads to stay under FastF1's 500 calls/hr limit.
                #Without this, loading 3 full seasons in one run will hit the cap
                #partway through 2023 and abort. Cached races skip the API entirely
                #so this only adds real time on a first run.
                time.sleep(API_CALL_DELAY)

            except fastf1.exceptions.RateLimitExceededError:
                print(f"  Rate limit hit at {season} round {rnd} — waiting 60s before retrying...")
                time.sleep(60)
                try:
                    session = fastf1.get_session(season, rnd, "R")
                    session.load()
                    race_name = session.event["EventName"]
                    laps = session.laps.copy()
                    laps["season"] = season
                    laps["race"] = race_name
                    results = session.results.copy()
                    results["season"] = season
                    results["race"] = race_name
                    weather = session.weather_data.copy()
                    weather["season"] = season
                    weather["race"] = race_name
                    all_laps.append(laps)
                    all_results.append(results)
                    all_weather.append(weather)
                    print(f"  Retry succeeded: {season} {race_name}")
                    time.sleep(API_CALL_DELAY)
                except Exception as retry_e:
                    print(f"  Retry failed for {season} round {rnd}: {retry_e}")

            except Exception as e:
                print(f"  Skipped {season} round {rnd}: {e}")

    df_laps = pd.concat(all_laps, ignore_index=True)
    df_results = pd.concat(all_results, ignore_index=True)
    df_weather = pd.concat(all_weather, ignore_index=True)

    #Normalize driver identifier (FastF1 sometimes uses "Abbreviation" instead of "Driver")
    if "Driver" not in df_results.columns and "Abbreviation" in df_results.columns:
        df_results["Driver"] = df_results["Abbreviation"]
    if "Driver" not in df_laps.columns and "Abbreviation" in df_laps.columns:
        df_laps["Driver"] = df_laps["Abbreviation"]

    #Store raw session data as CSV (as per roadmap)
    os.makedirs(RAW_DIR, exist_ok=True)
    df_laps.to_csv(os.path.join(RAW_DIR, "raw_laps.csv"), index=False)
    df_results.to_csv(os.path.join(RAW_DIR, "raw_results.csv"), index=False)
    df_weather.to_csv(os.path.join(RAW_DIR, "raw_weather.csv"), index=False)
    print(f"  Raw data saved to {RAW_DIR}/ (laps, results, weather)")

    return df_laps, df_results, df_weather


#Timedelta columns FastF1 writes; to_csv stores them as text like "0 days 00:01:40.236000"
_TIMEDELTA_COLS = ["Time", "LapTime", "PitOutTime", "PitInTime", "Sector1Time", "Sector2Time", "Sector3Time",
                   "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", "LapStartTime",
                   "Q1", "Q2", "Q3"]


def load_raw_csvs():
    """Read the committed raw_data/ CSVs instead of FastF1.

    AUDITED 2026-10-21: .gitignore says raw_data/ is committed so the analysis can
    rerun without the multi-GB FastF1 cache, but nothing read it back. Every run
    went to the FastF1 API (and fails offline). main() now uses these CSVs when
    they exist; pass --refresh to re-download.
    """
    frames = []
    for name in ["raw_laps.csv", "raw_results.csv", "raw_weather.csv"]:
        df = pd.read_csv(os.path.join(RAW_DIR, name), low_memory=False)
        for col in _TIMEDELTA_COLS:
            if col in df.columns:
                df[col] = pd.to_timedelta(df[col], errors="coerce")
        frames.append(df)
    laps, results, weather = frames
    print(f"  Loaded raw_data/ CSVs: {len(laps):,} laps, {len(results):,} results, "
          f"{results.groupby('season')['race'].nunique().to_dict()} races per season")
    return laps, results, weather


def filter_to_majority_drivers(laps, results, majority_pct=MAJORITY_THRESHOLD):
    """
    Keep only drivers who participated in a majority of races in each season.
    Ensures we compare Ferrari against the full-time field, not one-off entries.
    Returns filtered laps and results (same schema).
    """
    #Races per season
    races_per_season = results.groupby("season")["race"].nunique().reset_index()
    races_per_season = races_per_season.rename(columns={"race": "season_races"})
    #Races per driver per season
    driver_races = results.groupby(["season", "Driver"]).size().reset_index(name="races_done")
    driver_races = driver_races.merge(races_per_season, on="season")
    driver_races["is_majority"] = driver_races["races_done"] >= (
        driver_races["season_races"] * majority_pct
    )
    full_time = driver_races[driver_races["is_majority"]][["season", "Driver"]].drop_duplicates()

    n_laps_before, n_results_before = len(laps), len(results)
    laps = laps.merge(full_time, on=["season", "Driver"], how="inner")
    results = results.merge(full_time, on=["season", "Driver"], how="inner")
    n_drivers = full_time.groupby("season").size()
    print(f"  Full-time drivers (≥{int(majority_pct*100)}% of races): {dict(n_drivers)}")
    print(f"  Laps: {n_laps_before} → {len(laps)}  |  Results rows: {n_results_before} → {len(results)}")
    return laps, results


def load_raw_to_sql(laps_raw, results_raw, weather_raw):
    """
    Write raw data (after majority-driver filter) into SQL so the DB holds
    a single source of truth. Tables: raw_laps, raw_results, raw_weather.
    Timedelta columns are stored as integer ns by SQLite.
    """
    conn = sqlite3.connect(OUTPUT_DB)
    laps_raw.to_sql("raw_laps", conn, if_exists="replace", index=False)
    results_raw.to_sql("raw_results", conn, if_exists="replace", index=False)
    weather_raw.to_sql("raw_weather", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Raw data written to SQL: raw_laps ({len(laps_raw)} rows), raw_results ({len(results_raw)} rows), raw_weather ({len(weather_raw)} rows)")


#PHASE 3 — PROCESS (clean, normalize, load into SQL, build summary tables)

def clean_laps(laps, results):
    """
    Apply report-aligned cleaning. Log before/after row counts for audit.
    Steps: drop null lap time → remove in/out laps → dry compounds only →
    110% fastest-lap filter → exclude DNF/DSQ → z-score normalize per race.
    """
    n_start = len(laps)
    print(f"  Cleaning: starting rows = {n_start}")

    #1. Null lap time drop
    laps = laps.dropna(subset=["LapTime"])
    n_after_null = len(laps)
    print(f"    After dropping null LapTime: {n_after_null} (removed {n_start - n_after_null})")
    #Convert lap time to seconds for analysis
    laps = laps.copy()
    laps["lap_time_sec"] = laps["LapTime"].dt.total_seconds()

    #2. In lap and out lap removal (pit in/out laps are not representative pace)
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    n_after_pit = len(laps)
    print(f"    After removing in/out laps: {n_after_pit} (removed {n_after_null - n_after_pit})")

    #3. Wet compound filter — dry tires only
    laps = laps[laps["Compound"].str.upper().isin(DRY_COMPOUNDS)]
    n_after_dry = len(laps)
    print(f"    After dry-compound filter: {n_after_dry} (removed {n_after_pit - n_after_dry})")

    #4. 110% fastest lap filter (safety car / VSC / incident laps)
    race_fastest = laps.groupby(["season", "race"])["lap_time_sec"].transform("min")
    laps = laps[laps["lap_time_sec"] <= race_fastest * 1.10]
    n_after_110 = len(laps)
    print(f"    After 110% fastest-lap filter: {n_after_110} (removed {n_after_dry - n_after_110})")

    #5. DNF/DSQ removal — keep only laps for drivers who have a valid finish position
    if "Status" in results.columns:
        finished = results[results["Status"].astype(str).str.upper().str.contains("LAP|FINISHED", na=False)]
    else:
        finished = results.dropna(subset=["Position"])
    finished_keys = finished[["season", "race", "Driver"]].drop_duplicates()
    laps = laps.merge(finished_keys, on=["season", "race", "Driver"], how="inner")
    n_after_dnf = len(laps)
    print(f"    After DNF/DSQ removal: {n_after_dnf} (removed {n_after_110 - n_after_dnf})")

    #6. Race-level z-score normalization (cross-circuit comparison)
    laps["lap_time_norm"] = laps.groupby(["season", "race"])["lap_time_sec"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print(f"    Z-score normalization added (no rows removed). Final clean rows = {n_after_dnf}")

    return laps, results

def build_lap_in_stint(laps):
    """Add lap number within stint (TyreLife proxy) for degradation regression."""
    laps = laps.sort_values(["season", "race", "Driver", "LapNumber"])
    laps["lap_in_stint"] = laps.groupby(["season", "race", "Driver", "Stint"]).cumcount() + 1
    return laps

def load_to_sql_and_summarize(laps, results, pit_source=None):
    """
    Load cleaned data into SQLite and build summary tables:
    average pace per stint, pit stop deltas, position changes, degradation by compound/circuit.
    """
    conn = sqlite3.connect(OUTPUT_DB)

    #Clean laps
    laps.to_sql("laps_clean", conn, if_exists="replace", index=False)

    #Position groups: Podium (P1–P3), Top10 (P4–P10), Back (P11+)
    results = results.copy()
    results["position_group"] = pd.cut(
        results["Position"],
        bins=[0, 3, 10, 100],
        labels=POSITION_GROUP_LABELS,
    )
    results.to_sql("results_raw", conn, if_exists="replace", index=False)

    #Summary: average pace per stint
    stint_summary = (
        laps.groupby(["season", "race", "Driver", "Stint", "Compound"])
        .agg(
            stint_length=("LapNumber", "count"),
            avg_pace_sec=("lap_time_sec", "mean"),
            fastest_lap_sec=("lap_time_sec", "min"),
        )
        .reset_index()
    )
    stint_summary.to_sql("stint_summary", conn, if_exists="replace", index=False)

    #Summary: pit stops (first pit lap, count) and position change
    # Pit laps must come from the uncleaned laps: clean_laps() drops every in/out lap,
    # so searching the cleaned frame for PitInTime always found zero pit stops.
    pit_frame = pit_source if pit_source is not None else laps
    pit_laps = pit_frame[pit_frame["PitInTime"].notna()].copy()
    if len(pit_laps) > 0:
        pit_summary = (
            pit_laps.groupby(["season", "race", "Driver"])
            .agg(first_pit_lap=("LapNumber", "min"), pit_count=("LapNumber", "count"))
            .reset_index()
        )
    else:
        pit_summary = pd.DataFrame(columns=["season", "race", "Driver", "first_pit_lap", "pit_count"])

    pos_by_driver = (
        laps.groupby(["season", "race", "Driver"])
        .agg(start_pos=("Position", "first"), end_pos=("Position", "last"))
        .reset_index()
    )
    pos_by_driver["position_change"] = pos_by_driver["start_pos"] - pos_by_driver["end_pos"]

    #Merge for race-level strategy view
    res_cols = ["Driver", "race", "season", "Position", "position_group", "TeamName"]
    if "GridPosition" in results.columns:
        res_cols.append("GridPosition")
    race_strategy = pos_by_driver.merge(
        results[res_cols],
        on=["Driver", "race", "season"],
        how="left",
    )
    if "first_pit_lap" in pit_summary.columns:
        race_strategy = race_strategy.merge(
            pit_summary,
            on=["season", "race", "Driver"],
            how="left",
        )

    stint_summary.to_sql("stint_summary", conn, if_exists="replace", index=False)
    pit_summary.to_sql("pit_strategy", conn, if_exists="replace", index=False)
    race_strategy.to_sql("race_strategy", conn, if_exists="replace", index=False)

    conn.close()
    return stint_summary, pit_summary, race_strategy, results

#PHASE 4 — ANALYZE (SQL-style aggregates + Python metrics)

def calc_degradation_slope(group, min_laps=5):
    """OLS regression: lap_time_sec ~ lap_in_stint. Return slope (sec per lap)."""
    if len(group) < min_laps:
        return np.nan
    slope, _, _, _, _ = stats.linregress(group["lap_in_stint"], group["lap_time_sec"])
    return round(slope, 5)


def build_degradation_curves(laps):
    """Degradation rate per compound and stint; by position group and race for summary tables."""
    degradation_list = []
    for (season, race, driver, stint), group in laps.groupby(["season", "race", "Driver", "Stint"]):
        slope = calc_degradation_slope(group)
        if np.isnan(slope):
            continue
        degradation_list.append({
            "season": season,
            "race": race,
            "Driver": driver,
            "stint": stint,
            "compound": group["Compound"].iloc[0],
            "lap_time_degradation_slope": slope,
            "stint_length": len(group),
        })
    return pd.DataFrame(degradation_list)


def build_strategy_delta(laps, results):
    """Pace rank minus finish position. Positive = finished better than raw pace."""
    avg_pace = laps.groupby(["season", "race", "Driver"])["lap_time_sec"].mean().reset_index()
    avg_pace = avg_pace.rename(columns={"lap_time_sec": "avg_lap_sec"})
    avg_pace["pace_rank"] = avg_pace.groupby(["season", "race"])["avg_lap_sec"].rank(method="min")
    res = results[["season", "race", "Driver", "Position", "TeamName", "position_group"]].copy()
    res = res.rename(columns={"Position": "finish_position"})
    merged = avg_pace.merge(res, on=["season", "race", "Driver"], how="inner")
    merged["strategy_delta"] = merged["pace_rank"] - merged["finish_position"]
    merged["delta_to_leader"] = merged["avg_lap_sec"] - merged.groupby(["season", "race"])["avg_lap_sec"].transform("min")
    return merged


def build_pace_adjusted_delta(strategy_delta, n_boot=2000, seed=42):
    """Strategy delta relative to cars with the same race pace.

    AUDITED 2026-10-21: raw strategy delta (pace_rank - finish) has a ceiling.
    The fastest car can't finish better than P1, so pace rank 1 averages -0.52
    and front-runners look like they "lose places to strategy" by construction.
    And "Podium" is defined by finishing P1-P3, so its delta is pushed up by
    selection. Comparing Ferrari (mean pace rank ~4) with podium finishers
    was therefore circular.

    Here each team is compared with the average delta that OTHER teams' cars
    get at the same pace rank (leave-one-team-out), so a team isn't graded
    against itself. CIs come from a bootstrap that resamples whole races,
    since the two cars in a race aren't independent.

    Returns (per driver-race frame with expected_delta and delta_vs_pace,
             per-team summary).
    """
    sd = strategy_delta.dropna(subset=["pace_rank", "finish_position", "TeamName"]).copy()
    sd["pace_rank"] = sd["pace_rank"].astype(int)
    sd["expected_delta"] = np.nan
    #If no other team ever ran at a pace rank (a dominant car), fall back to the all-car
    #average there rather than dropping those races; baseline_fallback counts how often
    all_cars = sd.groupby("pace_rank")["strategy_delta"].mean()
    sd["baseline_fallback"] = False
    for team in sd["TeamName"].unique():
        others = sd[sd["TeamName"] != team].groupby("pace_rank")["strategy_delta"].mean()
        mask = sd["TeamName"] == team
        expected = sd.loc[mask, "pace_rank"].map(others)
        sd.loc[mask, "baseline_fallback"] = expected.isna()
        sd.loc[mask, "expected_delta"] = expected.fillna(sd.loc[mask, "pace_rank"].map(all_cars))
    sd["delta_vs_pace"] = sd["strategy_delta"] - sd["expected_delta"]

    rng = np.random.default_rng(seed)
    rows = []
    for team, g in sd.dropna(subset=["delta_vs_pace"]).groupby("TeamName"):
        by_race = g.groupby(["season", "race"])["delta_vs_pace"].agg(["sum", "count"])
        sums, counts = by_race["sum"].to_numpy(), by_race["count"].to_numpy()
        idx = rng.integers(0, len(by_race), size=(n_boot, len(by_race)))
        boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        rows.append({
            "TeamName": team,
            "driver_races": len(g),
            "baseline_fallback_rows": int(g["baseline_fallback"].sum()),
            "races": len(by_race),
            "mean_pace_rank": g["pace_rank"].mean(),
            "mean_finish": g["finish_position"].mean(),
            "raw_strategy_delta": g["strategy_delta"].mean(),
            "delta_vs_pace": g["delta_vs_pace"].mean(),
            "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5),
        })
    summary = pd.DataFrame(rows).sort_values("delta_vs_pace", ascending=False).reset_index(drop=True)
    return sd, summary


def build_pit_window_analysis(laps, results, race_strategy):
    """Pit window: early (<33%), mid (33-66%), late (>66%) of race distance.
    One row per pit stop per driver per race, labeled by window timing."""
    #Total laps per race to compute pit timing as a percentage of race distance
    total_laps = laps.groupby(["season", "race"])["LapNumber"].max().reset_index()
    total_laps = total_laps.rename(columns={"LapNumber": "total_laps"})

    #All pit-in laps — one row per pit stop (PitInTime notna = lap where car pitted)
    pit_in_laps = laps[laps["PitInTime"].notna()][["season", "race", "Driver", "LapNumber"]].copy()

    if len(pit_in_laps) == 0:
        return pd.DataFrame(columns=["season", "race", "Driver", "LapNumber",
                                     "total_laps", "pit_lap_pct", "pit_window"])

    #Merge total laps so we can express pit lap as fraction of race distance
    pits = pit_in_laps.merge(total_laps, on=["season", "race"])
    pits["pit_lap_pct"] = pits["LapNumber"] / pits["total_laps"]

    #Bin into three windows: early (<33%), mid (33-66%), late (>66%)
    pits["pit_window"] = pd.cut(
        pits["pit_lap_pct"],
        bins=[0, 1/3, 2/3, 1.0],
        labels=["early", "mid", "late"],
    )

    #Join position group and TeamName from results for Ferrari benchmark filtering
    res_cols = ["season", "race", "Driver", "position_group"]
    if "TeamName" in results.columns:
        res_cols.append("TeamName")
    res_short = results[res_cols].drop_duplicates()
    pits = pits.merge(res_short, on=["season", "race", "Driver"], how="left")

    return pits

#PHASE 5 — SHARE (exports for Tableau)

def export_tableau_files(laps, degradation, race_strategy, strategy_delta, pit_windows, stint_summary):
    """Export CSVs for Tableau: Gantt-style timelines, degradation curves, heatmap, scatter.
    Also export Ferrari-only subset for benchmarking from Ferrari's perspective."""
    out = os.path.join(_SCRIPT_DIR, "tableau_export")
    os.makedirs(out, exist_ok=True)

    #Tire strategy timelines (Gantt-style: race, driver, stint, compound, start_lap, end_lap)
    gantt = (
        laps.groupby(["season", "race", "Driver", "Stint", "Compound"])
        .agg(start_lap=("LapNumber", "min"), end_lap=("LapNumber", "max"))
        .reset_index()
    )
    gantt.to_csv(os.path.join(out, "tire_strategy_timeline.csv"), index=False)

    #Degradation curves by compound and finishing group (need position_group on laps)
    deg_with_group = degradation.merge(
        race_strategy[["season", "race", "Driver", "position_group"]],
        on=["season", "race", "Driver"],
        how="left",
    )
    deg_with_group.to_csv(os.path.join(out, "degradation_curves.csv"), index=False)

    #Circuit pit strategy (pit window by race) for heatmap
    pit_windows.to_csv(os.path.join(out, "pit_windows_by_race.csv"), index=False)

    #Scatter: pit stop lap vs net positions gained
    scatter = race_strategy[["season", "race", "Driver", "first_pit_lap", "position_change", "position_group"]].copy()
    scatter = scatter.rename(columns={"first_pit_lap": "pit_stop_lap", "position_change": "net_positions_gained"})
    scatter.to_csv(os.path.join(out, "pit_lap_vs_positions_gained.csv"), index=False)

    #Lap-level and stint summary for dashboards
    laps.to_csv(os.path.join(out, "laps_clean.csv"), index=False)
    stint_summary.to_csv(os.path.join(out, "stint_summary.csv"), index=False)
    strategy_delta.to_csv(os.path.join(out, "strategy_delta.csv"), index=False)

    #Ferrari-only subset (from Ferrari's perspective: compare to this benchmark)
    if "TeamName" in race_strategy.columns:
        ferrari = race_strategy[race_strategy["TeamName"] == TEAM_OF_INTEREST]
        if len(ferrari) > 0:
            ferrari.to_csv(os.path.join(out, "ferrari_race_strategy.csv"), index=False)
            ferrari_sd = strategy_delta[strategy_delta["TeamName"] == TEAM_OF_INTEREST]
            if len(ferrari_sd) > 0:
                ferrari_sd.to_csv(os.path.join(out, "ferrari_strategy_delta.csv"), index=False)
            print(f"  Ferrari benchmark exports: ferrari_race_strategy.csv, ferrari_strategy_delta.csv")

    #Ferrari benchmark summary (answers "how does Ferrari compare?")
    pace_adj, team_summary = build_pace_adjusted_delta(strategy_delta)
    pace_adj.to_csv(os.path.join(out, "strategy_delta_pace_adjusted.csv"), index=False)
    team_summary.to_csv(os.path.join(out, "team_pace_adjusted_delta.csv"), index=False)
    ferrari_benchmark = build_ferrari_benchmark(strategy_delta, pit_windows, laps, degradation)
    if ferrari_benchmark is not None and len(ferrari_benchmark) > 0:
        pd.DataFrame([ferrari_benchmark]).to_csv(os.path.join(out, "ferrari_benchmark.csv"), index=False)
        print(f"  Ferrari benchmark: ferrari_benchmark.csv")

    print(f"  Tableau exports written to {out}/")
    return ferrari_benchmark


def build_ferrari_benchmark(strategy_delta, pit_windows, laps, degradation):
    """Compute Ferrari vs Podium vs Field on key metrics. Returns a dict for CSV/print."""
    if not ("TeamName" in strategy_delta.columns and (strategy_delta["TeamName"] == TEAM_OF_INTEREST).any()):
        return None
    out = {}
    #Strategy delta
    out["ferrari_avg_strategy_delta"] = strategy_delta.loc[strategy_delta["TeamName"] == TEAM_OF_INTEREST, "strategy_delta"].mean()
    podium = strategy_delta[strategy_delta["position_group"] == "Podium"]
    out["podium_avg_strategy_delta"] = podium["strategy_delta"].mean() if len(podium) else np.nan
    out["field_avg_strategy_delta"] = strategy_delta["strategy_delta"].mean()
    #Early pit share
    if len(pit_windows) > 0 and "pit_window" in pit_windows.columns and "TeamName" in pit_windows.columns:
        ferrari_pits = pit_windows[pit_windows["TeamName"] == TEAM_OF_INTEREST]
        out["ferrari_pct_early_pit"] = 100 * (ferrari_pits["pit_window"] == "early").mean() if len(ferrari_pits) else np.nan
        podium_pits = pit_windows[pit_windows["position_group"] == "Podium"]
        out["podium_pct_early_pit"] = 100 * (podium_pits["pit_window"] == "early").mean() if len(podium_pits) else np.nan
        out["field_pct_early_pit"] = 100 * (pit_windows["pit_window"] == "early").mean()
    #Compound usage: % laps on SOFT / MEDIUM / HARD. Podium and field exclude Ferrari's
    #own laps so Ferrari isn't partly compared with itself
    if "TeamName" in laps.columns:
        not_ferrari = laps["TeamName"] != TEAM_OF_INTEREST
        groups = {
            "ferrari": laps[~not_ferrari],
            "podium": laps[(laps["position_group"] == "Podium") & not_ferrari],
            "field": laps[not_ferrari],
        }
        for comp in DRY_COMPOUNDS:
            for label, g in groups.items():
                out[f"{label}_pct_{comp}"] = 100 * (g["Compound"].str.upper() == comp).mean() if len(g) else np.nan
    _, team_summary = build_pace_adjusted_delta(strategy_delta)
    f = team_summary[team_summary["TeamName"] == TEAM_OF_INTEREST]
    if len(f):
        f = f.iloc[0]
        out["ferrari_mean_pace_rank"] = f["mean_pace_rank"]
        out["ferrari_delta_vs_pace"] = f["delta_vs_pace"]
        out["ferrari_delta_vs_pace_ci_low"] = f["ci_low"]
        out["ferrari_delta_vs_pace_ci_high"] = f["ci_high"]
    return out


def print_ferrari_benchmark(benchmark):
    """Print a short Ferrari vs Podium vs Field summary so the question is answered explicitly."""
    if not benchmark:
        return
    print("\n" + "=" * 60)
    print("FERRARI BENCHMARK (vs Podium vs Field)")
    print("=" * 60)
    print(f"  Strategy delta (avg positions gained over raw pace):")
    print(f"    Ferrari = {benchmark.get('ferrari_avg_strategy_delta', np.nan):.2f}  |  Podium = {benchmark.get('podium_avg_strategy_delta', np.nan):.2f}  |  Field = {benchmark.get('field_avg_strategy_delta', np.nan):.2f}")
    if "ferrari_pct_early_pit" in benchmark and not np.isnan(benchmark.get("ferrari_pct_early_pit", np.nan)):
        print(f"  Early pit stop share (<33% race distance):")
        print(f"    Ferrari = {benchmark.get('ferrari_pct_early_pit', np.nan):.1f}%  |  Podium = {benchmark.get('podium_pct_early_pit', np.nan):.1f}%  |  Field = {benchmark.get('field_pct_early_pit', np.nan):.1f}%")
    if "ferrari_delta_vs_pace" in benchmark:
        print(f"  Pace-adjusted (vs other teams' cars at the same pace rank):")
        print(f"    Ferrari = {benchmark['ferrari_delta_vs_pace']:+.2f} positions/race "
              f"(95% CI {benchmark['ferrari_delta_vs_pace_ci_low']:+.2f} to {benchmark['ferrari_delta_vs_pace_ci_high']:+.2f}), "
              f"mean pace rank {benchmark['ferrari_mean_pace_rank']:.1f}")
    if "ferrari_pct_HARD" in benchmark:
        print("  Lap share SOFT / MEDIUM / HARD:")
        for label in ["ferrari", "podium", "field"]:
            print(f"    {label.capitalize():<8} " + " / ".join(f"{benchmark[f'{label}_pct_{c}']:.1f}%" for c in DRY_COMPOUNDS)
                  + ("  (excl. Ferrari)" if label != "ferrari" else ""))
    print("  Raw delta vs Podium is circular (podium = finished P1-P3); read the pace-adjusted line.")
    print("=" * 60)

#CHARTS (same outputs as your original analysis — Ground Effect Era 2022–2025)

#Pirelli / position group colors (match your original charts)
SOFT_COLOR = "#E8002D"
MEDIUM_COLOR = "#FFF200"
HARD_COLOR = "#CCCCCC"
PODIUM_COLOR = "#D4AF37"
TOP10_COLOR = "#4A90D9"
BACK_COLOR = "#888888"
COMPOUND_COLORS = {"SOFT": SOFT_COLOR, "MEDIUM": MEDIUM_COLOR, "HARD": HARD_COLOR}
GROUP_COLORS = {"Podium": PODIUM_COLOR, "Top10": TOP10_COLOR, "Back": BACK_COLOR}


def generate_charts(laps, degradation, strategy_delta, race_strategy, pit_windows=None):
    """Generate the 6 analysis charts from V2 data. Saves PNGs to charts/. Optionally Chart 7 if pit_windows provided."""
    os.makedirs(CHARTS_DIR, exist_ok=True)

    #Chart 1: Lap time degradation by compound (3 panels: Podium, Top10, Back)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Lap Time Degradation by Compound - Ground Effect Era (2022-2025)", fontsize=14, fontweight="bold")
    for i, group in enumerate(POSITION_GROUP_LABELS):
        ax = axes[i]
        g_laps = laps[laps["position_group"] == group]
        for compound in DRY_COMPOUNDS:
            c_laps = g_laps[g_laps["Compound"].str.upper() == compound]
            if c_laps.empty:
                continue
            by_lap = c_laps.groupby("lap_in_stint")["lap_time_sec"].mean().reset_index()
            by_lap = by_lap[by_lap["lap_in_stint"] <= 35]
            ax.plot(by_lap["lap_in_stint"], by_lap["lap_time_sec"], color=COMPOUND_COLORS[compound], linewidth=2, label=compound)
        ax.set_title(group.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Lap in Stint")
        ax.set_ylabel("Avg Lap Time (sec)" if i == 0 else "")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart1_degradation_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart1_degradation_curves.png")

    #Chart 2: Compound usage share by finishing group (100% stacked bar)
    compound_pct = laps.copy()
    compound_pct["Compound"] = compound_pct["Compound"].str.upper()
    compound_pct = compound_pct.groupby(["position_group", "Compound"], observed=True).size().reset_index(name="lap_count")
    group_totals = compound_pct.groupby("position_group", observed=True)["lap_count"].sum().reset_index(name="group_total")
    compound_pct = compound_pct.merge(group_totals, on="position_group")
    compound_pct["pct"] = 100 * compound_pct["lap_count"] / compound_pct["group_total"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Compound Usage Share by Finishing Group (2022-2025)", fontsize=13, fontweight="bold")
    bottom = {g: 0 for g in POSITION_GROUP_LABELS}
    for compound in DRY_COMPOUNDS:
        vals = []
        for g in POSITION_GROUP_LABELS:
            row = compound_pct[(compound_pct["position_group"] == g) & (compound_pct["Compound"] == compound)]
            vals.append(row["pct"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(POSITION_GROUP_LABELS, vals, bottom=[bottom[g] for g in POSITION_GROUP_LABELS], color=COMPOUND_COLORS[compound], label=compound, edgecolor="white", linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[POSITION_GROUP_LABELS[j]] + val / 2, f"{val:.1f}%", ha="center", va="center", fontsize=9, color="black" if compound == "MEDIUM" else "white", fontweight="bold")
            bottom[POSITION_GROUP_LABELS[j]] += val
    ax.set_ylabel("% of Race Laps")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart2_compound_usage.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart2_compound_usage.png")

    #Chart 3: MEDIUM tire degradation rate by season
    med_deg = degradation[degradation["compound"].str.upper() == "MEDIUM"].groupby("season")["lap_time_degradation_slope"].mean()
    if len(med_deg) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("MEDIUM Tire Degradation Rate by Season (Ground Effect Era)", fontsize=13, fontweight="bold")
        #Report convention: degradation slope as negative sec/lap (slower over stint)
        vals = -med_deg.reindex(SEASONS).dropna()
        colors = [TOP10_COLOR] * len(vals)
        bars = ax.bar(vals.index.astype(str), vals.values, color=colors)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.01 if v < 0 else v + 0.01, f"{v:.4f} sec/lap", ha="center", va="top" if v < 0 else "bottom", fontsize=9)
        ax.set_ylabel("Degradation Slope (sec per lap)")
        ax.set_xlabel("Season")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "chart3_deg_by_season.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved chart3_deg_by_season.png")

    #Chart 4: Era strategy delta — top 10 drivers (avg positions gained over raw pace)
    if len(strategy_delta) > 0:
        by_driver = strategy_delta.groupby("Driver").agg(avg_strategy_delta=("strategy_delta", "mean")).reset_index()
        by_driver = by_driver.sort_values("avg_strategy_delta", ascending=False).head(10)
        if len(by_driver) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Era Strategy Delta: Who Finished Better Than Their Raw Pace? (2022-2025)", fontsize=13, fontweight="bold")
            ax.barh(by_driver["Driver"], by_driver["avg_strategy_delta"], color=TOP10_COLOR, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=1, linestyle="--")
            ax.set_xlabel("Avg Positions Gained Over Raw Pace Per Race")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, "chart4_strategy_delta.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved chart4_strategy_delta.png")

    #Chart 5: Tire degradation heatmap (compound x degradation_type)
    #Derive degradation_type per race: use SOFT slope tertiles → high/medium/low-deg
    if len(degradation) > 0 and "race" in degradation.columns:
        soft_by_race = degradation[degradation["compound"].str.upper() == "SOFT"].groupby("race")["lap_time_degradation_slope"].mean()
        if len(soft_by_race) >= 3:
            q1, q2 = soft_by_race.quantile(0.33), soft_by_race.quantile(0.67)
            def deg_type(slope):
                if slope <= q1:
                    return "high-deg"
                if slope <= q2:
                    return "medium-deg"
                return "low-deg"
            race_to_type = soft_by_race.apply(deg_type).to_dict()
            deg_with_type = degradation.copy()
            deg_with_type["degradation_type"] = deg_with_type["race"].map(race_to_type)
            deg_with_type["compound"] = deg_with_type["compound"].str.upper()
            pivot = deg_with_type.groupby(["compound", "degradation_type"])["lap_time_degradation_slope"].mean().unstack(fill_value=np.nan)
            #Order: high-deg, medium-deg, low-deg (cols); HARD, MEDIUM, SOFT (rows)
            for col in ["high-deg", "medium-deg", "low-deg"]:
                if col not in pivot.columns:
                    pivot[col] = np.nan
            pivot = pivot[["high-deg", "medium-deg", "low-deg"]]
            pivot = pivot.reindex(["HARD", "MEDIUM", "SOFT"])
            if not pivot.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.set_title("Tire Degradation Rate Heatmap - sec per lap (2022-2025)", fontsize=13, fontweight="bold")
                #Show as negative for "time lost per lap" convention
                im = ax.imshow(-pivot.values, cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=9)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=9)
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        v = pivot.values[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9, fontweight="bold")
                plt.colorbar(im, ax=ax, label="sec per lap (negative = time lost)")
                plt.tight_layout()
                plt.savefig(os.path.join(CHARTS_DIR, "chart5_deg_heatmap.png"), dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved chart5_deg_heatmap.png")

    #Chart 6: Raw pace rank vs finish position (points above diagonal = strategy gain)
    if len(strategy_delta) > 0 and "position_group" in strategy_delta.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("Raw Pace Rank vs Finish Position (2022-2025)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Pace Rank in Race (1 = fastest avg lap)")
        ax.set_ylabel("Finish Position (1 = winner)")
        for group in POSITION_GROUP_LABELS:
            g = strategy_delta[strategy_delta["position_group"] == group]
            if g.empty:
                continue
            ax.scatter(g["pace_rank"], g["finish_position"], alpha=0.5, s=25, c=GROUP_COLORS[group], label=group.lower())
        #Highlight Ferrari so the question "where does Ferrari sit?" is answered visually
        if "TeamName" in strategy_delta.columns:
            ferrari_sd = strategy_delta[strategy_delta["TeamName"] == TEAM_OF_INTEREST]
            if len(ferrari_sd) > 0:
                ax.scatter(ferrari_sd["pace_rank"], ferrari_sd["finish_position"], s=80, marker="*", c="#DC143C", edgecolors="black", linewidths=0.8, label=TEAM_OF_INTEREST, zorder=5)
        max_val = max(strategy_delta["pace_rank"].max(), strategy_delta["finish_position"].max()) if len(strategy_delta) else 20
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="pace = result")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_val + 0.5)
        ax.set_ylim(0.5, max_val + 0.5)
        ax.invert_yaxis()
        plt.figtext(0.5, 0.01, "Points above diagonal = strategy gain", ha="center", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(os.path.join(CHARTS_DIR, "chart6_pace_vs_finish.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved chart6_pace_vs_finish.png")

    #Chart 7: Ferrari vs Podium vs Field (answers "what should Ferrari change?")
    if pit_windows is not None and len(pit_windows) > 0 and "TeamName" in strategy_delta.columns and (strategy_delta["TeamName"] == TEAM_OF_INTEREST).any() and "TeamName" in pit_windows.columns:
        ferrari_sd = strategy_delta.loc[strategy_delta["TeamName"] == TEAM_OF_INTEREST, "strategy_delta"].mean()
        podium_sd = strategy_delta.loc[strategy_delta["position_group"] == "Podium", "strategy_delta"].mean()
        field_sd = strategy_delta["strategy_delta"].mean()
        fp = pit_windows[pit_windows["TeamName"] == TEAM_OF_INTEREST]
        pp = pit_windows[pit_windows["position_group"] == "Podium"]
        ferrari_early = 100 * (fp["pit_window"] == "early").mean() if len(fp) else 0
        podium_early = 100 * (pp["pit_window"] == "early").mean() if len(pp) else 0
        field_early = 100 * (pit_windows["pit_window"] == "early").mean()
        if np.isfinite(ferrari_sd) and np.isfinite(field_sd):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle("Ferrari vs Podium vs Field — Where to Gain (2022-2025)", fontsize=13, fontweight="bold")
            ax1 = axes[0]
            ax1.bar(["Ferrari", "Podium", "Field"], [ferrari_sd, podium_sd, field_sd], color=["#DC143C", PODIUM_COLOR, TOP10_COLOR], edgecolor="white")
            ax1.axhline(0, color="gray", linewidth=0.8)
            ax1.set_ylabel("Avg strategy delta (pos. gained over raw pace)")
            ax1.set_title("Strategy delta")
            ax1.grid(axis="y", alpha=0.3)
            ax2 = axes[1]
            ax2.bar(["Ferrari", "Podium", "Field"], [ferrari_early, podium_early, field_early], color=["#DC143C", PODIUM_COLOR, TOP10_COLOR], edgecolor="white")
            ax2.set_ylabel("% of pit stops in early window (<33%)")
            ax2.set_title("Early pit share")
            ax2.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, "chart7_ferrari_vs_podium_vs_field.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved chart7_ferrari_vs_podium_vs_field.png")

    print(f"  All charts written to {CHARTS_DIR}/")

#PHASE 6 — ACT (top 3 recommendations)

def plot_team_pace_adjusted(team_summary):
    """Positions gained per race vs other teams' cars at the same pace rank, with 95% CIs."""
    if team_summary is None or len(team_summary) == 0:
        return
    d = team_summary.sort_values("delta_vs_pace")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#DC143C" if t == TEAM_OF_INTEREST else TOP10_COLOR for t in d["TeamName"]]
    ax.barh(d["TeamName"], d["delta_vs_pace"], color=colors, alpha=0.85,
            xerr=[d["delta_vs_pace"] - d["ci_low"], d["ci_high"] - d["delta_vs_pace"]],
            error_kw={"elinewidth": 1, "capsize": 3})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Positions gained per race vs cars at the same race-pace rank (95% CI, races resampled)")
    ax.set_title("Pace-adjusted strategy delta by team, 2022-2024", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "chart9_team_pace_adjusted_delta.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved chart9_team_pace_adjusted_delta.png")


def print_recommendations(degradation, pit_windows, strategy_delta, race_strategy, ferrari_benchmark=None):
    """Recommendations from Ferrari's perspective. Every claim is computed from the data.

    AUDITED 2026-10-21: the old text told Ferrari to "commit to earlier stops to
    match podium" while printing that Ferrari already pits early MORE often,
    and the SOFT stint lengths (12-14 / 18-22 laps) and "SOFT -> HARD" advice
    were fixed strings, not outputs.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS (from Ferrari's perspective vs full-time field)")
    print("=" * 60)
    b = ferrari_benchmark or {}

    #1. Is Ferrari's gap strategy or pace?
    if "ferrari_delta_vs_pace" in b:
        lo, hi = b["ferrari_delta_vs_pace_ci_low"], b["ferrari_delta_vs_pace_ci_high"]
        verdict = ("better than" if lo > 0 else "worse than" if hi < 0 else "no different from")
        print("\n1. THE GAP IS PACE, NOT STRATEGY")
        print(f"   Ferrari's cars average pace rank {b['ferrari_mean_pace_rank']:.1f}. Against other teams' cars at the")
        print(f"   same pace rank, Ferrari finishes {b['ferrari_delta_vs_pace']:+.2f} positions/race "
              f"(95% CI {lo:+.2f} to {hi:+.2f}): {verdict} the field.")
        print("   Strategy is converting the pace Ferrari has. Closing the gap to the front needs race pace.")

    #2. Pit timing
    if "ferrari_pct_early_pit" in b and not np.isnan(b.get("ferrari_pct_early_pit", np.nan)):
        f, pod, fld = b["ferrari_pct_early_pit"], b["podium_pct_early_pit"], b["field_pct_early_pit"]
        print("\n2. PIT TIMING — leave it")
        print(f"   Early-window share (<33% of race distance): Ferrari {f:.1f}%, podium {pod:.1f}%, field {fld:.1f}%.")
        print(f"   Ferrari is within {abs(f - pod):.1f} points of podium finishers; moving stops earlier has no support here.")

    #3. Compound mix
    if "ferrari_pct_HARD" in b:
        fh, ph, flh = b["ferrari_pct_HARD"], b["podium_pct_HARD"], b["field_pct_HARD"]
        print("\n3. COMPOUND MIX — Ferrari runs like the field")
        print(f"   HARD lap share: Ferrari {fh:.1f}%, rest of field {flh:.1f}%, podium finishers (excl. Ferrari) {ph:.1f}%.")
        print("   Podium cars run less HARD than everyone, Ferrari included. That's likely a result of")
        print("   running at the front (track position, fewer stops), not a choice Ferrari can copy.")
    if len(degradation) > 0:
        by_compound = degradation.groupby("compound")["lap_time_degradation_slope"].median()
        print("   Median degradation slope (sec/lap, fuel burn-off NOT removed, so it understates wear): "
              + ", ".join(f"{c} {by_compound[c]:+.3f}" for c in DRY_COMPOUNDS if c in by_compound))

    #4. Where strategy moves results
    if len(strategy_delta) > 0:
        by_race = strategy_delta.groupby("race")["strategy_delta"].std().sort_values(ascending=False)
        if len(by_race) > 0:
            print("\n4. CIRCUITS WHERE FINISHING ORDER STRAYS MOST FROM PACE")
            print("   " + ", ".join(f"{r} ({v:.1f})" for r, v in by_race.head(3).items())
                  + " — std dev of strategy delta. Pre-race planning matters most here.")

    print("\n" + "=" * 60)

#PHASE 4b — PREDICTIVE MODELING
#Goal: Random Forest Regressor that predicts finish position from strategy
#features available after the race. Answers "which strategy factors matter
#most?" so Ferrari can prioritize what to optimize.
#
#Features engineered from existing pipeline outputs:
#  avg_lap_sec        — mean race pace (raw speed proxy)
#  delta_to_leader    — gap to fastest driver's average lap
#  first_pit_lap      — lap number of first pit stop
#  pit_count          — total pit stops made
#  avg_deg_slope      — mean tire degradation slope across all stints
#  soft_usage_pct     — fraction of clean laps run on SOFT compound
#  stint_count        — number of separate stints
#
#Target: finish_position (numeric, lower = better)
#Model:  RandomForestRegressor, 200 estimators, 80/20 train-test split

def predict_finish_position(laps, degradation, strategy_delta, race_strategy, pit_summary):
    """Build and evaluate a Random Forest model predicting finish position from
    strategy features. Saves feature importance chart and predictions CSV to
    the tableau_export folder."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    print("\n" + "=" * 60)
    print("PHASE 4b — PREDICTIVE MODELING: Predict Finish Position")
    print("=" * 60)

    try:
        #Step 1: Base pace features per driver per race from strategy_delta
        pace_features = (
            strategy_delta[["season", "race", "Driver", "avg_lap_sec",
                             "delta_to_leader", "finish_position", "TeamName"]]
            .copy()
            .dropna(subset=["finish_position"])
        )

        #Step 2: Join pit stop features (first pit lap, pit count) from race_strategy
        if "first_pit_lap" in race_strategy.columns:
            keep = ["season", "race", "Driver", "first_pit_lap", "pit_count"]
            if "GridPosition" in race_strategy.columns:
                keep.append("GridPosition")
            pit_cols = race_strategy[keep].copy()
            pace_features = pace_features.merge(pit_cols, on=["season", "race", "Driver"], how="left")

        #Step 3: Average degradation slope per driver per race
        if len(degradation) > 0:
            avg_deg = (
                degradation.groupby(["season", "race", "Driver"])["lap_time_degradation_slope"]
                .mean()
                .reset_index()
                .rename(columns={"lap_time_degradation_slope": "avg_deg_slope"})
            )
            pace_features = pace_features.merge(avg_deg, on=["season", "race", "Driver"], how="left")

        #Step 4: SOFT usage percentage and stint count from clean laps
        if "Compound" in laps.columns:
            soft_usage = (
                laps.groupby(["season", "race", "Driver"])
                ["Compound"].apply(lambda c: (c.str.upper() == "SOFT").mean())
                .reset_index(name="soft_usage_pct")
            )
            stint_count = (
                laps.groupby(["season", "race", "Driver"])["Stint"]
                .nunique()
                .reset_index(name="stint_count")
            )
            pace_features = pace_features.merge(soft_usage, on=["season", "race", "Driver"], how="left")
            pace_features = pace_features.merge(stint_count, on=["season", "race", "Driver"], how="left")

        #Step 5: Select feature columns — drop rows with any NaN in features or target
        feature_cols = [c for c in [
            "avg_lap_sec", "delta_to_leader", "first_pit_lap",
            "pit_count", "avg_deg_slope", "soft_usage_pct", "stint_count"
        ] if c in pace_features.columns]

        model_df = pace_features.dropna(subset=feature_cols + ["finish_position"]).reset_index(drop=True)
        print(f"\n  Modeling dataset: {len(model_df)} driver-race records, {len(feature_cols)} features")

        if len(model_df) < 20:
            print("  Insufficient data for reliable modeling (need >= 20 records). Skipping.")
            return

        X = model_df[feature_cols]
        y = model_df["finish_position"]

        #Step 6: 80/20 train-test split with fixed seed for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Step 7: Fit Random Forest — handles non-linear feature interactions well
        #n_estimators=200 gives stable importance estimates; n_jobs=-1 uses all cores
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        #Step 8: Evaluate on held-out test set
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\n  Random Forest Results:")
        print(f"    R2   (variance in finish position explained): {r2:.4f}")
        print(f"    RMSE (avg position prediction error)        : {rmse:.2f} places")

        #AUDITED 2026-10-21: a random 80/20 split puts drivers from the same race in both
        #train and test, so the model partly learns each race's field and R2 is flattered
        #(0.73). Grouped CV holds out whole races. A pace + grid baseline shows how much
        #the strategy features (pits, compounds, degradation) actually add.
        from sklearn.model_selection import GroupKFold, cross_val_predict
        groups = model_df["season"].astype(str) + " " + model_df["race"]
        strategy_cols = [c for c in ["first_pit_lap", "pit_count", "avg_deg_slope", "soft_usage_pct", "stint_count"]
                         if c in feature_cols]
        base_cols = ["delta_to_leader"] + (["GridPosition"] if "GridPosition" in model_df.columns else [])
        specs = {
            "original features": feature_cols,
            "pace + grid (baseline)": base_cols,
            "pace + grid + strategy": base_cols + strategy_cols,
        }
        grouped = []
        for name, cols in specs.items():
            d = model_df.dropna(subset=cols)
            g = groups.loc[d.index]
            pred = cross_val_predict(
                RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=5),
                d[cols], d["finish_position"], groups=g, cv=GroupKFold(n_splits=5),
            )
            grouped.append({"model": name, "features": ", ".join(cols), "n": len(d),
                            "grouped_cv_r2": r2_score(d["finish_position"], pred),
                            "grouped_cv_rmse": float(np.sqrt(mean_squared_error(d["finish_position"], pred)))})
        grouped = pd.DataFrame(grouped)
        print("\n  Race-grouped 5-fold CV (whole races held out):")
        for _, row in grouped.iterrows():
            print(f"    {row['model']:<24} R2={row['grouped_cv_r2']:.3f}  RMSE={row['grouped_cv_rmse']:.2f}")
        gain = (grouped.loc[grouped["model"] == "pace + grid + strategy", "grouped_cv_r2"].iloc[0]
                - grouped.loc[grouped["model"] == "pace + grid (baseline)", "grouped_cv_r2"].iloc[0])
        print(f"    Strategy features add {gain:+.3f} R2 on top of race pace and grid position.")

        #Step 9: Feature importances — higher = more predictive of finish position
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        print(f"\n  Feature Importances (higher = more predictive of finish position):")
        for _, row in importance_df.iterrows():
            bar = "|" * int(row["Importance"] * 40)
            print(f"    {row['Feature']:<22}: {row['Importance']:.4f}  {bar}")

        #Step 10: Save feature importance chart to tableau_export folder
        out_dir = os.path.join(_SCRIPT_DIR, "tableau_export")
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(importance_df)))
        ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1],
                color=colors[::-1], edgecolor="black", alpha=0.85)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            f"Predictive Model: Features Driving F1 Finish Position\n"
            f"(Random Forest  R2={r2:.3f}, RMSE={rmse:.2f} places)",
            fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        chart_path = os.path.join(out_dir, "predictive_model_feature_importance.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\n  Saved: predictive_model_feature_importance.png")

        #Step 11: Save predictions vs actuals and feature importance table to CSV
        pred_out = model_df[["finish_position"]].copy().reset_index(drop=True)
        pred_out["predicted_position"] = rf.predict(X).round(1)
        pred_out["residual"] = pred_out["finish_position"] - pred_out["predicted_position"]
        pred_out.to_csv(os.path.join(out_dir, "predictive_model_predictions.csv"), index=False)
        importance_df.to_csv(os.path.join(out_dir, "predictive_model_importance.csv"), index=False)
        grouped.to_csv(os.path.join(out_dir, "predictive_model_grouped_cv.csv"), index=False)
        print(f"  Saved: predictive_model_predictions.csv")
        print(f"  Saved: predictive_model_importance.csv")

    except Exception as e:
        print(f"  Predictive modeling FAILED: {e!r}")
        raise

    print("=" * 60)


#MAIN PIPELINE

def main():
    print("PHASE 1 — ASK: Problem defined (see docstring at top).")
    import sys
    have_raw = all(os.path.exists(os.path.join(RAW_DIR, f)) for f in ["raw_laps.csv", "raw_results.csv", "raw_weather.csv"])
    if have_raw and "--refresh" not in sys.argv:
        print("\nPHASE 2 — PREPARE: Loading committed raw_data/ CSVs (use --refresh to re-download via FastF1)...")
        laps_raw, results_raw, weather_raw = load_raw_csvs()
    else:
        print("\nPHASE 2 — PREPARE: Loading race data from FastF1 / f1_cache...")
        laps_raw, results_raw, weather_raw = prepare_season_data(SEASONS)

    print("\n  Filtering to drivers who participated in a majority of races (Ferrari perspective)...")
    laps_raw, results_raw = filter_to_majority_drivers(laps_raw, results_raw, MAJORITY_THRESHOLD)

    print("\n  Writing raw data to SQL (single source of truth)...")
    load_raw_to_sql(laps_raw, results_raw, weather_raw)

    print("\nPHASE 3 — PROCESS: Cleaning and loading into SQL...")
    laps_clean, results_clean = clean_laps(laps_raw, results_raw)
    laps_clean = build_lap_in_stint(laps_clean)
    stint_summary, pit_summary, race_strategy, results_with_groups = load_to_sql_and_summarize(
        laps_clean, results_raw, pit_source=laps_raw
    )  #results_raw here is already filtered to majority drivers

    #Persist cleaned laps with position_group and TeamName for analysis (merge from results)
    merge_cols = ["season", "race", "Driver", "position_group"]
    if "TeamName" in results_with_groups.columns:
        merge_cols.append("TeamName")
    results_for_merge = results_with_groups[merge_cols]
    laps_clean = laps_clean.merge(results_for_merge, on=["season", "race", "Driver"], how="left")

    print("\nPHASE 4 — ANALYZE: Building degradation, strategy delta, pit windows...")
    degradation = build_degradation_curves(laps_clean)
    strategy_delta = build_strategy_delta(laps_clean, results_with_groups)
    pit_windows = build_pit_window_analysis(laps_raw, results_with_groups, race_strategy)  # raw laps still have pit stops

    #Save degradation to DB
    conn = sqlite3.connect(OUTPUT_DB)
    degradation.to_sql("tire_degradation", conn, if_exists="replace", index=False)
    conn.close()

    print("\nPHASE 4b — PREDICTIVE MODELING: Random Forest finish position predictor...")
    predict_finish_position(laps_clean, degradation, strategy_delta, race_strategy, pit_summary)

    print("\nPHASE 5 — SHARE: Exporting Tableau-ready CSVs...")
    ferrari_benchmark = export_tableau_files(
        laps_clean, degradation, race_strategy, strategy_delta, pit_windows, stint_summary
    )
    if ferrari_benchmark:
        print_ferrari_benchmark(ferrari_benchmark)

    print("\n  Generating charts (same style as your original outputs)...")
    generate_charts(laps_clean, degradation, strategy_delta, race_strategy, pit_windows)
    plot_team_pace_adjusted(build_pace_adjusted_delta(strategy_delta)[1])

    print("\nPHASE 6 — ACT: Recommendations")
    print_recommendations(degradation, pit_windows, strategy_delta, race_strategy, ferrari_benchmark)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()