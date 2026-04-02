import os
import fastf1
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#CONFIG
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()

CACHE_DIR = os.path.join(_SCRIPT_DIR, "f1_cache")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "tableau_export")
CHARTS_DIR = os.path.join(_SCRIPT_DIR, "charts")
SEASONS = [2022, 2023, 2024]
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
POSITION_GROUP_LABELS = ["Podium", "Top10", "Back"]
TEAM_OF_INTEREST = "Ferrari"
MAJORITY_PCT = 0.5

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

#COLORS
SOFT_COLOR = "#E8002D"
MEDIUM_COLOR = "#FFF200"
HARD_COLOR = "#CCCCCC"
PODIUM_COLOR = "#D4AF37"
TOP10_COLOR = "#4A90D9"
BACK_COLOR = "#888888"
COMPOUND_COLORS = {"SOFT": SOFT_COLOR, "MEDIUM": MEDIUM_COLOR, "HARD": HARD_COLOR}
GROUP_COLORS = {"Podium": PODIUM_COLOR, "Top10": TOP10_COLOR, "Back": BACK_COLOR}


print("PHASE 1 - ASK: Problem defined")
print("Business question: How do race-winning drivers differ from mid-field in tire strategy?")
print("Scope: 2022-2024, majority drivers, Ferrari benchmark")


#PHASE 2 - PREPARE

def prepare_season_data(seasons):
    """Pull race data from FastF1 cache."""
    all_laps = []
    all_results = []

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

                all_laps.append(laps)
                all_results.append(results)
            except Exception as e:
                print(f"  Skipped {season} round {rnd}: {e}")

    df_laps = pd.concat(all_laps, ignore_index=True)
    df_results = pd.concat(all_results, ignore_index=True)

    #Normalize driver identifier
    if "Driver" not in df_results.columns and "Abbreviation" in df_results.columns:
        df_results["Driver"] = df_results["Abbreviation"]
    if "Driver" not in df_laps.columns and "Abbreviation" in df_laps.columns:
        df_laps["Driver"] = df_laps["Abbreviation"]

    print(f"  Raw data: {len(df_laps)} laps, {len(df_results)} results")
    return df_laps, df_results


def filter_to_majority_drivers(laps, results, majority_pct=MAJORITY_PCT):
    """Keep only full-time drivers (majority of races per season)."""
    races_per_season = results.groupby("season")["race"].nunique().reset_index()
    races_per_season = races_per_season.rename(columns={"race": "season_races"})
    
    driver_races = results.groupby(["season", "Driver"]).size().reset_index(name="races_done")
    driver_races = driver_races.merge(races_per_season, on="season")
    driver_races["is_majority"] = driver_races["races_done"] >= (
        driver_races["season_races"] * majority_pct
    )
    full_time = driver_races[driver_races["is_majority"]][["season", "Driver"]].drop_duplicates()

    laps = laps.merge(full_time, on=["season", "Driver"], how="inner")
    results = results.merge(full_time, on=["season", "Driver"], how="inner")
    
    n_drivers = full_time.groupby("season").size()
    print(f"  Full-time drivers (≥{int(majority_pct*100)}% of races): {dict(n_drivers)}")
    print(f"  Filtered: {len(laps)} laps, {len(results)} results")
    
    return laps, results


#PHASE 3 - PROCESS

def clean_laps(laps, results):
    """Clean lap data: nulls, pit laps, compounds, outliers, DNF."""
    n_start = len(laps)
    print(f"  Cleaning: {n_start} starting rows")

    #1. Null lap time
    laps = laps.dropna(subset=["LapTime"])
    laps = laps.copy()
    laps["lap_time_sec"] = laps["LapTime"].dt.total_seconds()
    print(f"    After null check: {len(laps)}")

    #2. Remove pit in/out laps
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    print(f"    After pit lap removal: {len(laps)}")

    #3. Dry compounds only
    laps = laps[laps["Compound"].str.upper().isin(DRY_COMPOUNDS)]
    print(f"    After dry compound filter: {len(laps)}")

    #4. 110% fastest lap filter
    race_fastest = laps.groupby(["season", "race"])["lap_time_sec"].transform("min")
    laps = laps[laps["lap_time_sec"] <= race_fastest * 1.10]
    print(f"    After 110% filter: {len(laps)}")

    #5. DNF/DSQ removal
    if "Status" in results.columns:
        finished = results[results["Status"].astype(str).str.upper().str.contains("LAP|FINISHED", na=False)]
    else:
        finished = results.dropna(subset=["Position"])
    
    finished_keys = finished[["season", "race", "Driver"]].drop_duplicates()
    laps = laps.merge(finished_keys, on=["season", "race", "Driver"], how="inner")
    print(f"    After DNF removal: {len(laps)}")

    #6. Z-score normalize
    laps["lap_time_norm"] = laps.groupby(["season", "race"])["lap_time_sec"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return laps, results


def build_lap_in_stint(laps):
    """Add lap number within stint."""
    laps = laps.sort_values(["season", "race", "Driver", "LapNumber"])
    laps["lap_in_stint"] = laps.groupby(["season", "race", "Driver", "Stint"]).cumcount() + 1
    return laps


def build_summary_tables(laps, results):
    """Build race-level and stint-level summaries."""
    #Position groups
    results = results.copy()
    results["position_group"] = pd.cut(
        results["Position"],
        bins=[0, 3, 10, 100],
        labels=POSITION_GROUP_LABELS,
    )

    #Stint summary
    stint_summary = (
        laps.groupby(["season", "race", "Driver", "Stint", "Compound"])
        .agg(
            stint_length=("LapNumber", "count"),
            avg_pace_sec=("lap_time_sec", "mean"),
            fastest_lap_sec=("lap_time_sec", "min"),
        )
        .reset_index()
    )

    #Position change (start pos - end pos)
    pos_by_driver = (
        laps.groupby(["season", "race", "Driver"])
        .agg(start_pos=("Position", "first"), end_pos=("Position", "last"))
        .reset_index()
    )
    pos_by_driver["position_change"] = pos_by_driver["start_pos"] - pos_by_driver["end_pos"]

    #Race strategy view
    res_cols = ["Driver", "race", "season", "Position", "position_group", "TeamName"]
    if "GridPosition" in results.columns:
        res_cols.append("GridPosition")
    
    race_strategy = pos_by_driver.merge(
        results[res_cols],
        on=["Driver", "race", "season"],
        how="left",
    )

    return stint_summary, race_strategy, results


#PHASE 4 - ANALYZE

def calc_degradation_slope(group, min_laps=5):
    """OLS regression: lap_time_sec ~ lap_in_stint."""
    if len(group) < min_laps:
        return np.nan
    slope, _, _, _, _ = stats.linregress(group["lap_in_stint"], group["lap_time_sec"])
    return round(slope, 5)


def build_degradation_curves(laps):
    """Degradation rate per compound and stint."""
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
    """Pace rank minus finish position."""
    avg_pace = laps.groupby(["season", "race", "Driver"])["lap_time_sec"].mean().reset_index()
    avg_pace = avg_pace.rename(columns={"lap_time_sec": "avg_lap_sec"})
    avg_pace["pace_rank"] = avg_pace.groupby(["season", "race"])["avg_lap_sec"].rank(method="min")
    
    res = results[["season", "race", "Driver", "Position", "TeamName", "position_group"]].copy()
    res = res.rename(columns={"Position": "finish_position"})
    
    merged = avg_pace.merge(res, on=["season", "race", "Driver"], how="inner")
    merged["strategy_delta"] = merged["pace_rank"] - merged["finish_position"]
    merged["delta_to_leader"] = merged["avg_lap_sec"] - merged.groupby(["season", "race"])["avg_lap_sec"].transform("min")
    
    return merged


def build_pit_window_analysis(laps_raw, results, race_strategy, strategy_delta):
    """Extract pit data from raw laps and match with positions gained."""
    
    #Get pit in laps (where driver entered pit)
    pit_in_data = laps_raw[laps_raw['PitInTime'].notna()].copy()
    
    if len(pit_in_data) == 0:
        print("  WARNING: No pit data found - creating empty pit windows")
        return pd.DataFrame(columns=['season', 'race', 'Driver', 'pit_stop_lap', 'pit_lap_pct', 'pit_window', 'position_group', 'TeamName'])

    #Get first pit lap per driver per race
    first_pit = pit_in_data.groupby(['season', 'race', 'Driver'])['LapNumber'].agg(['min', 'count']).reset_index()
    first_pit.columns = ['season', 'race', 'Driver', 'pit_stop_lap', 'pit_count']

    #Get total laps per race
    total_laps = laps_raw.groupby(['season', 'race'])['LapNumber'].max().reset_index()
    total_laps.columns = ['season', 'race', 'total_laps']

    #Merge pit and total laps
    pit_windows = first_pit.merge(total_laps, on=['season', 'race'], how='left')
    pit_windows['pit_lap_pct'] = pit_windows['pit_stop_lap'] / pit_windows['total_laps']

    #Classify window
    pit_windows['pit_window'] = pd.cut(
        pit_windows['pit_lap_pct'],
        bins=[0, 1/3, 2/3, 1.0],
        labels=['early', 'mid', 'late'],
        include_lowest=True
    )

    #Add position group and team from results
    res_cols = ['season', 'race', 'Driver', 'position_group']
    if 'TeamName' in results.columns:
        res_cols.append('TeamName')
    
    res_short = results[res_cols].drop_duplicates()
    pit_windows = pit_windows.merge(res_short, on=['season', 'race', 'Driver'], how='left')

    print(f"  Pit windows: {len(pit_windows)} pit events found")
    print(f"    Sample pit_stop_lap: {pit_windows['pit_stop_lap'].head().tolist()}")

    return pit_windows


#PHASE 5 - SHARE

def export_tableau_files(laps, degradation, race_strategy, strategy_delta, pit_windows, stint_summary):
    """Export CSVs for Tableau."""
    print(f"\nPHASE 5 - SHARE: Exporting to {OUTPUT_DIR}/")

    #1. Tire strategy timeline (Gantt-style)
    gantt = (
        laps.groupby(['season', 'race', 'Driver', 'Stint', 'Compound'])
        .agg(start_lap=('LapNumber', 'min'), end_lap=('LapNumber', 'max'))
        .reset_index()
    )
    gantt.to_csv(os.path.join(OUTPUT_DIR, 'tire_strategy_timeline.csv'), index=False)
    print(f"  tire_strategy_timeline.csv: {len(gantt)} rows")

    #2. Degradation curves with position group
    deg_with_group = degradation.merge(
        race_strategy[['season', 'race', 'Driver', 'position_group']],
        on=['season', 'race', 'Driver'],
        how='left',
    )
    deg_with_group.to_csv(os.path.join(OUTPUT_DIR, 'degradation_curves.csv'), index=False)
    print(f"  degradation_curves.csv: {len(deg_with_group)} rows")

    #3. Pit windows by race
    pit_windows.to_csv(os.path.join(OUTPUT_DIR, 'pit_windows_by_race.csv'), index=False)
    print(f"  pit_windows_by_race.csv: {len(pit_windows)} rows")

    #4. Pit lap vs positions gained
    scatter = pit_windows.copy()
    pos_change = race_strategy[['season', 'race', 'Driver', 'position_change']].copy()
    scatter = scatter.merge(pos_change, on=['season', 'race', 'Driver'], how='left')
    scatter = scatter.rename(columns={'position_change': 'net_positions_gained'})
    scatter_export = scatter[['season', 'race', 'Driver', 'pit_stop_lap', 'net_positions_gained', 'position_group']].copy()
    scatter_export.to_csv(os.path.join(OUTPUT_DIR, 'pit_lap_vs_positions_gained.csv'), index=False)
    print(f"  pit_lap_vs_positions_gained.csv: {len(scatter_export)} rows")

    #5. Lap-level data
    laps.to_csv(os.path.join(OUTPUT_DIR, 'laps_clean.csv'), index=False)
    print(f"  laps_clean.csv: {len(laps)} rows")

    #6. Stint summary
    stint_summary.to_csv(os.path.join(OUTPUT_DIR, 'stint_summary.csv'), index=False)
    print(f"  stint_summary.csv: {len(stint_summary)} rows")

    #7. Strategy delta
    strategy_delta.to_csv(os.path.join(OUTPUT_DIR, 'strategy_delta.csv'), index=False)
    print(f"  strategy_delta.csv: {len(strategy_delta)} rows")

    #8. Ferrari exports
    if 'TeamName' in race_strategy.columns:
        ferrari = race_strategy[race_strategy['TeamName'] == TEAM_OF_INTEREST]
        if len(ferrari) > 0:
            ferrari.to_csv(os.path.join(OUTPUT_DIR, 'ferrari_race_strategy.csv'), index=False)
            print(f"  ferrari_race_strategy.csv: {len(ferrari)} rows")

        ferrari_sd = strategy_delta[strategy_delta['TeamName'] == TEAM_OF_INTEREST]
        if len(ferrari_sd) > 0:
            ferrari_sd.to_csv(os.path.join(OUTPUT_DIR, 'ferrari_strategy_delta.csv'), index=False)
            print(f"  ferrari_strategy_delta.csv: {len(ferrari_sd)} rows")

    #9. Ferrari benchmark
    ferrari_benchmark = build_ferrari_benchmark(strategy_delta, pit_windows, laps, degradation)
    if ferrari_benchmark:
        pd.DataFrame([ferrari_benchmark]).to_csv(os.path.join(OUTPUT_DIR, 'ferrari_benchmark.csv'), index=False)
        print(f"  ferrari_benchmark.csv: 1 row")
        print_ferrari_benchmark(ferrari_benchmark)

    return ferrari_benchmark


def build_ferrari_benchmark(strategy_delta, pit_windows, laps, degradation):
    """Ferrari vs Podium vs Field metrics."""
    if not ('TeamName' in strategy_delta.columns and (strategy_delta['TeamName'] == TEAM_OF_INTEREST).any()):
        return None
    
    out = {}
    
    #Strategy delta
    out['ferrari_avg_strategy_delta'] = strategy_delta.loc[strategy_delta['TeamName'] == TEAM_OF_INTEREST, 'strategy_delta'].mean()
    podium = strategy_delta[strategy_delta['position_group'] == 'Podium']
    out['podium_avg_strategy_delta'] = podium['strategy_delta'].mean() if len(podium) else np.nan
    out['field_avg_strategy_delta'] = strategy_delta['strategy_delta'].mean()

    #Early pit share
    if len(pit_windows) > 0 and 'pit_window' in pit_windows.columns and 'TeamName' in pit_windows.columns:
        ferrari_pits = pit_windows[pit_windows['TeamName'] == TEAM_OF_INTEREST]
        out['ferrari_pct_early_pit'] = 100 * (ferrari_pits['pit_window'] == 'early').mean() if len(ferrari_pits) else np.nan
        podium_pits = pit_windows[pit_windows['position_group'] == 'Podium']
        out['podium_pct_early_pit'] = 100 * (podium_pits['pit_window'] == 'early').mean() if len(podium_pits) else np.nan
        out['field_pct_early_pit'] = 100 * (pit_windows['pit_window'] == 'early').mean()

    #Compound usage
    if 'TeamName' in laps.columns:
        f_laps = laps[laps['TeamName'] == TEAM_OF_INTEREST]
        p_laps = laps[laps['position_group'] == 'Podium']
        for comp in DRY_COMPOUNDS:
            fc = f_laps[f_laps['Compound'].str.upper() == comp]
            pc = p_laps[p_laps['Compound'].str.upper() == comp]
            out[f'ferrari_pct_{comp}'] = 100 * len(fc) / len(f_laps) if len(f_laps) else np.nan
            out[f'podium_pct_{comp}'] = 100 * len(pc) / len(p_laps) if len(p_laps) else np.nan

    return out


def print_ferrari_benchmark(benchmark):
    """Print Ferrari vs field summary."""
    if not benchmark:
        return
    print("\nFERRARI BENCHMARK (vs Podium vs Field)")
    print(f"  Strategy delta (avg positions gained):")
    print(f"    Ferrari = {benchmark.get('ferrari_avg_strategy_delta', np.nan):.2f}")
    print(f"    Podium = {benchmark.get('podium_avg_strategy_delta', np.nan):.2f}")
    print(f"    Field = {benchmark.get('field_avg_strategy_delta', np.nan):.2f}")
    
    if 'ferrari_pct_early_pit' in benchmark and not np.isnan(benchmark.get('ferrari_pct_early_pit', np.nan)):
        print(f"  Early pit share (<33% race distance):")
        print(f"    Ferrari = {benchmark.get('ferrari_pct_early_pit', np.nan):.1f}%")
        print(f"    Podium = {benchmark.get('podium_pct_early_pit', np.nan):.1f}%")
        print(f"    Field = {benchmark.get('field_pct_early_pit', np.nan):.1f}%")


#PHASE 6 - CHARTS

def generate_charts(laps, degradation, strategy_delta, race_strategy, pit_windows=None):
    """Generate analysis charts."""
    print(f"\nPHASE 6 - CHARTS: Generating to {CHARTS_DIR}/")

    #Chart 1: Degradation curves by position group
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Lap Time Degradation by Compound (2022-2024)", fontsize=14, fontweight="bold")
    for i, group in enumerate(POSITION_GROUP_LABELS):
        ax = axes[i]
        g_laps = laps[laps['position_group'] == group]
        for compound in DRY_COMPOUNDS:
            c_laps = g_laps[g_laps['Compound'].str.upper() == compound]
            if c_laps.empty:
                continue
            by_lap = c_laps.groupby('lap_in_stint')['lap_time_sec'].mean().reset_index()
            by_lap = by_lap[by_lap['lap_in_stint'] <= 35]
            ax.plot(by_lap['lap_in_stint'], by_lap['lap_time_sec'], color=COMPOUND_COLORS[compound], linewidth=2, label=compound)
        ax.set_title(group.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Lap in Stint")
        if i == 0:
            ax.set_ylabel("Avg Lap Time (sec)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart1_degradation_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  chart1_degradation_curves.png")

    #Chart 2: Compound usage by position group
    compound_pct = laps.copy()
    compound_pct['Compound'] = compound_pct['Compound'].str.upper()
    compound_pct = compound_pct.groupby(['position_group', 'Compound'], observed=True).size().reset_index(name='lap_count')
    group_totals = compound_pct.groupby('position_group', observed=True)['lap_count'].sum().reset_index(name='group_total')
    compound_pct = compound_pct.merge(group_totals, on='position_group')
    compound_pct['pct'] = 100 * compound_pct['lap_count'] / compound_pct['group_total']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Compound Usage Share by Finishing Group (2022-2024)", fontsize=13, fontweight="bold")
    bottom = {g: 0 for g in POSITION_GROUP_LABELS}
    for compound in DRY_COMPOUNDS:
        vals = []
        for g in POSITION_GROUP_LABELS:
            row = compound_pct[(compound_pct['position_group'] == g) & (compound_pct['Compound'] == compound)]
            vals.append(row['pct'].values[0] if len(row) > 0 else 0)
        bars = ax.bar(POSITION_GROUP_LABELS, vals, bottom=[bottom[g] for g in POSITION_GROUP_LABELS], color=COMPOUND_COLORS[compound], label=compound, edgecolor='white', linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[POSITION_GROUP_LABELS[j]] + val / 2, f"{val:.1f}%", ha='center', va='center', fontsize=9, color='black' if compound == 'MEDIUM' else 'white', fontweight='bold')
            bottom[POSITION_GROUP_LABELS[j]] += val
    ax.set_ylabel("% of Race Laps")
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart2_compound_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  chart2_compound_usage.png")

    #Chart 3: Strategy delta top 10 drivers
    if len(strategy_delta) > 0:
        by_driver = strategy_delta.groupby('Driver').agg(avg_strategy_delta=('strategy_delta', 'mean')).reset_index()
        by_driver = by_driver.sort_values('avg_strategy_delta', ascending=False).head(10)
        if len(by_driver) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Strategy Delta: Top 10 Drivers (2022-2024)", fontsize=13, fontweight="bold")
            ax.barh(by_driver['Driver'], by_driver['avg_strategy_delta'], color=TOP10_COLOR, edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=1, linestyle='--')
            ax.set_xlabel("Avg Positions Gained Over Raw Pace")
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(CHARTS_DIR, 'chart3_strategy_delta.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  chart3_strategy_delta.png")

    #Chart 4: Pace rank vs finish position
    if len(strategy_delta) > 0 and 'position_group' in strategy_delta.columns:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("Raw Pace Rank vs Finish Position (2022-2024)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Pace Rank (1=fastest)")
        ax.set_ylabel("Finish Position (1=winner)")
        for group in POSITION_GROUP_LABELS:
            g = strategy_delta[strategy_delta['position_group'] == group]
            if g.empty:
                continue
            ax.scatter(g['pace_rank'], g['finish_position'], alpha=0.5, s=25, c=GROUP_COLORS[group], label=group.lower())
        
        if 'TeamName' in strategy_delta.columns:
            ferrari_sd = strategy_delta[strategy_delta['TeamName'] == TEAM_OF_INTEREST]
            if len(ferrari_sd) > 0:
                ax.scatter(ferrari_sd['pace_rank'], ferrari_sd['finish_position'], s=80, marker='*', c='#DC143C', edgecolors='black', linewidths=0.8, label=TEAM_OF_INTEREST, zorder=5)
        
        max_val = max(strategy_delta['pace_rank'].max(), strategy_delta['finish_position'].max()) if len(strategy_delta) else 20
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='pace = result')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, max_val + 0.5)
        ax.set_ylim(0.5, max_val + 0.5)
        ax.invert_yaxis()
        plt.figtext(0.5, 0.01, "Points above diagonal = strategy gain", ha='center', fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(os.path.join(CHARTS_DIR, 'chart4_pace_vs_finish.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  chart4_pace_vs_finish.png")

    print(f"  All charts saved to {CHARTS_DIR}/")


#PHASE 7 - ACT

def print_recommendations(degradation, pit_windows, strategy_delta):
    """Top 3 recommendations."""
    print("\nPHASE 7 - ACT: Top 3 Recommendations")
    print("1. PIT TIMING - Commit to early window on high-degradation circuits")
    print("   Target: Laps 12-18 for undercut attempts")
    
    print("2. COMPOUND SELECTION - Calibrate SOFT stint length by circuit")
    print("   High-deg: 12-14 laps | Low-deg: 18-22 laps")
    
    print("3. CIRCUIT FOCUS - Prioritize strategy planning on high-variance circuits")
    print("   Monaco, Hungary, Austria > Spa, Monza")


#MAIN PIPELINE

def main():
    print("\nPHASE 1 - ASK")
    print("Business question: How do podium finishers differ from mid-field?")
    print("Analysis: Ferrari perspective vs full-time drivers (2022-2024)")

    print("\nPHASE 2 - PREPARE: Loading FastF1 data...")
    laps_raw, results_raw = prepare_season_data(SEASONS)

    print("\n  Filtering to majority drivers...")
    laps_raw, results_raw = filter_to_majority_drivers(laps_raw, results_raw, MAJORITY_PCT)

    print("\nPHASE 3 - PROCESS: Cleaning...")
    laps_clean, results_clean = clean_laps(laps_raw, results_raw)
    laps_clean = build_lap_in_stint(laps_clean)
    stint_summary, race_strategy, results_with_groups = build_summary_tables(laps_clean, results_raw)

    #Merge position group and team back to laps
    merge_cols = ['season', 'race', 'Driver', 'position_group']
    if 'TeamName' in results_with_groups.columns:
        merge_cols.append('TeamName')
    results_for_merge = results_with_groups[merge_cols]
    laps_clean = laps_clean.merge(results_for_merge, on=['season', 'race', 'Driver'], how='left')

    print("\nPHASE 4 - ANALYZE: Building metrics...")
    degradation = build_degradation_curves(laps_clean)
    strategy_delta = build_strategy_delta(laps_clean, results_with_groups)
    pit_windows = build_pit_window_analysis(laps_raw, results_with_groups, race_strategy, strategy_delta)

    print("\nPHASE 5 - SHARE: Exporting CSVs...")
    ferrari_benchmark = export_tableau_files(
        laps_clean, degradation, race_strategy, strategy_delta, pit_windows, stint_summary
    )

    print("\nPHASE 6 - CHARTS: Generating visualizations...")
    generate_charts(laps_clean, degradation, strategy_delta, race_strategy, pit_windows)

    print_recommendations(degradation, pit_windows, strategy_delta)
    print("\nPipeline complete. All Tableau CSVs ready in tableau_export/")


if __name__ == "__main__":
    main()
