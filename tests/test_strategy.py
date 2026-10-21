"""Tests for cleaning, pit detection and the pace-adjusted strategy delta.

Run: python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import F1_Strategy_Champion_Python as f1  # noqa: E402


def lap(driver, n, secs, stint=1, compound="MEDIUM", pit_in=False, pit_out=False, race="Test GP", team="T"):
    return {"season": 2024, "race": race, "Driver": driver, "LapNumber": n, "Stint": stint,
            "LapTime": pd.Timedelta(seconds=secs) if secs is not None else pd.NaT,
            "PitInTime": pd.Timedelta(seconds=1) if pit_in else pd.NaT,
            "PitOutTime": pd.Timedelta(seconds=1) if pit_out else pd.NaT,
            "Compound": compound, "Position": 1, "Team": team}


def test_cleaning_rules_and_pit_detection_from_raw_laps():
    laps = pd.DataFrame([
        lap("AAA", 1, 90), lap("AAA", 2, 90.5), lap("AAA", 3, 91, pit_in=True),
        lap("AAA", 4, 95, stint=2, compound="HARD", pit_out=True), lap("AAA", 5, 90.2, stint=2, compound="HARD"),
        lap("AAA", 6, None), lap("AAA", 7, 90, compound="INTERMEDIATE"), lap("AAA", 8, 120),  # null, wet, >110%
        lap("BBB", 1, 91), lap("BBB", 2, 91.5),                                            # BBB retired
    ])
    results = pd.DataFrame({"season": 2024, "race": "Test GP", "Driver": ["AAA", "BBB"],
                            "Status": ["Finished", "Retired"], "Position": [1, 2], "TeamName": ["T", "U"]})
    clean, _ = f1.clean_laps(laps, results)
    assert sorted(clean["LapNumber"]) == [1, 2, 5]
    # Pit stops have to come from the uncleaned laps: cleaning removes every in/out lap
    assert clean["PitInTime"].isna().all()
    results["position_group"] = ["Podium", "Podium"]
    pits = f1.build_pit_window_analysis(laps, results, None)
    assert pits[["Driver", "LapNumber"]].values.tolist() == [["AAA", 3]]
    assert pits["pit_window"].iloc[0] == "mid"  # lap 3 of 8


def _strategy_delta_with_ceiling(n_races=60, seed=0):
    """20 cars, 10 teams, every team converts pace identically, but finishes are clipped to the
    grid (the ceiling effect). Team 'Front' usually has the two fastest cars, so its RAW delta is
    negative even though it does nothing different."""
    rng = np.random.default_rng(seed)
    teams = ["Front"] + [f"T{i}" for i in range(1, 10)]
    rows = []
    for r in range(n_races):
        rest = list(rng.permutation(np.arange(2, 20)))
        order = [0, 1] + rest if rng.random() < 0.8 else list(rng.permutation(20))
        pace = np.empty(20, dtype=int)
        pace[order] = np.arange(1, 21)
        finish = np.argsort(np.argsort(pace + rng.normal(0, 2.0, 20))) + 1
        for i, (p, fpos) in enumerate(zip(pace, finish)):
            rows.append({"season": 2024, "race": f"R{r}", "Driver": f"D{i}", "TeamName": teams[i // 2],
                         "pace_rank": p, "finish_position": fpos, "strategy_delta": p - fpos})
    return pd.DataFrame(rows)


def test_pace_adjustment_removes_ceiling_effect():
    sd = _strategy_delta_with_ceiling()
    _, summary = f1.build_pace_adjusted_delta(sd, n_boot=500)
    s = summary.set_index("TeamName")
    assert s.loc["Front", "raw_strategy_delta"] < -0.4        # looks like Front "loses places"
    assert s.loc["Front", "ci_low"] < 0 < s.loc["Front", "ci_high"]   # but converts pace like everyone
    assert s["delta_vs_pace"].abs().max() < 0.5


def test_team_with_exclusive_pace_ranks_is_kept():
    sd = _strategy_delta_with_ceiling()
    front = sd["TeamName"] == "Front"
    sd.loc[~front & (sd["pace_rank"] <= 2), "pace_rank"] = 3   # nobody else is ever P1/P2 on pace
    sd.loc[front, "pace_rank"] = sd.loc[front].groupby("race").cumcount() + 1
    sd["strategy_delta"] = sd["pace_rank"] - sd["finish_position"]
    _, summary = f1.build_pace_adjusted_delta(sd, n_boot=200)
    row = summary.set_index("TeamName").loc["Front"]
    assert row["baseline_fallback_rows"] == row["driver_races"]
    assert np.isfinite(row["delta_vs_pace"])


def test_pace_adjustment_detects_a_real_edge():
    sd = _strategy_delta_with_ceiling()
    edge = (sd["TeamName"] == "T5") & (sd["finish_position"] > 1)
    sd.loc[edge, "finish_position"] -= 1  # T5 gains a place whenever it can
    sd["strategy_delta"] = sd["pace_rank"] - sd["finish_position"]
    _, summary = f1.build_pace_adjusted_delta(sd, n_boot=500)
    assert summary.set_index("TeamName").loc["T5", "ci_low"] > 0.5


def test_raw_csvs_round_trip_timedeltas(tmp_path, monkeypatch):
    laps = pd.DataFrame([lap("AAA", 1, 90.25, pit_in=True)])
    laps["Time"] = pd.Timedelta(minutes=61)
    results = pd.DataFrame({"season": [2024], "race": ["Test GP"], "Driver": ["AAA"], "Q1": [pd.NaT]})
    weather = pd.DataFrame({"season": [2024], "race": ["Test GP"], "Time": [pd.Timedelta(seconds=63)]})
    for name, df in [("raw_laps.csv", laps), ("raw_results.csv", results), ("raw_weather.csv", weather)]:
        df.to_csv(tmp_path / name, index=False)
    monkeypatch.setattr(f1, "RAW_DIR", str(tmp_path))
    l2, r2, w2 = f1.load_raw_csvs()
    assert l2["LapTime"].iloc[0].total_seconds() == pytest.approx(90.25)
    assert l2["PitInTime"].notna().iloc[0] and l2["PitOutTime"].isna().iloc[0]
    assert w2["Time"].iloc[0] == pd.Timedelta(seconds=63)


def test_committed_raw_data_loads():
    laps, results, _ = f1.load_raw_csvs()
    assert results.groupby("season")["race"].nunique().to_dict() == {2022: 22, 2023: 22, 2024: 24}
    assert pd.api.types.is_timedelta64_dtype(laps["LapTime"])
    assert laps["PitInTime"].notna().sum() > 2000
