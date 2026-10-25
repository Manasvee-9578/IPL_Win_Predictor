"""
ChaseIQ — Data Loader (stage 1 of the pipeline)
=====================================================================
Turns the two raw CSVs into two clean, analysis-ready tables:

* ``matches_clean.csv``  – one row per match (teams, venue, toss, winner,
                           1st-innings score, whether the chase is usable)
* ``chase_states.csv``   – one row per *legal ball* of every usable 2nd
                           innings, i.e. every "match situation" the model
                           learns from.

Cleaning decisions (and why)
----------------------------
* Super-over deliveries are removed – they are a different game.
* Matches with no result / tied in regulation are removed – there is no
  clean "chasing team won / lost" label.
* Rain-shortened or otherwise interrupted innings are removed – a chase
  that stops before the 20th over without a winner or all-out would teach
  the model wrong lessons about what 120 balls means.
* Terminal states (target reached, all out, 120 balls) are removed – the
  outcome is already known, the API answers those deterministically.
* Venues are mapped to canonical names (59 raw spellings -> ~35 grounds).

Usage:
    python -m src.data_loader
"""

import sys

import numpy as np
import pandas as pd

from src import config as C
from src.reference import canonical_venue

TOTAL_BALLS = C.TOTAL_BALLS


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (C.MATCHES_CSV, C.DELIVERIES_CSV):
        if not path.is_file():
            sys.exit(f"  x Missing raw file: {path}")
    return pd.read_csv(C.MATCHES_CSV), pd.read_csv(C.DELIVERIES_CSV)


def _innings_summary(deliv: pd.DataFrame, innings: int, prefix: str) -> pd.DataFrame:
    """Final runs / wickets / legal balls / batting team of one innings per match."""
    d = deliv[deliv["innings"] == innings].copy()
    d["legal"] = (~d["is_wide_ball"] & ~d["is_no_ball"]).astype(int)
    out = d.groupby("match_id").agg(
        **{
            f"{prefix}_team": ("team_batting", "first"),
            f"{prefix}_runs": ("total_runs", "sum"),
            f"{prefix}_wickets": ("is_wicket", "sum"),
            f"{prefix}_balls": ("legal", "sum"),
        }
    )
    return out.reset_index()


def build_matches(matches: pd.DataFrame, deliv: pd.DataFrame) -> pd.DataFrame:
    """One clean row per match with a ``valid_chase`` quality flag."""
    m = matches.copy()
    m["venue"] = m["venue"].map(canonical_venue)
    m["season"] = m["season_id"].astype(int)
    m = m.rename(columns={"match_winner": "winner"})

    first = _innings_summary(deliv, 1, "first")
    second = _innings_summary(deliv, 2, "chase")
    m = m.merge(first, on="match_id", how="left").merge(second, on="match_id", how="left")

    target = m["first_runs"] + 1
    first_complete = (m["first_wickets"] >= 10) | (m["first_balls"] >= TOTAL_BALLS - 1)
    chase_decided = (
        (m["chase_runs"] >= target)
        | (m["chase_wickets"] >= 10)
        | (m["chase_balls"] >= TOTAL_BALLS - 1)
    )
    m["valid_chase"] = (
        (m["result"] == "win")
        & first_complete
        & chase_decided
        & m["chase_team"].notna()
    )
    m["chasing_team_won"] = (m["winner"] == m["chase_team"]).astype(int)

    keep = [
        "match_id", "season", "match_date", "city", "venue", "team1", "team2",
        "toss_winner", "toss_decision", "winner", "result",
        "first_team", "first_runs", "first_wickets", "first_balls",
        "chase_team", "chase_runs", "chase_wickets", "chase_balls",
        "win_by_runs", "win_by_wickets", "valid_chase", "chasing_team_won",
    ]
    m = m[keep].sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return m


def build_chase_states(deliv: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One non-terminal match situation per legal ball of every valid chase."""
    valid = matches[matches["valid_chase"]].set_index("match_id")

    d = deliv[(deliv["innings"] == 2) & deliv["match_id"].isin(valid.index)].copy()
    d = d.sort_values(["match_id", "over_number", "ball_number"], kind="stable")
    d["legal"] = (~d["is_wide_ball"] & ~d["is_no_ball"]).astype(int)
    d["cum_runs"] = d.groupby("match_id")["total_runs"].cumsum()
    d["cum_wickets"] = d.groupby("match_id")["is_wicket"].cumsum().astype(int)
    d["balls_bowled"] = d.groupby("match_id")["legal"].cumsum()

    # state after the last delivery that shares a legal-ball count
    states = (
        d.groupby(["match_id", "balls_bowled"], as_index=False)
        .last()[["match_id", "balls_bowled", "cum_runs", "cum_wickets"]]
    )

    # the pre-match situation: 0 balls, 0 runs, 0 wickets
    start = pd.DataFrame({"match_id": valid.index, "balls_bowled": 0,
                          "cum_runs": 0, "cum_wickets": 0})
    states = pd.concat([start, states], ignore_index=True)

    info = valid[["season", "venue", "chase_team", "first_team", "first_runs",
                  "toss_winner", "winner", "match_date"]]
    states = states.merge(info, left_on="match_id", right_index=True)
    states["target_score"] = states["first_runs"] + 1
    states["batting_team"] = states["chase_team"].astype(int)
    states["bowling_team"] = states["first_team"].astype(int)
    states[C.TARGET] = (states["winner"] == states["chase_team"]).astype(int)

    # drop terminal states – their outcome is not a prediction problem
    live = (
        (states["cum_runs"] < states["target_score"])
        & (states["cum_wickets"] < C.TOTAL_WICKETS)
        & (states["balls_bowled"] < TOTAL_BALLS)
    )
    states = states[live].copy()
    states["toss_advantage"] = (states["toss_winner"] == states["batting_team"]).astype(int)

    cols = ["match_id", "season", "match_date", "venue", "batting_team", "bowling_team",
            "target_score", "cum_runs", "cum_wickets", "balls_bowled",
            "toss_advantage", C.TARGET]
    return states[cols].sort_values(["match_id", "balls_bowled"]).reset_index(drop=True)


def run() -> None:
    print("=" * 60 + "\nSTAGE 1 - DATA LOADING & CLEANING\n" + "=" * 60)
    matches_raw, deliv = load_raw()
    print(f"  raw matches    : {matches_raw.shape}")
    print(f"  raw deliveries : {deliv.shape}")

    deliv = deliv[~deliv["is_super_over"]].copy()
    for col in ("is_wide_ball", "is_no_ball", "is_wicket"):
        deliv[col] = deliv[col].astype(bool)
    print(f"  deliveries after removing super overs : {len(deliv):,}")

    matches = build_matches(matches_raw, deliv)
    n_valid = int(matches["valid_chase"].sum())
    print(f"  matches        : {len(matches)}  (usable chases: {n_valid})")
    print(f"  venues         : {matches_raw['venue'].nunique()} raw -> "
          f"{matches['venue'].nunique()} canonical")
    print("  excluded       :",
          matches.loc[~matches['valid_chase'], 'result'].value_counts().to_dict())

    states = build_chase_states(deliv, matches)
    print(f"  chase states   : {len(states):,} rows "
          f"({states['match_id'].nunique()} matches, "
          f"{states[C.TARGET].mean():.1%} chasing-team wins)")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(C.MATCHES_CLEAN_CSV, index=False)
    states.to_csv(C.CHASE_STATES_CSV, index=False)
    print(f"  saved -> {C.MATCHES_CLEAN_CSV.name}, {C.CHASE_STATES_CSV.name}")


if __name__ == "__main__":
    run()
