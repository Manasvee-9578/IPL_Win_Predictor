"""
IPL Win Probability Predictor — Feature Engineering
====================================================
Loads the cleaned ball-by-ball dataset, filters to 2nd-innings
chases, and builds features that power the win-probability model.

Usage:
    python src/feature_engineering.py
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
INPUT_CSV     = os.path.join(PROCESSED_DIR, "ipl_cleaned.csv")
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "ipl_features.csv")
VENUE_ENC_PKL = os.path.join(MODELS_DIR, "venue_encoder.pkl")


# ===========================  HELPERS  =====================================

def _first_innings_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every match, compute the 1st-innings total score.
    Returns a DataFrame with columns [match_id, target_score].
    """
    first = df[df["innings"] == 1]
    totals = (
        first.groupby("match_id")["total_runs"]
        .sum()
        .reset_index()
        .rename(columns={"total_runs": "target_score"})
    )
    return totals


def _team_form(matches_df: pd.DataFrame, n: int = 5) -> dict:
    """
    Pre-compute a dictionary keyed by (team, match_id) → win-ratio
    in the last *n* completed matches for that team prior to this match.

    Parameters
    ----------
    matches_df : DataFrame
        Must contain at least [match_id, team_batting, match_winner, season].
        Should already be sorted by match_id (chronological).
    n : int
        Look-back window.

    Returns
    -------
    dict  {(team, match_id): float}
    """
    # Deduplicate to one row per match for form calculation
    # (use first occurrence per match_id — all rows in a match share the same winner)
    match_level = (
        matches_df.drop_duplicates(subset="match_id")[
            ["match_id", "team_batting", "match_winner", "season"]
        ]
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    # Build set of all teams that appear in team_batting
    teams = match_level["team_batting"].unique()

    form: dict = {}

    for team in teams:
        # All matches this team participated in (as batting or bowling)
        team_matches = match_level[
            match_level["match_id"].isin(
                matches_df.loc[
                    (matches_df["team_batting"] == team)
                    | (matches_df["team_bowling"] == team),
                    "match_id",
                ].unique()
            )
        ].sort_values("match_id")

        wins_list: list[int] = []
        for _, row in team_matches.iterrows():
            mid = row["match_id"]
            # Form based on *previous* matches only
            if len(wins_list) == 0:
                form[(team, mid)] = 0.5          # first match → neutral
            else:
                window = wins_list[-n:]
                form[(team, mid)] = sum(window) / len(window)

            # Record whether this team won
            wins_list.append(1 if row["match_winner"] == team else 0)

    return form


def _head_to_head(matches_df: pd.DataFrame) -> dict:
    """
    Pre-compute historical head-to-head win ratio for every (teamA, teamB)
    pair using only matches played *before* the current one.

    Returns lookup entries for **both** team orderings per match so that
    2nd-innings rows (where batting/bowling roles are swapped relative to
    innings 1) can find the correct value.

    Returns
    -------
    dict  {(team_a, team_b, match_id): float}
    """
    # Get the two teams per match (team1, team2 from the original matches
    # columns survive the merge; fall back to batting/bowling from the
    # first row if not available).
    needed = ["match_id", "match_winner"]
    if "team1" in matches_df.columns and "team2" in matches_df.columns:
        needed += ["team1", "team2"]
        use_team12 = True
    else:
        needed += ["team_batting", "team_bowling"]
        use_team12 = False

    match_level = (
        matches_df.drop_duplicates(subset="match_id")[needed]
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    h2h: dict = {}
    pair_history: dict = {}          # canonical pair → [winner_name, ...]

    for _, row in match_level.iterrows():
        mid    = row["match_id"]
        winner = row["match_winner"]

        if use_team12:
            t_a, t_b = row["team1"], row["team2"]
        else:
            t_a, t_b = row["team_batting"], row["team_bowling"]

        pair = tuple(sorted([t_a, t_b]))
        history = pair_history.get(pair, [])

        if len(history) == 0:
            ratio_a = 0.5
            ratio_b = 0.5
        else:
            a_wins = sum(1 for w in history if w == t_a)
            b_wins = sum(1 for w in history if w == t_b)
            ratio_a = a_wins / len(history)
            ratio_b = b_wins / len(history)

        # Store for both orderings so look-ups from either innings work
        h2h[(t_a, t_b, mid)] = ratio_a
        h2h[(t_b, t_a, mid)] = ratio_b

        pair_history.setdefault(pair, []).append(winner)

    return h2h


# ===========================  MAIN PIPELINE  ==============================

def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load cleaned data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Loading cleaned data")
    print("=" * 60)

    if not os.path.isfile(INPUT_CSV):
        sys.exit(f"  ✖ File not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  ✔ Loaded {INPUT_CSV}  →  {df.shape}")

    # ------------------------------------------------------------------
    # 2. Get 1st-innings totals & filter to 2nd innings
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — First-innings totals & 2nd-innings filter")
    print("=" * 60)

    totals = _first_innings_totals(df)
    print(f"  ✔ Computed 1st-innings totals for {len(totals)} matches")

    chase = df[df["innings"] == 2].copy()
    print(f"  ✔ Filtered to 2nd innings  →  {chase.shape}")

    chase = chase.merge(totals, on="match_id", how="left")
    print(f"  ✔ Merged target_score  →  {chase.shape}")

    # ------------------------------------------------------------------
    # 3. Per-ball features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Engineering per-ball features")
    print("=" * 60)

    # 3a  runs_remaining
    chase["runs_remaining"] = chase["target_score"] - chase["cum_runs"]
    print("  ✔ runs_remaining")

    # 3b  balls_remaining
    chase["balls_remaining"] = 120 - chase["balls_bowled"]
    print("  ✔ balls_remaining")

    # 3c  wickets_remaining
    chase["wickets_remaining"] = 10 - chase["cum_wickets"]
    print("  ✔ wickets_remaining")

    # 3d  required_run_rate  (runs_remaining / overs_remaining)
    overs_remaining = chase["balls_remaining"] / 6.0
    chase["required_run_rate"] = np.where(
        overs_remaining > 0,
        chase["runs_remaining"] / overs_remaining,
        0.0,
    )
    print("  ✔ required_run_rate")

    # 3e  current_run_rate  (cum_runs / overs_bowled)
    overs_bowled = chase["balls_bowled"] / 6.0
    chase["current_run_rate"] = np.where(
        overs_bowled > 0,
        chase["cum_runs"] / overs_bowled,
        0.0,
    )
    print("  ✔ current_run_rate")

    # 3f  run_rate_diff
    chase["run_rate_diff"] = (
        chase["current_run_rate"] - chase["required_run_rate"]
    )
    print("  ✔ run_rate_diff")

    # 3g  toss_advantage
    chase["toss_advantage"] = (
        chase["team_batting"].eq(chase["toss_winner"]).astype(int)
    )
    print("  ✔ toss_advantage")

    # ------------------------------------------------------------------
    # 4. Venue encoding (LabelEncoder → pkl)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Venue encoding")
    print("=" * 60)

    le_venue = LabelEncoder()
    chase["venue_encoded"] = le_venue.fit_transform(
        chase["venue"].astype(str)
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le_venue, VENUE_ENC_PKL)
    print(f"  ✔ venue_encoded ({le_venue.classes_.shape[0]} unique venues)")
    print(f"  ✔ Encoder saved → {VENUE_ENC_PKL}")

    # ------------------------------------------------------------------
    # 5. Team form (last 5 matches)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Batting team form (last 5 matches)")
    print("=" * 60)

    form_dict = _team_form(df, n=5)
    chase["batting_team_form"] = chase.apply(
        lambda r: form_dict.get((r["team_batting"], r["match_id"]), 0.5),
        axis=1,
    )
    print(f"  ✔ batting_team_form  (look-up size: {len(form_dict)})")

    # ------------------------------------------------------------------
    # 6. Head-to-head ratio
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Head-to-head win ratio")
    print("=" * 60)

    h2h_dict = _head_to_head(df)
    chase["head_to_head_ratio"] = chase.apply(
        lambda r: h2h_dict.get(
            (r["team_batting"], r["team_bowling"], r["match_id"]), 0.5
        ),
        axis=1,
    )
    print(f"  ✔ head_to_head_ratio  (look-up size: {len(h2h_dict)})")

    # ------------------------------------------------------------------
    # 7. Drop NaN rows in any feature column
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — Dropping NaN rows")
    print("=" * 60)

    feature_cols = [
        "runs_remaining",
        "balls_remaining",
        "wickets_remaining",
        "required_run_rate",
        "current_run_rate",
        "run_rate_diff",
        "toss_advantage",
        "venue_encoded",
        "batting_team_form",
        "head_to_head_ratio",
        "batting_team_wins",
    ]

    before = len(chase)
    chase.dropna(subset=feature_cols, inplace=True)
    dropped = before - len(chase)
    print(f"  ✔ Dropped {dropped} rows with NaN  →  {chase.shape}")

    # ------------------------------------------------------------------
    # 8. Save feature-engineered CSV
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 — Saving feature-engineered data")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    chase.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✔ Saved to {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # 9. Summary: shape, correlations, sample row
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9 — Summary")
    print("=" * 60)

    print(f"\n  Shape: {chase.shape}")

    print("\n  Feature correlations with batting_team_wins:")
    corr = chase[feature_cols].corr()["batting_team_wins"].drop("batting_team_wins")
    for feat, val in corr.sort_values(ascending=False).items():
        print(f"    {feat:>25s}  {val:+.4f}")

    print("\n  Sample row (feature columns):")
    sample = chase[feature_cols].iloc[0]
    for feat, val in sample.items():
        print(f"    {feat:>25s}  {val}")

    print("\n✅ Feature engineering complete!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
