"""
IPL Win Probability Predictor — Data Loader
============================================
Loads, cleans, merges and feature-engineers the IPL matches & deliveries
CSVs and writes out a single analysis-ready file.

Usage:
    python src/data_loader.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
MATCHES_CSV   = os.path.join(RAW_DIR, "matches.csv")
DELIVERIES_CSV = os.path.join(RAW_DIR, "deliveries.csv")
OUTPUT_CSV    = os.path.join(PROCESSED_DIR, "ipl_cleaned.csv")


# ---------------------------------------------------------------------------
# Team-name standardisation mapping
# ---------------------------------------------------------------------------
TEAM_NAME_MAP = {
    "Delhi Daredevils":              "Delhi Capitals",
    "Kings XI Punjab":               "Punjab Kings",
    "Deccan Chargers":               "Sunrisers Hyderabad",
    "Rising Pune Supergiant":        "Rising Pune Supergiants",
    "Royal Challengers Bangalore":   "Royal Challengers Bengaluru",
}


# ---------------------------------------------------------------------------
# Venue → City dictionary (fixes missing city values)
# ---------------------------------------------------------------------------
VENUE_CITY_MAP = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium":             "Sharjah",
    "Sheikh Zayed Stadium":                "Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi":    "Abu Dhabi",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_match_id_col(df: pd.DataFrame, label: str) -> str:
    """Return the first column name that looks like a match-ID field."""
    candidates = ["match_id", "id", "ID", "Match_Id", "matchId"]
    for c in candidates:
        if c in df.columns:
            print(f"  ✔ Detected match-ID column in {label}: '{c}'")
            return c
    raise KeyError(
        f"Could not auto-detect a match-ID column in {label}. "
        f"Available columns: {list(df.columns)}"
    )


def _standardise_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Replace legacy franchise names in every team-related column."""
    team_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in
               ["team", "winner", "toss_winner", "batting", "bowling"])
    ]
    for col in team_cols:
        if df[col].dtype == object:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    return df


# ===========================  MAIN PIPELINE  ==============================
def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load CSVs
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Loading CSVs")
    print("=" * 60)

    try:
        matches = pd.read_csv(MATCHES_CSV)
        print(f"  ✔ Loaded matches.csv  →  {matches.shape}")
    except FileNotFoundError:
        sys.exit(f"  ✖ File not found: {MATCHES_CSV}")

    try:
        deliveries = pd.read_csv(DELIVERIES_CSV)
        print(f"  ✔ Loaded deliveries.csv  →  {deliveries.shape}")
    except FileNotFoundError:
        sys.exit(f"  ✖ File not found: {DELIVERIES_CSV}")

    # ------------------------------------------------------------------
    # 2. Print all column names
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Column names")
    print("=" * 60)
    print(f"\n  matches.csv columns ({len(matches.columns)}):")
    for col in matches.columns:
        print(f"    • {col}")

    print(f"\n  deliveries.csv columns ({len(deliveries.columns)}):")
    for col in deliveries.columns:
        print(f"    • {col}")

    # ------------------------------------------------------------------
    # 3. Auto-detect match-ID column
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Auto-detecting match-ID column")
    print("=" * 60)
    try:
        mid_matches = _detect_match_id_col(matches, "matches")
        mid_deliveries = _detect_match_id_col(deliveries, "deliveries")
    except KeyError as e:
        sys.exit(str(e))

    # ------------------------------------------------------------------
    # 4. Remove no-result & D/L affected matches
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Removing no-result & D/L matches")
    print("=" * 60)
    before = len(matches)

    # 4a  No-result
    if "result" in matches.columns:
        matches = matches[matches["result"] != "no result"]
        print(f"  ✔ Dropped no-result matches  ({before - len(matches)} removed)")
    else:
        print("  ⚠ 'result' column not found — skipping no-result filter")

    # 4b  Duckworth-Lewis
    dl_cols = ["method", "dl_applied", "DL Applied", "dl"]
    dl_col_found = None
    for dc in dl_cols:
        if dc in matches.columns:
            dl_col_found = dc
            break

    if dl_col_found is not None:
        before_dl = len(matches)
        # Covers both string ("D/L") and int (1) indicators
        if matches[dl_col_found].dtype == object:
            matches = matches[
                matches[dl_col_found].isna()
                | (matches[dl_col_found].str.upper() == "NA")
                | (matches[dl_col_found] == "")
            ]
        else:
            matches = matches[matches[dl_col_found] != 1]
        print(f"  ✔ Dropped D/L matches using '{dl_col_found}'  "
              f"({before_dl - len(matches)} removed)")
    else:
        print("  ⚠ No D/L column found — skipping D/L filter")

    print(f"  → Matches remaining: {len(matches)}")

    # ------------------------------------------------------------------
    # 5. Standardise team names
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Standardising team names")
    print("=" * 60)
    matches = _standardise_teams(matches)
    deliveries = _standardise_teams(deliveries)
    print("  ✔ Team names standardised in both DataFrames")

    # ------------------------------------------------------------------
    # 6. Fix missing city values using venue→city map
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Fixing missing city values")
    print("=" * 60)

    if "city" in matches.columns and "venue" in matches.columns:
        missing_before = matches["city"].isna().sum()
        matches["city"] = matches.apply(
            lambda row: VENUE_CITY_MAP.get(row["venue"], row["city"])
            if pd.isna(row["city"]) else row["city"],
            axis=1,
        )
        fixed = missing_before - matches["city"].isna().sum()
        print(f"  ✔ Fixed {fixed} missing city values  "
              f"(remaining NaN: {matches['city'].isna().sum()})")
    else:
        print("  ⚠ 'city' or 'venue' column not found — skipping city fix")

    # ------------------------------------------------------------------
    # 7. Merge deliveries with matches
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — Merging deliveries with matches")
    print("=" * 60)

    try:
        merged = deliveries.merge(
            matches,
            left_on=mid_deliveries,
            right_on=mid_matches,
            how="inner",
            suffixes=("", "_match"),
        )
        print(f"  ✔ Merged shape: {merged.shape}")
    except Exception as e:
        sys.exit(f"  ✖ Merge failed: {e}")

    # ------------------------------------------------------------------
    # 8. Cumulative columns per ball per innings
    #    cum_runs, cum_wickets, balls_bowled
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 — Creating cumulative columns")
    print("=" * 60)

    try:
        # Determine the innings grouping column
        innings_col = None
        for candidate in ["innings", "inning", "innings_number"]:
            if candidate in merged.columns:
                innings_col = candidate
                break

        if innings_col is None:
            print("  ⚠ No innings column found — grouping by match_id only")
            group_keys = [mid_deliveries]
        else:
            group_keys = [mid_deliveries, innings_col]

        # Sort to guarantee correct cumulative order
        sort_cols = group_keys.copy()
        if "over_number" in merged.columns:
            sort_cols.append("over_number")
        if "ball_number" in merged.columns:
            sort_cols.append("ball_number")
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

        # -- cum_runs
        runs_col = "total_runs" if "total_runs" in merged.columns else None
        if runs_col:
            merged["cum_runs"] = merged.groupby(group_keys)[runs_col].cumsum()
            print(f"  ✔ cum_runs  (based on '{runs_col}')")
        else:
            print("  ⚠ 'total_runs' not found — cum_runs skipped")

        # -- cum_wickets
        wicket_col = "is_wicket" if "is_wicket" in merged.columns else None
        if wicket_col:
            merged["cum_wickets"] = (
                merged.groupby(group_keys)[wicket_col].cumsum().astype(int)
            )
            print(f"  ✔ cum_wickets  (based on '{wicket_col}')")
        else:
            print("  ⚠ 'is_wicket' not found — cum_wickets skipped")

        # -- balls_bowled (count of legal deliveries)
        #    Exclude wides and no-balls when the columns exist
        if "is_wide_ball" in merged.columns:
            is_legal = (~merged["is_wide_ball"].astype(bool))
        else:
            is_legal = pd.Series(True, index=merged.index)

        if "is_no_ball" in merged.columns:
            is_legal = is_legal & (~merged["is_no_ball"].astype(bool))

        merged["is_legal"] = is_legal.astype(int)
        merged["balls_bowled"] = (
            merged.groupby(group_keys)["is_legal"].cumsum().astype(int)
        )
        merged.drop(columns=["is_legal"], inplace=True)
        print("  ✔ balls_bowled  (legal deliveries only)")

    except Exception as e:
        print(f"  ⚠ Error creating cumulative columns: {e}")

    # ------------------------------------------------------------------
    # 9. Target label: batting_team_wins
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9 — Creating target label (batting_team_wins)")
    print("=" * 60)

    batting_col = None
    for candidate in ["team_batting", "batting_team", "BattingTeam"]:
        if candidate in merged.columns:
            batting_col = candidate
            break

    winner_col = None
    # Check both original and merge-suffixed names
    for candidate in ["match_winner", "winner", "Winner",
                       "match_winner_match", "winner_match", "Winner_match"]:
        if candidate in merged.columns:
            winner_col = candidate
            break

    if batting_col and winner_col:
        merged["batting_team_wins"] = (
            merged[batting_col].eq(merged[winner_col]).astype(int)
        )
        print(f"  ✔ batting_team_wins created  "
              f"('{batting_col}' == '{winner_col}')")
    else:
        missing = []
        if not batting_col:
            missing.append("batting_team")
        if not winner_col:
            missing.append("winner")
        print(f"  ⚠ Could not create target — missing column(s): {missing}")

    # ------------------------------------------------------------------
    # 10. Save cleaned CSV
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10 — Saving cleaned dataframe")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    try:
        merged.to_csv(OUTPUT_CSV, index=False)
        print(f"  ✔ Saved to {OUTPUT_CSV}")
    except Exception as e:
        sys.exit(f"  ✖ Failed to save: {e}")

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 11 — Final summary")
    print("=" * 60)
    print(f"  Shape: {merged.shape}")
    if "batting_team_wins" in merged.columns:
        print(f"\n  batting_team_wins value_counts:")
        vc = merged["batting_team_wins"].value_counts()
        for val, count in vc.items():
            print(f"    {val}: {count}")
    print("\n✅ Pipeline complete!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
