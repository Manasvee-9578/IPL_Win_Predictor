"""
ChaseIQ - Feature Engineering (stage 2)
=============================================================
Adds *context* features that describe the two teams and the ground, then
expands every chase state into the model's feature vector.

Context features are computed strictly chronologically: the value attached
to a match only uses matches played **before** it (no future leakage).

* Elo rating       - every team starts at 1500, K=24, with 25% regression to
                     the mean each season (squads change between auctions).
* Form             - smoothed win ratio over the last 5 matches.
* Head-to-head     - smoothed win ratio of the batting team vs this opponent.
* Venue average    - shrunk mean 1st-innings score at the ground.
* Venue chase bias - shrunk share of matches won by the chasing side.

Usage:
    python -m src.feature_engineering
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src import config as C
from src.features import make_features

VENUE_PRIOR_WEIGHT = 3      # matches of pseudo-evidence for the prior mean
CHASE_PRIOR_WEIGHT = 4


class ContextState:
    """Ratings that evolve match by match. After the last match it is the
    *current* state used for live predictions."""

    def __init__(self):
        self.elo = defaultdict(lambda: C.ELO_START)
        self.history = defaultdict(list)                      # 1/0 results per team
        self.h2h = defaultdict(lambda: [0, 0])                # (a, b) -> [a wins, games]
        self.venue_sum = defaultdict(float)
        self.venue_n = defaultdict(int)
        self.venue_chase_wins = defaultdict(int)
        self.season = None

    # ---- read ---------------------------------------------------------
    def form(self, team: int) -> float:
        last = self.history[team][-C.FORM_WINDOW:]
        return (sum(last) + 1.0) / (len(last) + 2.0)          # Laplace smoothing

    def h2h_ratio(self, a: int, b: int) -> float:
        wins, games = self.h2h[(a, b)]
        return (wins + 1.0) / (games + 2.0)

    def venue_avg(self, venue: str) -> float:
        n = self.venue_n[venue]
        return (self.venue_sum[venue] + VENUE_PRIOR_WEIGHT * C.PRIOR_FIRST_INNINGS) / (
            n + VENUE_PRIOR_WEIGHT)

    def venue_bias(self, venue: str) -> float:
        n = self.venue_n[venue]
        return (self.venue_chase_wins[venue] + CHASE_PRIOR_WEIGHT / 2) / (
            n + CHASE_PRIOR_WEIGHT)

    # ---- write --------------------------------------------------------
    def new_season(self, season: int) -> None:
        if self.season is not None and season != self.season:
            for t in list(self.elo):
                self.elo[t] = C.ELO_START + (self.elo[t] - C.ELO_START) * (
                    1 - C.ELO_SEASON_REGRESSION)
        self.season = season

    def update(self, row) -> None:
        t1, t2, winner = int(row.team1), int(row.team2), row.winner
        if row.result in ("win", "tie") and pd.notna(winner):
            winner = int(winner)
            loser = t2 if winner == t1 else t1
            expected = 1.0 / (1.0 + 10 ** ((self.elo[loser] - self.elo[winner]) / 400.0))
            delta = C.ELO_K * (1.0 - expected)
            self.elo[winner] += delta
            self.elo[loser] -= delta
            self.history[winner].append(1)
            self.history[loser].append(0)
            self.h2h[(winner, loser)][0] += 1
            self.h2h[(winner, loser)][1] += 1
            self.h2h[(loser, winner)][1] += 1
        if row.valid_chase:
            self.venue_sum[row.venue] += row.first_runs
            self.venue_n[row.venue] += 1
            self.venue_chase_wins[row.venue] += int(row.chasing_team_won)


def compute_context(matches: pd.DataFrame) -> tuple[pd.DataFrame, ContextState]:
    """Pre-match context for every match + the final state after the last match."""
    state = ContextState()
    records = []
    for row in matches.sort_values(["match_date", "match_id"]).itertuples(index=False):
        state.new_season(int(row.season))
        t1, t2 = int(row.team1), int(row.team2)
        records.append({
            "match_id": row.match_id,
            "elo1": state.elo[t1], "elo2": state.elo[t2],
            "form1": state.form(t1), "form2": state.form(t2),
            "h2h1": state.h2h_ratio(t1, t2), "h2h2": state.h2h_ratio(t2, t1),
            "venue_avg": state.venue_avg(row.venue),
            "venue_bias": state.venue_bias(row.venue),
        })
        state.update(row)
    return pd.DataFrame(records), state


def build_feature_table(states: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    ctx, _ = compute_context(matches)
    df = states.merge(ctx, on="match_id").merge(matches[["match_id", "team1"]], on="match_id")
    bat_is_t1 = (df["batting_team"] == df["team1"]).to_numpy()

    def pick(a: str, b: str) -> np.ndarray:
        return np.where(bat_is_t1, df[a], df[b])

    feats = make_features(
        target_score=df["target_score"],
        current_score=df["cum_runs"],
        wickets_fallen=df["cum_wickets"],
        balls_bowled=df["balls_bowled"],
        elo_diff=pick("elo1", "elo2") - pick("elo2", "elo1"),
        batting_team_form=pick("form1", "form2"),
        head_to_head_ratio=pick("h2h1", "h2h2"),
        venue_avg_score=df["venue_avg"],
        venue_chase_bias=df["venue_bias"],
        toss_advantage=df["toss_advantage"],
    )
    meta = df[["match_id", "season", "balls_bowled", C.TARGET]].reset_index(drop=True)
    return pd.concat([meta, feats.reset_index(drop=True)], axis=1)


def run() -> None:
    print("=" * 60 + "\nSTAGE 2 - FEATURE ENGINEERING\n" + "=" * 60)
    matches = pd.read_csv(C.MATCHES_CLEAN_CSV)
    states = pd.read_csv(C.CHASE_STATES_CSV)

    table = build_feature_table(states, matches)
    table.to_csv(C.FEATURES_CSV, index=False)
    print(f"  feature table : {table.shape}  -> {C.FEATURES_CSV.name}")

    corr = table[C.FEATURE_COLS + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET)
    print("\n  correlation with 'chasing team wins':")
    for name, val in corr.sort_values(ascending=False).items():
        print(f"    {name:<22s}{val:+.3f}")


if __name__ == "__main__":
    run()
