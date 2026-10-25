"""
ChaseIQ - Database Builder (stage 4)
==========================================================
Creates ``database/ipl.db`` from the cleaned tables and the trained model:

* teams / venues / matches / head_to_head  - reference + history
* teams.elo, teams.form                    - the *current* ratings used live
* chase_states.win_prob                    - the model's win probability after
                                             every ball of every historical chase
                                             (powers Match Replay & Analytics)

The two operational tables (prediction_log, live_snapshots) are created
empty and are filled by the API at run time.

Usage:
    python -m src.build_database
"""

import joblib
import pandas as pd

from src import config as C
from src import database as db
from src.feature_engineering import compute_context
from src.reference import TEAMS

BATCH = 50_000


def run() -> None:
    print("=" * 60 + "\nSTAGE 4 - BUILD DATABASE\n" + "=" * 60)
    matches = pd.read_csv(C.MATCHES_CLEAN_CSV)
    features = pd.read_csv(C.FEATURES_CSV)
    model = joblib.load(C.MODEL_PKL)

    C.DB_PATH.parent.mkdir(exist_ok=True)
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()

    _, ctx = compute_context(matches)

    with db.session() as conn:
        db.init_schema(conn)

        # ---- teams ------------------------------------------------------
        played = pd.concat([matches["team1"], matches["team2"]]).value_counts()
        wins = matches.loc[matches["result"].isin(["win", "tie"]), "winner"].value_counts()
        for t in TEAMS:
            conn.execute(
                "INSERT INTO teams VALUES (?,?,?,?,?,?,?,?,?)",
                (t.team_id, t.name, t.abbr, t.color, int(t.active),
                 round(ctx.elo[t.team_id], 2), round(ctx.form(t.team_id), 4),
                 int(played.get(t.team_id, 0)), int(wins.get(t.team_id, 0))))

        # ---- venues -----------------------------------------------------
        city = matches.groupby("venue")["city"].agg(
            lambda s: s.dropna().mode().iat[0] if s.notna().any() else None)
        for name in sorted(matches["venue"].unique()):
            conn.execute(
                "INSERT INTO venues (name, city, matches, avg_first_innings, chase_bias)"
                " VALUES (?,?,?,?,?)",
                (name, city.get(name), int(ctx.venue_n[name]),
                 round(ctx.venue_avg(name), 2), round(ctx.venue_bias(name), 4)))
        venue_id = {r["name"]: r["venue_id"]
                    for r in db.rows(conn, "SELECT venue_id, name FROM venues")}

        # ---- matches ----------------------------------------------------
        def nn(v):
            return None if pd.isna(v) else v

        for r in matches.itertuples(index=False):
            conn.execute(
                "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (int(r.match_id), int(r.season), r.match_date, venue_id[r.venue],
                 int(r.team1), int(r.team2),
                 nn(r.toss_winner) and int(r.toss_winner), nn(r.toss_decision),
                 nn(r.winner) and int(r.winner), r.result,
                 nn(r.first_team) and int(r.first_team), nn(r.first_runs) and int(r.first_runs),
                 nn(r.chase_team) and int(r.chase_team), nn(r.chase_runs) and int(r.chase_runs),
                 nn(r.chase_wickets) and int(r.chase_wickets),
                 nn(r.chase_balls) and int(r.chase_balls),
                 nn(r.win_by_runs), nn(r.win_by_wickets), int(r.valid_chase),
                 "train" if r.season <= C.TRAIN_END else "test"))

        # ---- head to head ----------------------------------------------
        for (a, b), (w, g) in ctx.h2h.items():
            conn.execute("INSERT INTO head_to_head VALUES (?,?,?,?)", (a, b, w, g))

        # ---- chase states (model output for every historical ball) --------
        probs = model.predict_proba(features[C.FEATURE_COLS])[:, 1]
        meta = pd.read_csv(C.CHASE_STATES_CSV)
        states = pd.DataFrame({
            "match_id": meta["match_id"], "balls": meta["balls_bowled"],
            "runs": meta["cum_runs"], "wickets": meta["cum_wickets"],
            "win_prob": probs.round(4)})

        # terminal state of each chase: outcome known -> probability 0 or 1
        valid = matches[matches["valid_chase"]]
        terminal = pd.DataFrame({
            "match_id": valid["match_id"], "balls": valid["chase_balls"].astype(int),
            "runs": valid["chase_runs"].astype(int),
            "wickets": valid["chase_wickets"].astype(int),
            "win_prob": valid["chasing_team_won"].astype(float)})
        states = (pd.concat([states, terminal], ignore_index=True)
                  .drop_duplicates(["match_id", "balls"], keep="last"))

        for i in range(0, len(states), BATCH):
            conn.executemany("INSERT INTO chase_states VALUES (?,?,?,?,?)",
                             states.iloc[i:i + BATCH].itertuples(index=False, name=None))

        # ---- pre-match context of every chase (first state of each match) -----
        first = features[features["balls_bowled"] == 0].drop_duplicates("match_id")
        ctx_rows = [(int(r.match_id), int(r.target_score), float(r.elo_diff), float(r.batting_team_form),
                     float(r.head_to_head_ratio), float(r.target_score - r.target_vs_venue),
                     float(r.venue_chase_bias), int(r.toss_advantage)) for r in first.itertuples()]
        conn.executemany("INSERT INTO match_context VALUES (?,?,?,?,?,?,?,?)", ctx_rows)

        for tbl in ("teams", "venues", "matches", "head_to_head", "chase_states", "match_context"):
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:<14s}{n:>9,} rows")

    size = C.DB_PATH.stat().st_size / 1e6
    print(f"  -> {C.DB_PATH.relative_to(C.ROOT)}  ({size:.1f} MB)")


if __name__ == "__main__":
    run()
