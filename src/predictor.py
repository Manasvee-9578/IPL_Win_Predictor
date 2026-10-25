"""
Prediction engine used by the API (and the tests).

Responsibilities
----------------
* load the trained model and the current ratings from the database
* turn a human-readable match situation into the model's feature vector
* answer terminal situations (chase won / lost / tied) deterministically
* explain a prediction (which factors help or hurt the chasing team)
* run "what-if" scenarios and win-probability heat-maps
"""

import difflib
import json
import math
from dataclasses import dataclass

import joblib
import numpy as np

from src import config as C
from src import database as db
from src.features import make_features
from src.cricket.classify import team_group
from src.reference import TEAM_BY_ID, canonical_venue, team_id

# Human-friendly groups of correlated features, used for explanations.
FEATURE_GROUPS = {
    "Required scoring rate": ["runs_remaining", "balls_remaining", "required_run_rate",
                              "current_run_rate", "run_rate_diff"],
    "Wickets in hand": ["wickets_remaining"],
    "Target vs. this ground": ["target_score", "target_vs_venue", "venue_chase_bias"],
    "Team strength & form": ["elo_diff", "batting_team_form", "head_to_head_ratio"],
    "Toss": ["toss_advantage"],
}

FEATURE_LABELS = {
    "runs_remaining": "Runs still needed",
    "balls_remaining": "Balls left",
    "wickets_remaining": "Wickets in hand",
    "required_run_rate": "Required run rate",
    "current_run_rate": "Current run rate",
    "run_rate_diff": "Run-rate gap (CRR - RRR)",
    "target_score": "Size of the target",
    "target_vs_venue": "Target vs. ground average",
    "elo_diff": "Team strength (Elo gap)",
    "batting_team_form": "Recent form",
    "head_to_head_ratio": "Head-to-head record",
    "venue_chase_bias": "Chasing record at this ground",
    "toss_advantage": "Toss won by chasing team",
}


MODEL_NAME = "IPL Second-Innings Chase Model"
MODEL_SCOPE = ("Trained only on IPL (2008-2025) second-innings run chases. ChaseIQ never applies it to ODI, "
               "T20I or Test matches, or to first innings.")


class PredictionError(ValueError):
    """Raised for requests that cannot be interpreted (unknown team, etc.)."""


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Situation:
    batting_team: int
    bowling_team: int
    venue: str
    toss_winner: int | None
    target_score: int
    current_score: int
    wickets_fallen: int
    balls_bowled: int
    batting_team_form: float | None = None
    head_to_head_ratio: float | None = None


class Predictor:
    def __init__(self, db_path=None, model_path=None):
        self.model = joblib.load(model_path or C.MODEL_PKL)
        self.db_path = db_path or C.DB_PATH
        self.metrics = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
        self.reload_ratings()

    # ------------------------------------------------------------------
    # reference data (held in memory; refreshed by reload_ratings)
    # ------------------------------------------------------------------
    def reload_ratings(self) -> None:
        with db.session(self.db_path) as conn:
            self.team_rows = {r["team_id"]: r for r in db.rows(conn, "SELECT * FROM teams")}
            self.venue_rows = {r["name"]: r for r in db.rows(conn, "SELECT * FROM venues")}
            self.h2h = {(r["team_a"], r["team_b"]): (r["a_wins"], r["games"])
                        for r in db.rows(conn, "SELECT * FROM head_to_head")}
            self.match_ctx = {r["match_id"]: r for r in db.rows(conn, "SELECT * FROM match_context")}
        self._venue_lookup = {n.lower(): n for n in self.venue_rows}

    def resolve_team(self, name) -> int:
        tid = team_id(name)
        if tid is None or tid not in self.team_rows:
            if isinstance(name, str) and team_group(name):
                raise PredictionError(
                    f"{name!r} is an international side. The {MODEL_NAME} only supports IPL franchises, so "
                    "ChaseIQ does not predict ODI, T20I or Test matches.")
            raise PredictionError(f"Unknown IPL team: {name!r}")
        return tid

    def resolve_venue(self, name: str) -> tuple[str, bool]:
        """Return (canonical venue name, known?). Fuzzy-matches unfamiliar spellings."""
        canon = canonical_venue(name)
        if canon in self.venue_rows:
            return canon, True
        key = canon.lower()
        if key in self._venue_lookup:
            return self._venue_lookup[key], True
        close = difflib.get_close_matches(key, list(self._venue_lookup), n=1, cutoff=0.72)
        if close:
            return self._venue_lookup[close[0]], True
        return canon, False

    def context(self, batting: int, bowling: int, venue: str) -> dict:
        """Team-strength and ground context from the database."""
        venue_name, known = self.resolve_venue(venue)
        v = self.venue_rows.get(venue_name)
        wins, games = self.h2h.get((batting, bowling), (0, 0))
        elo_b = self.team_rows[batting]["elo"]
        elo_o = self.team_rows[bowling]["elo"]
        return {
            "venue": venue_name,
            "venue_known": known,
            "elo_batting": elo_b,
            "elo_bowling": elo_o,
            "elo_diff": elo_b - elo_o,
            "batting_team_form": self.team_rows[batting]["form"],
            "head_to_head_ratio": (wins + 1.0) / (games + 2.0),
            "head_to_head_games": games,
            "venue_avg_score": v["avg_first_innings"] if v else C.PRIOR_FIRST_INNINGS,
            "venue_chase_bias": v["chase_bias"] if v else 0.5,
        }

    # ------------------------------------------------------------------
    # core prediction
    # ------------------------------------------------------------------
    def _features(self, s: Situation, ctx: dict):
        return make_features(
            target_score=s.target_score, current_score=s.current_score,
            wickets_fallen=s.wickets_fallen, balls_bowled=s.balls_bowled,
            elo_diff=ctx["elo_diff"],
            batting_team_form=ctx["batting_team_form"] if s.batting_team_form is None
            else s.batting_team_form,
            head_to_head_ratio=ctx["head_to_head_ratio"] if s.head_to_head_ratio is None
            else s.head_to_head_ratio,
            venue_avg_score=ctx["venue_avg_score"], venue_chase_bias=ctx["venue_chase_bias"],
            toss_advantage=int(s.toss_winner == s.batting_team),
        )

    @staticmethod
    def terminal(s: Situation) -> tuple[str, float] | None:
        """(status, win probability) if the chase is already decided, else None."""
        if s.current_score >= s.target_score:
            return "chase_complete", 1.0
        if s.wickets_fallen >= C.TOTAL_WICKETS or s.balls_bowled >= C.TOTAL_BALLS:
            if s.current_score == s.target_score - 1:
                return "tied", 0.5
            return "chase_failed", 0.0
        return None

    def predict(self, s: Situation, explain: bool = True) -> dict:
        ctx = self.context(s.batting_team, s.bowling_team, s.venue)
        s.venue = ctx["venue"]
        done = self.terminal(s)
        if done:
            status, p = done
            return {"win_probability": p, "lose_probability": 1 - p, "status": status,
                    "context": ctx, "explanation": None}

        X = self._features(s, ctx)
        p = float(self.model.predict_proba(X)[0, 1])
        out = {"win_probability": round(p, 4), "lose_probability": round(1 - p, 4),
               "status": "in_progress", "context": ctx, "explanation": None,
               "features": {k: float(v) for k, v in X.iloc[0].items()}}
        if explain and hasattr(self.model, "contributions"):
            out["explanation"] = self._explain(X)
        return out

    def predict_many(self, situations: list[Situation]) -> list[float]:
        """Vectorised probabilities for a list of in-progress situations (same teams/venue)."""
        s0 = situations[0]
        ctx = self.context(s0.batting_team, s0.bowling_team, s0.venue)
        frames = [self._features(s, ctx) for s in situations]
        import pandas as pd
        X = pd.concat(frames, ignore_index=True)
        return self.model.predict_proba(X)[:, 1].tolist()

    # ------------------------------------------------------------------
    def _explain(self, X) -> dict:
        parts, base = self.model.contributions(X)
        base = float(base[0])
        row = parts.iloc[0]
        base_p = _sigmoid(base)

        factors = []
        for group, cols in FEATURE_GROUPS.items():
            c = float(row[cols].sum())
            factors.append({"factor": group, "log_odds": round(c, 4),
                            "impact_pct": round((_sigmoid(base + c) - base_p) * 100, 2)})
        factors.sort(key=lambda f: abs(f["impact_pct"]), reverse=True)

        detail = [{"feature": FEATURE_LABELS[c], "value": round(float(X.iloc[0][c]), 3),
                   "log_odds": round(float(row[c]), 4)} for c in X.columns]
        detail.sort(key=lambda d: abs(d["log_odds"]), reverse=True)
        return {"baseline_win_probability": round(base_p, 4), "factors": factors,
                "detail": detail}

    # ------------------------------------------------------------------
    # what-if analysis
    # ------------------------------------------------------------------
    SCENARIOS = [
        ("Maiden over (0 runs)", 0, 0),
        ("Tight over (5 runs)", 5, 0),
        ("Par over (9 runs)", 9, 0),
        ("Big over (15 runs)", 15, 0),
        ("Wicket + 5 runs", 5, 1),
        ("Two wickets + 4 runs", 4, 2),
    ]

    def what_if(self, s: Situation, balls_ahead: int = 6) -> dict:
        base = self.predict(s, explain=False)
        balls_ahead = max(1, min(balls_ahead, C.TOTAL_BALLS - s.balls_bowled))
        out = []
        for label, runs, wkts in self.SCENARIOS:
            nxt = Situation(**{**s.__dict__,
                               "current_score": s.current_score + runs,
                               "wickets_fallen": min(10, s.wickets_fallen + wkts),
                               "balls_bowled": s.balls_bowled + balls_ahead})
            p = self.predict(nxt, explain=False)["win_probability"]
            out.append({"scenario": label, "runs": runs, "wickets": wkts,
                        "win_probability": round(p, 4),
                        "change_pct": round((p - base["win_probability"]) * 100, 2)})
        return {"current_win_probability": base["win_probability"],
                "balls_ahead": balls_ahead, "scenarios": out}

    def heatmap(self, s: Situation, wickets: int) -> dict:
        """Win probability over a grid of (runs still needed) x (balls left)."""
        runs_grid = list(range(10, min(s.target_score, 200) + 1, 10))
        balls_left_grid = list(range(6, 121, 6))
        sits = []
        for rr in runs_grid:
            for bl in balls_left_grid:
                sits.append(Situation(**{**s.__dict__, "current_score": s.target_score - rr,
                                         "wickets_fallen": wickets,
                                         "balls_bowled": C.TOTAL_BALLS - bl}))
        probs = np.array(self.predict_many(sits)).reshape(len(runs_grid), len(balls_left_grid))
        return {"runs_needed": runs_grid, "balls_left": balls_left_grid,
                "win_probability": probs.round(4).tolist(), "wickets_fallen": wickets}
