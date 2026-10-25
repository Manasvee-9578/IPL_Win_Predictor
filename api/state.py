"""Process-wide objects shared by the routers (model, live-match cache)."""

from dataclasses import dataclass, field

from src.cricket.service import CricketService
from src.predictor import Predictor, Situation


@dataclass
class AppState:
    predictor: Predictor | None = None
    live: dict | None = None                 # latest live prediction (None = no live match)
    live_checked_at: str | None = None
    live_enabled: bool = False
    cricket: CricketService | None = None
    extras: dict = field(default_factory=dict)


state = AppState()


def get_predictor() -> Predictor:
    if state.predictor is None:
        raise RuntimeError("Predictor not initialised")
    return state.predictor


def to_situation(p: Predictor, body) -> Situation:
    """Convert a validated ``MatchState`` into the engine's ``Situation``."""
    toss = p.resolve_team(body.toss_winner) if body.toss_winner else None
    return Situation(
        batting_team=p.resolve_team(body.batting_team),
        bowling_team=p.resolve_team(body.bowling_team),
        venue=body.venue, toss_winner=toss,
        target_score=body.target_score, current_score=body.current_score,
        wickets_fallen=body.wickets_fallen, balls_bowled=body.balls_bowled,
        batting_team_form=body.batting_team_form,
        head_to_head_ratio=body.head_to_head_ratio,
    )


def describe(p: Predictor, s: Situation, status: str) -> str:
    bat, bowl = p.team_rows[s.batting_team]["name"], p.team_rows[s.bowling_team]["name"]
    overs = f"{s.balls_bowled // 6}.{s.balls_bowled % 6}"
    score = f"{s.current_score}/{s.wickets_fallen}"
    if status == "chase_complete":
        return f"{bat} have reached the target of {s.target_score} ({score} in {overs} overs)"
    if status == "chase_failed":
        return f"{bat} fell short: {score} in {overs} overs, target {s.target_score}"
    if status == "tied":
        return f"Scores level at {s.current_score} - {bat} vs {bowl} goes to a Super Over"
    return (f"{bat} need {s.target_score - s.current_score} runs from "
            f"{120 - s.balls_bowled} balls ({score} after {overs} overs) "
            f"vs {bowl} at {s.venue}")
