"""Pydantic request / response models for the FastAPI backend (ChaseIQ)."""

from pydantic import BaseModel, Field, model_validator


class MatchState(BaseModel):
    """Situation in the 2nd innings (the chase)."""

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "batting_team": "Mumbai Indians",
                "bowling_team": "Chennai Super Kings",
                "venue": "Wankhede Stadium, Mumbai",
                "toss_winner": "Mumbai Indians",
                "target_score": 180,
                "current_score": 90,
                "wickets_fallen": 3,
                "balls_bowled": 60,
            }]
        }
    }

    batting_team: str = Field(..., description="Chasing team (name, abbreviation or legacy name)")
    bowling_team: str = Field(..., description="Defending team")
    venue: str = Field(..., description="Match venue (any common spelling)")
    toss_winner: str | None = Field(None, description="Team that won the toss")

    target_score: int = Field(..., gt=0, le=400, description="1st-innings total + 1")
    current_score: int = Field(..., ge=0, le=500, description="Runs scored so far in the chase")
    wickets_fallen: int = Field(..., ge=0, le=10)
    balls_bowled: int = Field(..., ge=0, le=120, description="Legal balls bowled (0-120)")

    competition: str = Field(
        "IPL", description="Must be IPL. The IPL Second-Innings Chase Model is never applied to ODI, T20I or Test.")

    batting_team_form: float | None = Field(
        None, ge=0, le=1, description="Optional override; default = from the database")
    head_to_head_ratio: float | None = Field(
        None, ge=0, le=1, description="Optional override; default = from the database")

    @model_validator(mode="after")
    def _ipl_only(self):
        if self.competition.strip().lower() not in ("ipl", "indian premier league"):
            raise ValueError("ChaseIQ's IPL Second-Innings Chase Model only supports IPL matches; it is never "
                             "applied to ODI, T20I or Test cricket")
        return self

    @model_validator(mode="after")
    def _teams_differ(self):
        if self.batting_team.strip().lower() == self.bowling_team.strip().lower():
            raise ValueError("batting_team and bowling_team must be different")
        return self


class Factor(BaseModel):
    factor: str
    log_odds: float
    impact_pct: float = Field(..., description="Effect on win probability vs. an average situation, in points")


class Explanation(BaseModel):
    baseline_win_probability: float
    factors: list[Factor]
    detail: list[dict]


class PredictionResponse(BaseModel):
    status: str = "success"
    model: str = "IPL Second-Innings Chase Model"
    mode: str = Field("hypothetical", description="live | historical | hypothetical")
    match_status: str = Field(..., description="in_progress | chase_complete | chase_failed | tied")
    win_probability: float
    lose_probability: float
    match_state_summary: str
    context: dict
    explanation: Explanation | None = None
    latency_ms: float | None = None


class WhatIfResponse(BaseModel):
    current_win_probability: float
    balls_ahead: int
    scenarios: list[dict]
