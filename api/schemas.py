"""
IPL Win Probability Predictor — Pydantic Schemas
=================================================
Request / response models for the FastAPI prediction endpoint.
"""

from pydantic import BaseModel, Field


class MatchState(BaseModel):
    """Current state of a live IPL match (2nd innings chase)."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batting_team": "Mumbai Indians",
                    "bowling_team": "Chennai Super Kings",
                    "venue": "Wankhede Stadium, Mumbai",
                    "toss_winner": "Mumbai Indians",
                    "target_score": 180,
                    "current_score": 90,
                    "wickets_fallen": 3,
                    "balls_bowled": 60,
                    "batting_team_form": 0.6,
                    "head_to_head_ratio": 0.45,
                }
            ]
        }
    }

    batting_team: str = Field(..., description="Name of the batting team")
    bowling_team: str = Field(..., description="Name of the bowling team")
    venue: str = Field(..., description="Match venue")
    toss_winner: str = Field(..., description="Team that won the toss")

    target_score: int = Field(..., gt=0, description="1st innings total + 1")
    current_score: int = Field(..., ge=0, description="Runs scored so far in the chase")
    wickets_fallen: int = Field(..., ge=0, le=10, description="Wickets lost (0-10)")
    balls_bowled: int = Field(..., ge=0, le=120, description="Balls bowled in 2nd innings (0-120)")

    batting_team_form: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Recent win ratio of the batting team (0-1)",
    )
    head_to_head_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Historical H2H win ratio for batting team vs bowling team (0-1)",
    )


class PredictionResponse(BaseModel):
    """Prediction output returned by the /predict endpoint."""

    status: str = Field(default="success", description="Response status")
    win_probability: float = Field(..., description="Batting team win probability")
    lose_probability: float = Field(..., description="Batting team lose probability")
    match_state_summary: str = Field(
        ..., description="Human-readable summary of the current match state"
    )


class LivePredictionResponse(PredictionResponse):
    """Extended response for the /live endpoint with freshness metadata."""

    is_live: bool = Field(True, description="Whether this is from a live match")
    last_updated: str = Field(
        ..., description="ISO timestamp of when the data was last fetched"
    )
