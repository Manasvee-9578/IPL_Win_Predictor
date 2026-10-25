"""Prediction endpoints: /predict, /whatif, /heatmap, /context."""

import logging
import time

from fastapi import APIRouter, HTTPException, Query

from api.schemas import MatchState, PredictionResponse, WhatIfResponse
from api.state import describe, get_predictor, to_situation
from src import database as db
from src.predictor import PredictionError

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)


def _situation(body: MatchState):
    p = get_predictor()
    try:
        return p, to_situation(p, body)
    except PredictionError as exc:
        raise HTTPException(422, detail={"status": "error", "message": str(exc)})


def log_prediction(p, s, result: dict, source: str, latency_ms: float) -> None:
    """Persist one evaluated scenario. Never lets a DB problem break a prediction."""
    try:
        with db.session() as conn:
            conn.execute(
                "INSERT INTO prediction_log (source, batting_team, bowling_team, venue, target,"
                " score, wickets, balls, win_probability, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (source, p.team_rows[s.batting_team]["name"], p.team_rows[s.bowling_team]["name"],
                 s.venue, s.target_score, s.current_score, s.wickets_fallen, s.balls_bowled,
                 result["win_probability"], round(latency_ms, 2)))
    except Exception as exc:                          # pragma: no cover
        logger.warning("Could not log prediction: %s", exc)


@router.post("/predict", response_model=PredictionResponse)
def predict(
    body: MatchState,
    log: bool = Query(True, description="Store this scenario in prediction_log"),
    explain: bool = Query(True, description="Include the factor-by-factor explanation"),
    source: str = Query("api", max_length=20),
):
    """Win / lose probability for the chasing team, with an explanation."""
    t0 = time.perf_counter()
    p, s = _situation(body)
    try:
        result = p.predict(s, explain=explain)
    except PredictionError as exc:
        raise HTTPException(422, detail={"status": "error", "message": str(exc)})
    latency = (time.perf_counter() - t0) * 1000

    if log:
        log_prediction(p, s, result, source, latency)
    return PredictionResponse(
        match_status=result["status"],
        win_probability=result["win_probability"],
        lose_probability=result["lose_probability"],
        match_state_summary=describe(p, s, result["status"]),
        context=result["context"],
        explanation=result["explanation"],
        latency_ms=round(latency, 2),
    )


@router.post("/whatif", response_model=WhatIfResponse)
def whatif(body: MatchState, balls_ahead: int = Query(6, ge=1, le=30)):
    """How the win probability would change after the next over (or `balls_ahead` balls)."""
    p, s = _situation(body)
    if p.terminal(s):
        raise HTTPException(422, detail={"status": "error", "message": "The chase is already decided"})
    return p.what_if(s, balls_ahead)


@router.post("/heatmap")
def heatmap(body: MatchState, wickets: int = Query(3, ge=0, le=9)):
    """Win probability across every (runs needed x balls left) combination."""
    p, s = _situation(body)
    return p.heatmap(s, wickets)


@router.get("/context")
def context(batting_team: str, bowling_team: str, venue: str):
    """Team-strength / ground context the model would use for this fixture."""
    p = get_predictor()
    try:
        bat, bowl = p.resolve_team(batting_team), p.resolve_team(bowling_team)
    except PredictionError as exc:
        raise HTTPException(422, detail={"status": "error", "message": str(exc)})
    return p.context(bat, bowl, venue)
