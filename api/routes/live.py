"""Live IPL chase endpoint and the background pollers (CricketData.org live + Cricsheet results)."""

import asyncio
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.state import describe, state
from src import database as db
from src.live_data import find_ipl_live_chase
from src.predictor import Situation

router = APIRouter(prefix="/live", tags=["live"])
logger = logging.getLogger(__name__)

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))


def evaluate_live(match: dict) -> dict:
    """Run the model on a parsed live match and persist a snapshot if it changed."""
    p = state.predictor
    s = Situation(
        batting_team=match["batting_team"], bowling_team=match["bowling_team"],
        venue=match["venue"], toss_winner=match["toss_winner"],
        target_score=match["target_score"], current_score=match["current_score"],
        wickets_fallen=match["wickets_fallen"], balls_bowled=match["balls_bowled"])
    result = p.predict(s, explain=True)

    live = {
        "batting_team": p.team_rows[s.batting_team]["name"],
        "bowling_team": p.team_rows[s.bowling_team]["name"],
        "venue": s.venue,
        "target_score": s.target_score, "current_score": s.current_score,
        "wickets_fallen": s.wickets_fallen, "balls_bowled": s.balls_bowled,
        "win_probability": result["win_probability"],
        "lose_probability": result["lose_probability"],
        "match_status": result["status"],
        "match_state_summary": describe(p, s, result["status"]),
        "explanation": result["explanation"],
        "match_name": match.get("match_name", ""),
        "is_live": True,
        "model": "IPL Second-Innings Chase Model",
        "last_updated": match["last_updated"],
    }
    with db.session() as conn:
        last = conn.execute(
            "SELECT batting_team, score, wickets, balls FROM live_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        key = (live["batting_team"], s.current_score, s.wickets_fallen, s.balls_bowled)
        if last is None or tuple(last) != key:
            conn.execute(
                "INSERT INTO live_snapshots (batting_team, bowling_team, venue, target, score,"
                " wickets, balls, win_probability) VALUES (?,?,?,?,?,?,?,?)",
                (live["batting_team"], live["bowling_team"], s.venue, s.target_score,
                 s.current_score, s.wickets_fallen, s.balls_bowled, live["win_probability"]))
    return live


async def poll_forever() -> None:
    """Background task: CricketData.org -> live matches cache; the IPL chase (if any) is also scored."""
    loop = asyncio.get_running_loop()
    svc = state.cricket
    logger.info("Live polling started (adaptive interval, free-tier budget aware)")
    while True:
        try:
            raw = await loop.run_in_executor(None, svc.refresh_cricketdata)
            if raw is not None:
                chase = find_ipl_live_chase(raw)
                state.live = await loop.run_in_executor(None, evaluate_live, chase) if chase else None
                state.live_checked_at = datetime.now(timezone.utc).isoformat()
        except asyncio.CancelledError:
            raise
        except Exception as exc:                      # keep the loop alive no matter what
            logger.error("Live poll error: %s", exc)
        await asyncio.sleep(svc.next_delay())


async def results_forever() -> None:
    """Background task: completed matches from Cricsheet every few hours."""
    loop = asyncio.get_running_loop()
    from src.cricket.service import RESULTS_EVERY
    while True:
        try:
            await loop.run_in_executor(None, state.cricket.refresh_results)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Results refresh error: %s", exc)
        await asyncio.sleep(RESULTS_EVERY)


@router.get("")
def live():
    """Latest cached live-match prediction (404 when no IPL chase is in progress)."""
    if state.live is None:
        raise HTTPException(404, detail={
            "status": "error", "message": "No live IPL chase in progress",
            "polling": state.live_enabled, "last_checked": state.live_checked_at})
    return {"status": "success", "data": state.live}


@router.get("/history")
def live_history(limit: int = Query(300, ge=1, le=2000)):
    """Probability trail of the most recent live match (latest teams in the table)."""
    with db.session() as conn:
        latest = conn.execute(
            "SELECT batting_team, bowling_team FROM live_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if latest is None:
            return {"status": "success", "points": []}
        pts = db.rows(conn, """
            SELECT created_at, score, wickets, balls, target, win_probability
            FROM live_snapshots WHERE batting_team = ? AND bowling_team = ?
            ORDER BY id DESC LIMIT ?""", (latest["batting_team"], latest["bowling_team"], limit))
    return {"status": "success", "batting_team": latest["batting_team"],
            "bowling_team": latest["bowling_team"], "points": pts[::-1]}
