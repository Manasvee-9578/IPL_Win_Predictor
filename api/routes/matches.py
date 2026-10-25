"""Historical matches and ball-by-ball win-probability replay."""

from fastapi import APIRouter, HTTPException, Query

from src import database as db

router = APIRouter(prefix="/matches", tags=["history"])

_RESULT_SQL = """
    CASE WHEN m.winner IS NULL THEN m.result
         WHEN m.win_by_runs IS NOT NULL AND m.win_by_runs > 0
              THEN w.abbr || ' won by ' || CAST(m.win_by_runs AS INT) || ' runs'
         WHEN m.win_by_wickets IS NOT NULL AND m.win_by_wickets > 0
              THEN w.abbr || ' won by ' || CAST(m.win_by_wickets AS INT) || ' wickets'
         ELSE w.abbr || ' won (' || m.result || ')' END
"""

_BASE = f"""
    SELECT m.match_id, m.season, m.match_date, v.name AS venue,
           t1.name AS team1, t2.name AS team2, w.name AS winner,
           ct.name AS chasing_team, ft.name AS first_batting_team,
           m.first_runs, m.chase_runs, m.chase_wickets, m.chase_balls,
           m.valid_chase, m.split, {_RESULT_SQL} AS result_text
    FROM matches m
    JOIN venues v  ON v.venue_id = m.venue_id
    JOIN teams  t1 ON t1.team_id = m.team1
    JOIN teams  t2 ON t2.team_id = m.team2
    LEFT JOIN teams w  ON w.team_id  = m.winner
    LEFT JOIN teams ct ON ct.team_id = m.chase_team
    LEFT JOIN teams ft ON ft.team_id = m.first_team
"""


@router.get("")
def list_matches(
    season: int | None = None,
    team: str | None = Query(None, description="Team name (either side)"),
    only_replayable: bool = True,
    limit: int = Query(200, ge=1, le=1500),
):
    where, params = [], []
    if season:
        where.append("m.season = ?"); params.append(season)
    if team:
        where.append("(t1.name = ? OR t2.name = ?)"); params += [team, team]
    if only_replayable:
        where.append("m.valid_chase = 1")
    sql = _BASE + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY m.match_date DESC, m.match_id DESC LIMIT ?"
    with db.session() as conn:
        return {"status": "success", "matches": db.rows(conn, sql, (*params, limit))}


@router.get("/seasons")
def seasons():
    with db.session() as conn:
        data = db.rows(conn, "SELECT DISTINCT season FROM matches WHERE valid_chase=1 "
                             "ORDER BY season DESC")
    return {"seasons": [r["season"] for r in data]}


@router.get("/{match_id}/replay")
def replay(match_id: int):
    """Win-probability curve of the chase + the biggest turning points."""
    with db.session() as conn:
        meta = db.rows(conn, _BASE + " WHERE m.match_id = ?", (match_id,))
        if not meta:
            raise HTTPException(404, detail={"status": "error", "message": "Match not found"})
        meta = meta[0]
        if not meta["valid_chase"]:
            raise HTTPException(404, detail={"status": "error",
                                             "message": "No usable ball-by-ball chase for this match"})
        pts = db.rows(conn, "SELECT balls, runs, wickets, win_prob FROM chase_states "
                            "WHERE match_id = ? ORDER BY balls", (match_id,))

    target = int(meta["first_runs"]) + 1
    moments = []
    for prev, cur in zip(pts, pts[1:]):
        delta = cur["win_prob"] - prev["win_prob"]
        runs, wkts = cur["runs"] - prev["runs"], cur["wickets"] - prev["wickets"]
        what = ("WICKET" if wkts > 0 else "SIX/FOUR" if runs >= 4 else f"{runs} run(s)")
        moments.append({"over": f"{cur['balls'] // 6}.{cur['balls'] % 6}", "balls": cur["balls"],
                        "delta_pct": round(delta * 100, 1), "runs": runs, "wickets": wkts,
                        "event": what, "score": f"{cur['runs']}/{cur['wickets']}",
                        "win_prob": cur["win_prob"]})
    key = sorted(moments, key=lambda m: abs(m["delta_pct"]), reverse=True)[:6]
    return {"status": "success", "match": meta, "target": target, "points": pts,
            "key_moments": key,
            "note": ("Hold-out match: the model never saw it in training."
                     if meta["split"] == "test" else
                     "Training-period match: probabilities are in-sample.")}
