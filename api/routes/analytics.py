"""Analytics endpoints - everything here is a SQL query over the database."""

from fastapi import APIRouter, HTTPException

from src import database as db
from src.reference import team_id

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def overview():
    with db.session() as conn:
        totals = db.rows(conn, """
            SELECT (SELECT COUNT(*) FROM matches)                   AS matches,
                   (SELECT COUNT(*) FROM matches WHERE valid_chase) AS chases,
                   (SELECT COUNT(DISTINCT season) FROM matches)     AS seasons,
                   (SELECT COUNT(*) FROM venues WHERE matches > 0)  AS venues,
                   (SELECT COUNT(*) FROM chase_states)              AS ball_states""")[0]
        by_season = db.rows(conn, """
            SELECT season, COUNT(*) AS chases,
                   ROUND(AVG(chase_team = winner) * 100, 1) AS chase_win_pct,
                   ROUND(AVG(first_runs), 1)                AS avg_first_innings
            FROM matches WHERE valid_chase = 1 GROUP BY season ORDER BY season""")
        toss = db.rows(conn, """
            SELECT ROUND(AVG(toss_winner = winner) * 100, 1) AS toss_winner_wins_pct,
                   ROUND(AVG(toss_decision = 'field') * 100, 1) AS chose_to_field_pct
            FROM matches WHERE result = 'win'""")[0]
    return {"totals": totals, "by_season": by_season, "toss": toss}


@router.get("/teams")
def teams():
    with db.session() as conn:
        data = db.rows(conn, """
            SELECT t.team_id, t.name, t.abbr, t.color, t.active, ROUND(t.elo, 1) AS elo,
                   t.matches, t.wins,
                   ROUND(100.0 * t.wins / NULLIF(t.matches, 0), 1) AS win_pct,
                   (SELECT COUNT(*) FROM matches m WHERE m.valid_chase AND m.chase_team = t.team_id)
                       AS chases,
                   (SELECT COUNT(*) FROM matches m WHERE m.valid_chase AND m.chase_team = t.team_id
                       AND m.winner = t.team_id) AS chases_won,
                   (SELECT COUNT(*) FROM matches m WHERE m.valid_chase AND m.first_team = t.team_id)
                       AS defends,
                   (SELECT COUNT(*) FROM matches m WHERE m.valid_chase AND m.first_team = t.team_id
                       AND m.winner = t.team_id) AS defends_won
            FROM teams t ORDER BY t.elo DESC""")
    for r in data:
        r["chase_win_pct"] = round(100 * r["chases_won"] / r["chases"], 1) if r["chases"] else None
        r["defend_win_pct"] = round(100 * r["defends_won"] / r["defends"], 1) if r["defends"] else None
    return {"teams": data}


@router.get("/venues")
def venues():
    with db.session() as conn:
        data = db.rows(conn, """
            SELECT v.name, v.city, v.matches, ROUND(v.avg_first_innings, 1) AS avg_first_innings,
                   ROUND(v.chase_bias * 100, 1) AS chase_win_pct,
                   (SELECT MAX(first_runs) FROM matches m WHERE m.venue_id = v.venue_id) AS highest_total
            FROM venues v WHERE v.matches >= 5 ORDER BY v.matches DESC""")
    return {"venues": data}


@router.get("/head-to-head")
def head_to_head(team_a: str, team_b: str):
    a, b = team_id(team_a), team_id(team_b)
    if not a or not b or a == b:
        raise HTTPException(422, detail={"status": "error", "message": "Pick two different teams"})
    with db.session() as conn:
        summary = dict(conn.execute("""
            SELECT COUNT(*) AS games, SUM(winner = ?1) AS a_wins, SUM(winner = ?2) AS b_wins,
                   ROUND(AVG(first_runs), 1) AS avg_first_innings
            FROM matches
            WHERE ((team1 = ?1 AND team2 = ?2) OR (team1 = ?2 AND team2 = ?1))
              AND result IN ('win', 'tie')""", (a, b)).fetchone())
        recent = db.rows(conn, """
            SELECT m.match_id, m.season, m.match_date, w.name AS winner
            FROM matches m LEFT JOIN teams w ON w.team_id = m.winner
            WHERE ((m.team1 = ?1 AND m.team2 = ?2) OR (m.team1 = ?2 AND m.team2 = ?1))
              AND m.result IN ('win', 'tie')
            ORDER BY m.match_date DESC LIMIT 8""", (a, b))
    return {"summary": summary, "recent": recent}


@router.get("/turning-points")
def turning_points(limit: int = 8):
    """Greatest escapes and biggest collapses, found with a single SQL aggregate.

    For every chase we look at the worst (best) probability the eventual winner
    (loser) ever had.  Probabilities are the model's, so these are model-rated
    'comebacks', not official records.
    """
    base = """
        SELECT m.match_id, m.season, m.split, bt.name AS chasing_team, ot.name AS opponent,
               v.name AS venue, m.first_runs + 1 AS target, {agg}(cs.win_prob) AS extreme_prob
        FROM matches m
        JOIN chase_states cs ON cs.match_id = m.match_id
                            AND cs.win_prob > 0 AND cs.win_prob < 1
        JOIN teams bt ON bt.team_id = m.chase_team
        JOIN teams ot ON ot.team_id = m.first_team
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE m.valid_chase = 1 AND {cond}
        GROUP BY m.match_id
        ORDER BY extreme_prob {order} LIMIT ?"""
    with db.session() as conn:
        escapes = db.rows(conn, base.format(agg="MIN", cond="m.chase_team = m.winner",
                                            order="ASC"), (limit,))
        collapses = db.rows(conn, base.format(agg="MAX", cond="m.chase_team <> m.winner",
                                              order="DESC"), (limit,))
    return {"greatest_escapes": escapes, "biggest_collapses": collapses}
