"""Reference / meta endpoints: health, teams, venues, model info, prediction history."""

from fastapi import APIRouter, Query

from api.state import get_predictor, state
from src.predictor import MODEL_NAME, MODEL_SCOPE
from src import database as db

router = APIRouter(tags=["meta"])


@router.get("/health")
def health():
    p = state.predictor
    return {
        "status": "ok" if p else "starting",
        "model_loaded": p is not None,
        "model": (p.metrics.get("selected_model") if p else None),
        "model_name": MODEL_NAME,
        "live_polling": state.live_enabled,
        "live_match_active": state.live is not None,
        "live_last_checked": state.live_checked_at,
    }


@router.get("/teams")
def teams():
    p = get_predictor()
    details = sorted(p.team_rows.values(), key=lambda t: (-t["active"], t["name"]))
    return {"status": "success",
            "teams": [t["name"] for t in details if t["active"]],
            "details": [{**t, "elo": round(t["elo"], 1)} for t in details]}


@router.get("/venues")
def venues():
    p = get_predictor()
    details = sorted(p.venue_rows.values(), key=lambda v: -v["matches"])
    return {"status": "success",
            "venues": [v["name"] for v in details],
            "details": details}


@router.get("/model/info")
def model_info():
    p = get_predictor()
    return {"status": "success", "model_name": MODEL_NAME, "scope": MODEL_SCOPE, "metrics": p.metrics,
            "features": list(p.model.feature_names_) if hasattr(p.model, "feature_names_") else None}


@router.get("/history")
def history(limit: int = Query(100, ge=1, le=1000), source: str | None = None):
    sql, params = "SELECT * FROM prediction_log", ()
    if source:
        sql, params = sql + " WHERE source = ?", (source,)
    sql += " ORDER BY id DESC LIMIT ?"
    with db.session() as conn:
        rows = db.rows(conn, sql, (*params, limit))
        stats = db.rows(conn, """
            SELECT COUNT(*) AS total, ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
                   ROUND(AVG(win_probability) * 100, 1) AS avg_win_pct
            FROM prediction_log""")[0]
        by_source = db.rows(conn, "SELECT source, COUNT(*) AS n FROM prediction_log GROUP BY source")
    return {"status": "success", "stats": stats, "by_source": by_source, "rows": rows}


@router.delete("/history")
def clear_history():
    with db.session() as conn:
        n = conn.execute("DELETE FROM prediction_log").rowcount
    return {"status": "success", "deleted": n}
