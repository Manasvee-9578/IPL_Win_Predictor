"""Live Cricket endpoints: India Men / Women / A and IPL - live, upcoming and completed matches."""

from fastapi import APIRouter, HTTPException, Query

from api.state import state
from src.cricket.classify import FORMATS, GROUPS

router = APIRouter(prefix="/cricket", tags=["cricket"])


def _csv(value: str | None, allowed: set[str], upper: bool = False) -> set[str]:
    if value is None:
        return set(allowed)
    items = {v.strip().upper() if upper else v.strip().lower() for v in value.split(",") if v.strip()}
    bad = items - allowed
    if bad:
        raise HTTPException(422, detail={"status": "error", "message": f"Unknown filter value(s): {sorted(bad)}"})
    return items


@router.get("/matches")
def matches(
    state_: str = Query("all", alias="state", pattern="^(all|live|upcoming|result)$"),
    teams: str | None = Query(None, description="comma list of india_men,india_women,india_a"),
    formats: str | None = Query(None, description="comma list of T20I,ODI,TEST,IPL"),
    limit: int = Query(60, ge=1, le=200),
):
    """Real matches only - whatever the providers returned. Empty lists mean 'none available', never filler."""
    svc = state.cricket
    snap = svc.snapshot(state_, _csv(teams, set(GROUPS)), _csv(formats, set(FORMATS), upper=True), limit)
    return {"status": "success", **snap, "providers": svc.status()}


@router.get("/status")
def status():
    """Where the data comes from, its limits, and whether each source is currently working."""
    return {"status": "success", **state.cricket.status()}
