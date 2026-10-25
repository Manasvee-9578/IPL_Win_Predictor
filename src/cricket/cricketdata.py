"""
CricketData.org (CricAPI) client - live scores and fixtures.

Verified facts (cricketdata.org/pricing, checked 2026-10-25):
  * "Lifetime Free" plan, $0.00/month, 100 hits per day, 1 hit = 1 API call
  * free data is "always a few minutes behind real-time"
Not verifiable without a valid key: whether women's / India A fixtures are included.
This module therefore reports exactly what the provider returns and never fills gaps.

The 100-hit daily budget is persisted in SQLite (``api_usage``) so restarts cannot overspend it.
"""

import os
from datetime import datetime, timezone

import requests

from src import database as db
from src.cricket.classify import format_for, groups_for, is_ipl
from src.cricket.models import CricketMatch

BASE = "https://api.cricapi.com/v1"
PROVIDER = "cricketdata"


class ProviderError(Exception):
    pass


class KeyRejected(ProviderError):
    pass


class QuotaExceeded(ProviderError):
    pass


def configured_key() -> str:
    """CRICKETDATA_API_KEY wins; CRICKET_API_KEY (the old name) is still honoured."""
    for var in ("CRICKETDATA_API_KEY", "CRICKET_API_KEY"):
        key = os.getenv(var, "").strip()
        if key and key != "your_key_here":
            return key
    return ""


def daily_limit() -> int:
    return int(os.getenv("CRICKETDATA_DAILY_LIMIT", "100"))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def hits_today() -> int:
    with db.session() as conn:
        row = conn.execute("SELECT hits FROM api_usage WHERE day=? AND provider=?",
                           (_today(), PROVIDER)).fetchone()
    return int(row[0]) if row else 0


def _set_hits(n: int) -> None:
    with db.session() as conn:
        conn.execute("INSERT INTO api_usage (day, provider, hits) VALUES (?,?,?) "
                     "ON CONFLICT(day, provider) DO UPDATE SET hits = MAX(hits, excluded.hits)",
                     (_today(), PROVIDER, n))


def hits_left(reserve: int = 10) -> int:
    """Hits we are willing to spend today (a safety reserve is never touched)."""
    return max(0, daily_limit() - reserve - hits_today())


class CricketDataClient:
    def __init__(self, http: requests.Session | None = None):
        self.http = http or requests.Session()

    def get(self, endpoint: str, **params) -> dict:
        key = configured_key()
        if not key:
            raise KeyRejected("no API key configured")
        if hits_left(reserve=0) <= 0:
            raise QuotaExceeded("daily hit budget used")
        _set_hits(hits_today() + 1)                    # count first: a failed call may still be billed
        try:
            resp = self.http.get(f"{BASE}/{endpoint}", params={"apikey": key, **params}, timeout=15)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise ProviderError(f"{exc.__class__.__name__} while calling /{endpoint}") from exc

        info = data.get("info") or {}
        if isinstance(info.get("hitsToday"), int):     # trust the provider's own counter if present
            _set_hits(info["hitsToday"])
        if data.get("status") != "success":
            reason = str(data.get("reason") or "unknown error")
            low = reason.lower()
            if "api key" in low or "apikey" in low:
                raise KeyRejected(reason)
            if "hit" in low or "limit" in low or "quota" in low:
                _set_hits(daily_limit())
                raise QuotaExceeded(reason)
            raise ProviderError(reason)
        return data

    def current(self) -> list[dict]:
        return self.get("currentMatches", offset=0).get("data") or []

    def schedule(self, pages: int = 2) -> list[dict]:
        out: list[dict] = []
        for page in range(pages):
            out += self.get("matches", offset=page * 25).get("data") or []
        return out


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def _innings_team(label: str) -> str:
    import re
    return re.sub(r"\s+inning[s]?\s*\d*\s*$", "", label or "", flags=re.I).strip()


def parse_item(m: dict, fetched_at: str) -> CricketMatch | None:
    """CricAPI match object -> CricketMatch. None when it is not India / India A / IPL."""
    teams = list(m.get("teams") or [t.get("name") for t in (m.get("teamInfo") or []) if t.get("name")])
    if len(teams) < 2:
        return None
    ipl = is_ipl(str(m.get("series", "")), str(m.get("name", "")), teams)
    groups = groups_for(teams)
    if not groups and not ipl:
        return None
    base, label = format_for(m.get("matchType", ""), groups, ipl)
    if base == "OTHER":
        return None

    ended, started = bool(m.get("matchEnded")), bool(m.get("matchStarted"))
    state = "result" if ended else "live" if started else "upcoming"
    gmt = m.get("dateTimeGMT")
    start = (gmt if str(gmt).endswith("Z") else f"{gmt}Z") if gmt else None
    scores = []
    for i, s in enumerate(m.get("score") or [], 1):
        scores.append({"team": _innings_team(s.get("inning", "")), "runs": s.get("r"), "wickets": s.get("w"),
                       "overs": s.get("o"), "innings": i})
    status = str(m.get("status") or "")
    return CricketMatch(
        id=str(m.get("id")), source=PROVIDER, state=state, name=str(m.get("name") or " v ".join(teams)),
        teams=teams, groups=groups, format_base=base, format_label=label, series=str(m.get("series", "")),
        venue=m.get("venue"), date=str(m.get("date") or (gmt or "")[:10]) or None, start_utc=start,
        status_text=status, result_text=status if ended else None, scores=scores,
        gender="female" if "women" in " ".join(teams).lower() else "male", fetched_at=fetched_at)
