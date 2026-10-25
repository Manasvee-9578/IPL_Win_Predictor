"""
CricketService - one place that knows what real cricket data we have, where it came from and how fresh it is.

Rules it enforces
-----------------
* Only data returned by a provider is ever shown. Nothing is synthesised.
* Every provider reports a state (ok / no_key / invalid_key / quota / error) with a plain-language message.
* CricketData.org is refreshed adaptively so the free 100-hits/day budget lasts the whole day:
  10 min while something is live, 30 min otherwise, schedule pages every 6 h, never below a 10-hit reserve.
* The last good copy of every provider is cached in SQLite, so a restart or an outage still shows
  (clearly time-stamped) data instead of a blank page.
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

import requests

from src import database as db
from src.cricket import cricketdata as cd
from src.cricket import cricsheet
from src.cricket.classify import GROUPS, wanted
from src.cricket.models import CricketMatch

logger = logging.getLogger(__name__)

LIVE_INTERVAL = 600
IDLE_INTERVAL = 1800
SCHEDULE_EVERY = 6 * 3600
RESULTS_EVERY = 6 * 3600

MESSAGES = {
    "no_key": "No CricketData.org API key is configured, so live scores and upcoming fixtures are unavailable. "
              "Get a free key (100 requests/day, no card) at cricketdata.org and set CRICKETDATA_API_KEY in .env.",
    "invalid_key": "CricketData.org rejected the configured API key (\"Invalid API Key\"). Check CRICKETDATA_API_KEY "
                   "in .env - a free key from cricketdata.org works.",
    "quota": "Today's free CricketData.org quota is used up; live data resumes at 00:00 UTC.",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def _key(m: CricketMatch) -> tuple:
    return (m.date, frozenset(t.lower() for t in m.teams))


class CricketService:
    def __init__(self, client: cd.CricketDataClient | None = None, results_fetcher=None):
        self.client = client or cd.CricketDataClient()
        self._fetch_results = results_fetcher or cricsheet.fetch_recent
        self._lock = threading.RLock()
        self.matches: dict[str, CricketMatch] = {}
        self.providers = {
            "cricketdata": {"state": "no_key" if not cd.configured_key() else "pending", "message": "",
                            "last_success": None, "last_attempt": None},
            "cricsheet": {"state": "pending", "message": "", "last_success": None, "last_attempt": None},
        }
        self._last_schedule: datetime | None = None
        self._rejected_until: datetime | None = None
        self.load_cache()

    # ------------------------------------------------------------------ persistence
    def load_cache(self) -> None:
        try:
            with db.session() as conn:
                rows = db.rows(conn, "SELECT provider, payload, fetched_at FROM external_matches")
        except Exception as exc:                       # pragma: no cover - table missing on a very old DB
            logger.warning("Could not load cached matches: %s", exc)
            return
        with self._lock:
            for r in rows:
                m = CricketMatch.from_dict(json.loads(r["payload"]))
                self.matches[f"{m.source}:{m.id}"] = m
                p = self.providers[r["provider"]]
                p["last_success"] = max(p["last_success"] or "", r["fetched_at"])
            for p in self.providers.values():
                if p["last_success"] and p["state"] == "pending":
                    p["state"] = "cached"

    def _persist(self, provider: str) -> None:
        with self._lock:
            mine = [m for m in self.matches.values() if m.source == provider]
        with db.session() as conn:
            conn.execute("DELETE FROM external_matches WHERE provider = ?", (provider,))
            conn.executemany(
                "INSERT INTO external_matches (provider, match_key, state, start_utc, payload, fetched_at) "
                "VALUES (?,?,?,?,?,?)",
                [(provider, m.id, m.state, m.start_utc, json.dumps(m.to_dict()), m.fetched_at) for m in mine])

    # ------------------------------------------------------------------ CricketData.org
    def refresh_cricketdata(self) -> list[dict] | None:
        """Fetch live/current matches (and the schedule when due). Returns the raw current matches."""
        p = self.providers["cricketdata"]
        now = _now()
        if not cd.configured_key():
            p["state"], p["message"] = "no_key", MESSAGES["no_key"]
            return None
        if self._rejected_until and now < self._rejected_until:
            return None
        if cd.hits_left() <= 0:
            p["state"], p["message"] = "quota", MESSAGES["quota"]
            return None
        p["last_attempt"] = _iso(now)
        stamp = _iso(now)
        try:
            raw = self.client.current()
            items = list(raw)
            if (self._last_schedule is None or (now - self._last_schedule).total_seconds() > SCHEDULE_EVERY) \
                    and cd.hits_left() >= 3:
                items += self.client.schedule(pages=2)
                self._last_schedule = now
        except cd.KeyRejected as exc:
            p["state"], p["message"] = "invalid_key", MESSAGES["invalid_key"]
            self._rejected_until = now + timedelta(hours=6)
            logger.warning("CricketData.org rejected the API key: %s", exc)
            return None
        except cd.QuotaExceeded:
            p["state"], p["message"] = "quota", MESSAGES["quota"]
            return None
        except (cd.ProviderError, requests.RequestException) as exc:
            p["state"], p["message"] = "error", f"CricketData.org is unreachable right now ({exc})."
            logger.warning("CricketData.org error: %s", exc)
            return None

        fresh = {}
        for item in items:
            m = cd.parse_item(item, stamp)
            if m:
                fresh[f"{m.source}:{m.id}"] = m
        with self._lock:
            self.matches.update(fresh)
            # forget upcoming fixtures whose start time is long past and that the provider no longer lists
            for k, m in list(self.matches.items()):
                if m.source == "cricketdata" and k not in fresh and m.state == "upcoming" and m.start_utc \
                        and m.start_utc < _iso(now - timedelta(hours=12)):
                    del self.matches[k]
        self._persist("cricketdata")
        p.update(state="ok", message="", last_success=stamp)
        return raw

    # ------------------------------------------------------------------ Cricsheet
    def refresh_results(self) -> None:
        p = self.providers["cricsheet"]
        now = _now()
        p["last_attempt"] = _iso(now)
        try:
            found = self._fetch_results(30, fetched_at=_iso(now))
        except Exception as exc:
            p["state"], p["message"] = "error", f"Cricsheet is unreachable right now ({exc.__class__.__name__})."
            logger.warning("Cricsheet error: %s", exc)
            return
        with self._lock:
            for k in [k for k, m in self.matches.items() if m.source == "cricsheet"]:
                del self.matches[k]
            self.matches.update({f"{m.source}:{m.id}": m for m in found})
        self._persist("cricsheet")
        p.update(state="ok", message="", last_success=_iso(now))

    # ------------------------------------------------------------------ scheduling
    def next_delay(self) -> int:
        """Seconds until the next CricketData.org poll."""
        state = self.providers["cricketdata"]["state"]
        if state in ("no_key", "invalid_key"):
            return 3600
        if state == "quota":
            tomorrow = (_now() + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
            return max(300, int((tomorrow - _now()).total_seconds()))
        live = any(m.state == "live" and m.source == "cricketdata" for m in self.matches.values())
        delay = LIVE_INTERVAL if live else IDLE_INTERVAL
        if cd.hits_left() < 15:
            delay = max(delay, 3600)
        return delay

    # ------------------------------------------------------------------ queries
    def snapshot(self, state: str = "all", groups: set[str] | None = None, formats: set[str] | None = None,
                 limit: int = 60) -> dict:
        groups = set(GROUPS) if groups is None else groups
        formats = {"T20I", "ODI", "TEST", "IPL"} if formats is None else formats
        with self._lock:
            pool = list(self.matches.values())
        pool = [m for m in pool if wanted(m.groups, m.format_base, groups, formats)]

        sheet_keys = {_key(m) for m in pool if m.source == "cricsheet"}
        pool = [m for m in pool if not (m.source == "cricketdata" and m.state == "result" and _key(m) in sheet_keys)]

        def sort_key(m: CricketMatch):
            return m.start_utc or m.date or ""

        out = {
            "live": sorted([m for m in pool if m.state == "live"], key=sort_key, reverse=True),
            "upcoming": sorted([m for m in pool if m.state == "upcoming"], key=sort_key),
            "result": sorted([m for m in pool if m.state == "result"], key=sort_key, reverse=True),
        }
        counts = {k: len(v) for k, v in out.items()}
        if state != "all":
            out = {state: out[state]}
        return {"matches": {k: [m.to_dict() for m in v[:limit]] for k, v in out.items()}, "counts": counts}

    def status(self) -> dict:
        sheet = [m for m in self.matches.values() if m.source == "cricsheet"]
        through = max((m.date or "" for m in sheet), default=None)
        cdp, csp = dict(self.providers["cricketdata"]), dict(self.providers["cricsheet"])
        cdp.update(configured=bool(cd.configured_key()), hits_today=cd.hits_today(), hits_limit=cd.daily_limit(),
                   next_refresh_in=self.next_delay(),
                   coverage="Live scores and upcoming fixtures, as listed by CricketData.org's free tier. "
                            "Women's and India A coverage depends on what the provider lists.")
        if not cdp["message"] and cdp["state"] == "cached":
            cdp["message"] = "Showing the last data fetched earlier; waiting for the next refresh."
        csp.update(matches=len(sheet), data_through=through or None,
                   coverage="Completed international and IPL matches, published by Cricsheet with a delay of days. "
                            "India A appears only if Cricsheet carries the match.")
        return {"cricketdata": cdp, "cricsheet": csp,
                "attribution": "Results data: Cricsheet.org (ODC-BY). Live scores and fixtures: CricketData.org."}
