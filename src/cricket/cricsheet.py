"""
Cricsheet - completed matches (results), no API key.

Verified (2026-10-25): https://cricsheet.org/downloads/recently_added_30_json.zip returns every match
Cricsheet added in the last 30 days (227 matches, including 14 India men's / women's games) with
ball-by-ball data. Cricsheet publishes with a delay of days, so this is a *results* source - never live.

Licence: Open Data Commons Attribution (ODC-BY) - attribution is required; the UI credits
"Data: Cricsheet.org (ODC-BY)".  India A matches are only listed when Cricsheet carries them.
"""

import io
import json
import zipfile

import requests

from src.cricket.classify import format_for, groups_for
from src.cricket.models import CricketMatch

PROVIDER = "cricsheet"
URL = "https://cricsheet.org/downloads/recently_added_{days}_json.zip"
USER_AGENT = "ChaseIQ/1.0 (cricket analytics student project; contact via GitHub)"


def fetch_recent(days: int = 30, http: requests.Session | None = None, fetched_at: str = "") -> list[CricketMatch]:
    resp = (http or requests).get(URL.format(days=days), headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return parse_zip(resp.content, fetched_at)


def parse_zip(content: bytes, fetched_at: str = "") -> list[CricketMatch]:
    out = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            m = parse_match(name[:-5], json.loads(z.read(name)), fetched_at)
            if m:
                out.append(m)
    return out


def _outcome_text(info: dict) -> str:
    o = info.get("outcome") or {}
    method = f" ({o['method']})" if o.get("method") else ""
    if "winner" in o:
        by = o.get("by") or {}
        if "runs" in by:
            return f"{o['winner']} won by {by['runs']} runs{method}"
        if "wickets" in by:
            return f"{o['winner']} won by {by['wickets']} wickets{method}"
        if "innings" in by:
            return f"{o['winner']} won by an innings and {by.get('runs', 0)} runs{method}"
        return f"{o['winner']} won{method}"
    if o.get("result") == "tie":
        return "Match tied" + (f" - {o['eliminator']} won the super over" if o.get("eliminator") else "")
    return {"no result": "No result", "draw": "Match drawn"}.get(o.get("result", ""), "Result not recorded")


def _innings_scores(j: dict) -> list[dict]:
    rows = []
    for n, inn in enumerate(j.get("innings", []), 1):
        if inn.get("super_over"):
            continue
        runs = wkts = legal = 0
        for over in inn.get("overs", []):
            for d in over.get("deliveries", []):
                runs += d["runs"]["total"]
                wkts += len(d.get("wickets", []))
                extras = d.get("extras") or {}
                legal += 0 if ("wides" in extras or "noballs" in extras) else 1
        rows.append({"team": inn["team"], "runs": runs, "wickets": wkts,
                     "overs": f"{legal // 6}.{legal % 6}", "innings": n})
    return rows


def parse_match(key: str, j: dict, fetched_at: str = "") -> CricketMatch | None:
    info = j.get("info") or {}
    teams = list(info.get("teams") or [])
    if len(teams) != 2:
        return None
    gender = info.get("gender")
    event = (info.get("event") or {}).get("name", "")
    ipl = event.lower() == "indian premier league"
    groups = groups_for(teams, gender)
    if not groups and not ipl:
        return None
    base, label = format_for(info.get("match_type", ""), groups, ipl)
    if base == "OTHER":
        return None
    dates = info.get("dates") or []
    venue = ", ".join(x for x in (info.get("venue"), info.get("city")) if x and x not in str(info.get("venue")))
    text = _outcome_text(info)
    return CricketMatch(
        id=key, source=PROVIDER, state="result", name=f"{teams[0]} v {teams[1]}", teams=teams, groups=groups,
        format_base=base, format_label=label, series=event, venue=venue or info.get("venue"),
        date=str(dates[-1]) if dates else None, status_text=text, result_text=text,
        scores=_innings_scores(j), gender=gender, fetched_at=fetched_at)
