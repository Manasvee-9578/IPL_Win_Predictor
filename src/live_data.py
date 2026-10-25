"""
ChaseIQ - live IPL chase parsing
========================================
Turns a CricketData.org match object into an IPL second-innings *situation* the model can score.
Pure functions: all fetching, budgeting and caching lives in ``src.cricket.service`` so a poll costs
one API hit instead of three.

The win-probability model only understands IPL second innings, so only live IPL chases are returned.
"""


import re
from datetime import datetime, timezone

from src.cricket.classify import is_ipl
from src.reference import TEAMS, team_id


def overs_to_balls(overs: float) -> int:
    """15.3 overs -> 93 balls (the decimal part is balls, not tenths)."""
    whole = int(overs)
    partial = min(6, round((overs - whole) * 10))
    return min(120, whole * 6 + partial)


def _is_live(match: dict) -> bool:
    return bool(match.get("matchStarted")) and not match.get("matchEnded")


def _is_ipl(match: dict) -> bool:
    teams = list(match.get("teams") or [t.get("name") for t in (match.get("teamInfo") or [])])
    return is_ipl(str(match.get("series", "")), str(match.get("name", "")), teams)


def _is_t20(match: dict) -> bool:
    return (match.get("matchType") or "").lower() in ("t20", "t20i")


def _team_in_label(label: str) -> int | None:
    """Find which franchise an innings label like 'Mumbai Indians Inning 2' refers to."""
    label = label.lower()
    best = None
    for t in TEAMS:
        for key in (t.name, *t.aliases):
            if key.lower() in label and (best is None or len(key) > best[0]):
                best = (len(key), t.team_id)
    return best[1] if best else None


def parse_match(match: dict) -> dict | None:
    """Extract a chase situation from a CricAPI match object (None if not a live chase)."""
    scores = match.get("score") or []
    if len(scores) < 2:
        return None                                   # 1st innings still in progress

    first, second = scores[0], scores[1]
    for s in scores:                                  # prefer explicit labels when present
        label = (s.get("inning") or "").lower()
        if re.search(r"inning\s*2|2nd", label):
            second = s
        elif re.search(r"inning\s*1|1st", label):
            first = s

    bowling = _team_in_label(first.get("inning") or "")
    batting = _team_in_label(second.get("inning") or "")
    if batting is None or bowling is None or batting == bowling:
        names = [team_id(t.get("name") or t.get("shortname") or "")
                 for t in (match.get("teamInfo") or [])]
        names = [n for n in names if n]
        if len(names) != 2:
            return None
        bowling = bowling or next(n for n in names if n != batting)
        batting = batting or next(n for n in names if n != bowling)

    try:
        first_runs = int(first.get("r") or 0)
        runs = int(second.get("r") or 0)
        wickets = int(second.get("w") or 0)
        balls = overs_to_balls(float(second.get("o") or 0))
    except (TypeError, ValueError):
        return None
    if first_runs <= 0:
        return None

    toss = match.get("tossWinner") or ""
    return {
        "batting_team": batting,
        "bowling_team": bowling,
        "venue": match.get("venue") or "Unknown",
        "toss_winner": team_id(toss),
        "target_score": first_runs + 1,
        "current_score": runs,
        "wickets_fallen": min(10, wickets),
        "balls_bowled": balls,
        "match_name": match.get("name", ""),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


def find_ipl_live_chase(items: list[dict]) -> dict | None:
    """First live IPL match (from a CricketData.org `currentMatches` list) that is in its 2nd innings."""
    for match in items or []:
        if _is_t20(match) and _is_live(match) and _is_ipl(match):
            situation = parse_match(match)
            if situation is not None:
                return situation
    return None
