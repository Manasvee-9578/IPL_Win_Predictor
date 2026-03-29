"""
IPL Win Probability Predictor — Live Data Fetcher
===================================================
Fetches live IPL match data from CricAPI and formats it
into a dict matching the MatchState schema.

Usage (standalone test):
    python -c "from src.live_data import fetch_live_match; print(fetch_live_match())"
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

CRICAPI_BASE = "https://api.cricapi.com/v1"
API_KEY = os.getenv("CRICKET_API_KEY", "")


# ---------------------------------------------------------------------------
# IPL team name normalisation
# ---------------------------------------------------------------------------
# CricAPI sometimes returns short names or slight variants.
# Map them to the exact names used in our training data.
TEAM_ALIASES: dict[str, str] = {
    # Full names (identity)
    "Chennai Super Kings":          "Chennai Super Kings",
    "Delhi Capitals":               "Delhi Capitals",
    "Gujarat Titans":               "Gujarat Titans",
    "Kolkata Knight Riders":        "Kolkata Knight Riders",
    "Lucknow Super Giants":        "Lucknow Super Giants",
    "Mumbai Indians":               "Mumbai Indians",
    "Punjab Kings":                 "Punjab Kings",
    "Rajasthan Royals":             "Rajasthan Royals",
    "Royal Challengers Bengaluru":  "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad":          "Sunrisers Hyderabad",
    # Common short forms
    "CSK": "Chennai Super Kings",
    "DC":  "Delhi Capitals",
    "GT":  "Gujarat Titans",
    "KKR": "Kolkata Knight Riders",
    "LSG": "Lucknow Super Giants",
    "MI":  "Mumbai Indians",
    "PBKS": "Punjab Kings",
    "RR":  "Rajasthan Royals",
    "RCB": "Royal Challengers Bengaluru",
    "SRH": "Sunrisers Hyderabad",
    # Legacy / alternate
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
    "Rising Pune Supergiant":       "Rising Pune Supergiant",
    "Rising Pune Supergiants":      "Rising Pune Supergiant",
    "Pune Warriors":                "Pune Warriors",
    "Deccan Chargers":              "Deccan Chargers",
    "Kochi Tuskers Kerala":         "Kochi Tuskers Kerala",
    "Gujarat Lions":                "Gujarat Lions",
}


def _normalise_team(name: str) -> str:
    """Return the canonical team name, or the original if not found."""
    return TEAM_ALIASES.get(name.strip(), name.strip())


def _overs_to_balls(overs_float: float) -> int:
    """
    Convert overs like 15.3 → 93 balls.
    Handles both float overs (15.5 meaning 15 overs 5 balls) and
    integer overs.
    """
    whole_overs = int(overs_float)
    partial = round((overs_float - whole_overs) * 10)  # e.g. 0.3 → 3
    if partial > 6:
        partial = 6  # safety clamp
    return whole_overs * 6 + partial


# ---------------------------------------------------------------------------
# Helpers — live / IPL / T20 detection
# ---------------------------------------------------------------------------
_LIVE_KEYWORDS = [
    "live", "in progress", "innings break", "batting", "bowling",
    "required", "need", "trail", "lead", "opt to", "chasing",
    "run", "won the toss",
]


def _is_live_match(match: dict) -> bool:
    """
    Flexible live detection — does NOT rely strictly on matchStarted / matchEnded.
    """
    status = (match.get("status") or "").lower()
    started = match.get("matchStarted", False)
    ended = match.get("matchEnded", False)

    if any(kw in status for kw in _LIVE_KEYWORDS):
        return True
    if started and not ended:
        return True
    return False


def _is_ipl_series(series: str) -> bool:
    """Case-insensitive partial match for IPL series variants."""
    s = series.lower()
    return any(tag in s for tag in ["ipl", "indian premier league", "tata ipl"])


def _is_t20(match_type: str) -> bool:
    """Accept any T20 variant."""
    return match_type.lower() in ("t20", "t20i")


# ---------------------------------------------------------------------------
# API call helper — abstracts a single CricAPI endpoint
# ---------------------------------------------------------------------------
def _fetch_matches_from_endpoint(endpoint: str) -> list[dict]:
    """
    Hit a CricAPI v1 endpoint and return the match list (may be empty).
    Works for /currentMatches, /matches, /cricScore.
    """
    url = f"{CRICAPI_BASE}/{endpoint}"
    try:
        resp = requests.get(
            url,
            params={"apikey": API_KEY, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[live_data] ❌ /{endpoint} request failed: {exc}")
        return []

    if data.get("status") != "success":
        print(f"[live_data] ❌ /{endpoint} returned status: {data.get('status')}")
        return []

    matches = data.get("data", [])
    print(f"[live_data] 📡 /{endpoint} → {len(matches)} matches")
    return matches


# ---------------------------------------------------------------------------
# Single-match data extraction
# ---------------------------------------------------------------------------
def _extract_match_data(match: dict) -> dict | None:
    """
    Try to build the output dict from a single match object.
    Returns None if the match doesn't have enough data (e.g. no 2nd innings).
    """
    # --- Extract match info ---
    venue = match.get("venue", "Unknown")

    # Toss
    toss_winner_raw = match.get("tossWinner", "")
    toss_winner = _normalise_team(toss_winner_raw) if toss_winner_raw else "Unknown"

    # Teams
    teams_info = match.get("teamInfo") or []
    team_names = []
    for t in teams_info:
        name = (t.get("name") or t.get("shortname") or "").strip()
        if name:
            team_names.append(_normalise_team(name))

    if len(team_names) < 2:
        raw_teams = match.get("teams") or []
        team_names = [_normalise_team(t) for t in raw_teams if t]

    if len(team_names) < 2:
        print(f"[live_data]  │   ⏭ Skipped: could not identify two teams")
        return None

    # --- Score data ---
    score_list = match.get("score") or []
    if not score_list:
        print(f"[live_data]  │   ⏭ Skipped: no score data available")
        return None

    print(f"[live_data]  │   Score entries ({len(score_list)}):")
    for si, s in enumerate(score_list):
        print(f"[live_data]  │     [{si}] {s.get('inning', '?')} — "
              f"r={s.get('r')}, w={s.get('w')}, o={s.get('o')}")

    # Find innings by label first, then fall back to position
    second_innings = None
    first_innings = None
    for s in score_list:
        inning_str = (s.get("inning") or "").lower()
        if "2nd" in inning_str or "inning 2" in inning_str:
            second_innings = s
        elif "1st" in inning_str or "inning 1" in inning_str:
            first_innings = s

    # Positional fallback
    if first_innings is None and len(score_list) >= 1:
        first_innings = score_list[0]
    if second_innings is None and len(score_list) >= 2:
        second_innings = score_list[1]

    # 2nd innings is mandatory; 1st innings can be missing → default 0
    if second_innings is None:
        print(f"[live_data]  │   ⏭ Skipped: no 2nd innings data found")
        return None

    # --- Parse innings data ---
    first_runs = (first_innings.get("r", 0) if first_innings else 0) or 0
    target_score = first_runs + 1

    current_score = second_innings.get("r", 0) or 0
    wickets_fallen = second_innings.get("w", 0) or 0
    overs_raw = second_innings.get("o", 0) or 0

    try:
        overs_float = float(overs_raw)
    except (ValueError, TypeError):
        overs_float = 0.0

    balls_bowled = _overs_to_balls(overs_float)

    # Determine batting team from the 2nd innings label
    inning_label = second_innings.get("inning") or ""
    batting_team = None
    for tn in team_names:
        if tn.lower() in inning_label.lower():
            batting_team = tn
            break

    if batting_team is None:
        batting_team = team_names[1]

    bowling_team = (
        team_names[0] if batting_team == team_names[1] else team_names[1]
    )

    print(f"[live_data]  └─ 🏏 Match data: {batting_team} vs {bowling_team} | "
          f"target={target_score} current={current_score}/{wickets_fallen} "
          f"({overs_float} ov)")

    return {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "toss_winner": toss_winner,
        "target_score": target_score,
        "current_score": current_score,
        "wickets_fallen": wickets_fallen,
        "balls_bowled": min(balls_bowled, 120),
        "batting_team_form": 0.5,
        "head_to_head_ratio": 0.5,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Search a match list — returns (ipl_result, best_t20_fallback)
# ---------------------------------------------------------------------------
def _search_matches(
    matches: list[dict],
    source_label: str,
) -> tuple[dict | None, dict | None]:
    """
    Scan *matches* for:
      1. A live IPL T20 match with 2nd-innings data  → ipl_result
      2. Any live T20 match with 2nd-innings data     → t20_fallback

    Returns (ipl_result, t20_fallback).  Either may be None.
    """
    ipl_result: dict | None = None
    t20_fallback: dict | None = None

    for idx, match in enumerate(matches):
        match_name = match.get("name", "Unknown")
        match_type = (match.get("matchType") or "").lower()
        series = match.get("series") or ""
        status = match.get("status") or ""

        print(f"[live_data]  ├─ [{source_label}] Match {idx}: {match_name}")
        print(f"[live_data]  │   series={series}  type={match_type}  status={status}")
        print(f"[live_data]  │   matchStarted={match.get('matchStarted')}  "
              f"matchEnded={match.get('matchEnded')}")

        # --- T20 check ---
        if not _is_t20(match_type):
            print(f"[live_data]  │   ⏭ Skipped: matchType '{match_type}' is not T20")
            continue

        # --- Live check ---
        if not _is_live_match(match):
            print(f"[live_data]  │   ⏭ Skipped: match does not appear live")
            continue

        # --- Try to extract data ---
        print(f"[live_data]  │   ✅ Passed live+T20 filters — extracting …")
        result = _extract_match_data(match)
        if result is None:
            continue

        # --- IPL check ---
        if _is_ipl_series(series):
            print(f"[live_data]  │   🏆 IPL match found!")
            ipl_result = result
            return ipl_result, t20_fallback  # IPL takes priority, return immediately

        # Keep the first non-IPL T20 result as fallback
        if t20_fallback is None:
            print(f"[live_data]  │   📋 Stored as T20 fallback (not IPL)")
            t20_fallback = result

    return ipl_result, t20_fallback


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def fetch_live_match() -> dict | None:
    """
    Find a live IPL match with 2nd-innings data and return a dict
    matching the MatchState schema. Uses multiple CricAPI endpoints
    as fallbacks and, as a last resort, returns any live T20 match.

    Endpoint priority:
        1. /currentMatches   (most granular, but sometimes incomplete)
        2. /cricScore         (±7-day window, simplified scores)
        3. /matches           (general match list)

    Returned dict keys:
        batting_team, bowling_team, venue, toss_winner,
        target_score, current_score, wickets_fallen, balls_bowled,
        batting_team_form, head_to_head_ratio, last_updated
    """
    if not API_KEY or API_KEY == "your_key_here":
        print("[live_data] ⚠ No valid API key configured.")
        return None

    # Endpoints to try, in priority order
    endpoints = ["currentMatches", "cricScore", "matches"]

    best_t20_fallback: dict | None = None

    for ep in endpoints:
        print(f"\n[live_data] ── Trying /{ep} ──")
        matches = _fetch_matches_from_endpoint(ep)
        if not matches:
            continue

        ipl_result, t20_fb = _search_matches(matches, ep)

        if ipl_result is not None:
            print(f"[live_data] ✅ Returning IPL match from /{ep}")
            return ipl_result

        # Keep the best T20 fallback across endpoints
        if t20_fb is not None and best_t20_fallback is None:
            best_t20_fallback = t20_fb

    # --- Fallback: return any live T20 match ---
    if best_t20_fallback is not None:
        print("[live_data] ⚠ No IPL match found — returning best live T20 fallback")
        return best_t20_fallback

    print("[live_data] ℹ No live match found across all endpoints.")
    return None
