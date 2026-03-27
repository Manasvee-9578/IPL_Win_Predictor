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
# Main entry point
# ---------------------------------------------------------------------------
def fetch_live_match() -> dict | None:
    """
    Call CricAPI /currentMatches and return a dict for the first
    live IPL 2nd-innings match found, or None.

    Returned dict keys match MatchState fields:
        batting_team, bowling_team, venue, toss_winner,
        target_score, current_score, wickets_fallen, balls_bowled,
        batting_team_form, head_to_head_ratio
    """
    if not API_KEY or API_KEY == "your_key_here":
        return None

    try:
        resp = requests.get(
            f"{CRICAPI_BASE}/currentMatches",
            params={"apikey": API_KEY, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if data.get("status") != "success":
        return None

    matches = data.get("data", [])

    for match in matches:
        # Filter for IPL T20 matches that are currently live
        match_type = (match.get("matchType") or "").lower()
        series = (match.get("series") or "").lower()
        started = match.get("matchStarted", False)
        ended = match.get("matchEnded", False)

        is_ipl = "ipl" in series or "indian premier league" in series
        is_t20 = match_type in ("t20", "t20i")

        if not (is_ipl and is_t20 and started and not ended):
            continue

        # --- Extract match info ---
        venue = match.get("venue", "Unknown")

        # Toss
        toss_winner_raw = match.get("tossWinner", "")
        toss_winner = _normalise_team(toss_winner_raw)

        # Teams
        teams_info = match.get("teamInfo", [])
        team_names = []
        for t in teams_info:
            name = t.get("name", "") or t.get("shortname", "")
            team_names.append(_normalise_team(name))

        if len(team_names) < 2:
            # Fallback: try the teams list
            raw_teams = match.get("teams", [])
            team_names = [_normalise_team(t) for t in raw_teams]

        if len(team_names) < 2:
            continue

        # --- Score data ---
        # CricAPI returns a "score" array with entries per innings
        score_list = match.get("score", [])
        if not score_list:
            continue

        # Find 2nd innings entry
        second_innings = None
        first_innings = None
        for s in score_list:
            inning_str = (s.get("inning", "") or "").lower()
            if "2nd" in inning_str or "inning 2" in inning_str:
                second_innings = s
            elif "1st" in inning_str or "inning 1" in inning_str:
                first_innings = s

        # If we found no labelled innings, try by position
        if first_innings is None and len(score_list) >= 1:
            first_innings = score_list[0]
        if second_innings is None and len(score_list) >= 2:
            second_innings = score_list[1]

        if second_innings is None or first_innings is None:
            continue

        # --- Parse innings data ---
        # 1st innings → target
        first_runs = first_innings.get("r", 0)
        target_score = first_runs + 1  # target = 1st innings total + 1

        # 2nd innings → current state
        current_score = second_innings.get("r", 0)
        wickets_fallen = second_innings.get("w", 0)
        overs_raw = second_innings.get("o", 0)

        try:
            overs_float = float(overs_raw)
        except (ValueError, TypeError):
            overs_float = 0.0

        balls_bowled = _overs_to_balls(overs_float)

        # Determine batting team from the 2nd innings label
        inning_label = second_innings.get("inning", "")
        batting_team = None
        for tn in team_names:
            if tn.lower() in inning_label.lower():
                batting_team = tn
                break

        if batting_team is None:
            # Fallback: 2nd team in list is usually chasing
            batting_team = team_names[1]

        bowling_team = (
            team_names[0] if batting_team == team_names[1] else team_names[1]
        )

        return {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "toss_winner": toss_winner,
            "target_score": target_score,
            "current_score": current_score,
            "wickets_fallen": wickets_fallen,
            "balls_bowled": min(balls_bowled, 120),
            "batting_team_form": 0.5,       # neutral default for live
            "head_to_head_ratio": 0.5,       # neutral default for live
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    # No live IPL match found
    return None
