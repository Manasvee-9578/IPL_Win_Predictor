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
def _is_live_match(match: dict) -> bool:
    """
    Determine if a match is currently live using flexible detection.
    Does NOT rely strictly on matchStarted / matchEnded because CricAPI
    sometimes reports inconsistent values for these fields.
    """
    status = (match.get("status") or "").lower()
    started = match.get("matchStarted", False)
    ended = match.get("matchEnded", False)

    # Primary: check status text for live-indicating keywords
    live_keywords = ["live", "in progress", "innings break", "batting", "bowling",
                     "required", "need", "trail", "lead", "opt to"]
    if any(kw in status for kw in live_keywords):
        return True

    # Fallback: trust matchStarted when matchEnded is explicitly False/absent
    if started and not ended:
        return True

    return False


def _is_ipl_series(series: str) -> bool:
    """Case-insensitive partial match for IPL series variants."""
    s = series.lower()
    return any(tag in s for tag in ["ipl", "indian premier league", "tata ipl"])


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
        print("[live_data] ⚠ No valid API key configured.")
        return None

    try:
        resp = requests.get(
            f"{CRICAPI_BASE}/currentMatches",
            params={"apikey": API_KEY, "offset": 0},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[live_data] ❌ API request failed: {exc}")
        return None

    if data.get("status") != "success":
        print(f"[live_data] ❌ API returned non-success status: {data.get('status')}")
        return None

    matches = data.get("data", [])
    print(f"[live_data] 📡 Received {len(matches)} matches from CricAPI")

    for idx, match in enumerate(matches):
        match_name = match.get("name", "Unknown")
        match_type = (match.get("matchType") or "").lower()
        series = match.get("series") or ""
        status = match.get("status") or ""

        print(f"[live_data]  ├─ Match {idx}: {match_name}")
        print(f"[live_data]  │   series={series}  type={match_type}  status={status}")
        print(f"[live_data]  │   matchStarted={match.get('matchStarted')}  "
              f"matchEnded={match.get('matchEnded')}")

        # --- IPL filter (case-insensitive, partial match) ---
        if not _is_ipl_series(series):
            print(f"[live_data]  │   ⏭ Skipped: not an IPL series")
            continue

        # --- T20 filter ---
        is_t20 = match_type in ("t20", "t20i")
        if not is_t20:
            print(f"[live_data]  │   ⏭ Skipped: matchType '{match_type}' is not T20")
            continue

        # --- Live detection (flexible) ---
        if not _is_live_match(match):
            print(f"[live_data]  │   ⏭ Skipped: match does not appear live")
            continue

        print(f"[live_data]  │   ✅ Passed filters — extracting match data …")

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
            # Fallback: try the teams list
            raw_teams = match.get("teams") or []
            team_names = [_normalise_team(t) for t in raw_teams if t]

        if len(team_names) < 2:
            print(f"[live_data]  │   ⏭ Skipped: could not identify two teams")
            continue

        # --- Score data ---
        score_list = match.get("score") or []
        if not score_list:
            print(f"[live_data]  │   ⏭ Skipped: no score data available")
            continue

        print(f"[live_data]  │   Score entries ({len(score_list)}):")
        for si, s in enumerate(score_list):
            print(f"[live_data]  │     [{si}] {s.get('inning','?')} — "
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
            continue

        # --- Parse innings data ---
        # 1st innings → target (default to 0 if missing)
        first_runs = (first_innings.get("r", 0) if first_innings else 0) or 0
        target_score = first_runs + 1  # target = 1st innings total + 1

        # 2nd innings → current state
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
            # Fallback: 2nd team in list is usually chasing
            batting_team = team_names[1]

        bowling_team = (
            team_names[0] if batting_team == team_names[1] else team_names[1]
        )

        print(f"[live_data]  └─ 🏏 Returning: {batting_team} vs {bowling_team} | "
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
            "batting_team_form": 0.5,       # neutral default for live
            "head_to_head_ratio": 0.5,       # neutral default for live
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    # No live IPL match found
    print("[live_data] ℹ No live IPL 2nd-innings match found in current matches.")
    return None
