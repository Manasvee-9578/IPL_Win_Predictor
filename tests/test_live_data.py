"""Live-data parsing: pure functions, no network."""

import pytest

from src import live_data as L
from src.reference import team_id

CHASE = {
    "name": "Mumbai Indians vs Chennai Super Kings, 20th Match",
    "matchType": "t20", "series": "Indian Premier League 2026", "venue": "Wankhede Stadium, Mumbai",
    "tossWinner": "Mumbai Indians", "matchStarted": True, "matchEnded": False,
    "status": "Mumbai Indians need 41 runs in 30 balls",
    "score": [
        {"r": 176, "w": 6, "o": 20, "inning": "Chennai Super Kings Inning 1"},
        {"r": 136, "w": 4, "o": 15.0, "inning": "Mumbai Indians Inning 2"},
    ],
}


@pytest.mark.parametrize("overs,balls", [(0, 0), (0.1, 1), (15.3, 93), (19.5, 119), (20, 120), (7, 42)])
def test_overs_to_balls(overs, balls):
    assert L.overs_to_balls(overs) == balls


def test_parse_chase():
    m = L.parse_match(CHASE)
    assert m["batting_team"] == team_id("MI") and m["bowling_team"] == team_id("CSK")
    assert m["target_score"] == 177 and m["current_score"] == 136
    assert m["wickets_fallen"] == 4 and m["balls_bowled"] == 90
    assert m["toss_winner"] == team_id("MI")


def test_first_innings_in_progress_is_ignored():
    first_only = {**CHASE, "score": CHASE["score"][:1]}
    assert L.parse_match(first_only) is None


def test_finished_match_is_not_live():
    ended = {**CHASE, "matchEnded": True, "status": "Mumbai Indians won by 5 wickets"}
    assert not L._is_live(ended)
    assert L._is_live(CHASE)


def test_ipl_and_t20_filters():
    assert L._is_ipl(CHASE) and L._is_t20(CHASE)
    assert not L._is_ipl({"series": "Big Bash League", "name": "Sydney vs Perth"})
    assert not L._is_t20({"matchType": "odi"})


def test_legacy_names_resolve():
    ctx = {**CHASE, "score": [
        {"r": 160, "w": 8, "o": 20, "inning": "Royal Challengers Bangalore Inning 1"},
        {"r": 50, "w": 2, "o": 6.0, "inning": "Kings XI Punjab Inning 2"}]}
    m = L.parse_match(ctx)
    assert m["bowling_team"] == team_id("RCB") and m["batting_team"] == team_id("PBKS")


def test_find_ipl_live_chase_picks_only_ipl_chases():
    assert L.find_ipl_live_chase([CHASE])["batting_team"] == team_id("MI")
    intl = {**CHASE, "name": "India vs West Indies, 2nd T20I", "series": "West Indies tour of India",
            "teams": ["India", "West Indies"],
            "score": [{"r": 176, "w": 6, "o": 20, "inning": "West Indies Inning 1"},
                      {"r": 100, "w": 2, "o": 10, "inning": "India Inning 2"}]}
    assert L.find_ipl_live_chase([intl]) is None          # the chase model is never applied to internationals
    assert L.find_ipl_live_chase([]) is None
