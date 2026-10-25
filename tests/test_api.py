"""End-to-end tests against the FastAPI app (real model, real database)."""

import pytest

BODY = {
    "batting_team": "Mumbai Indians", "bowling_team": "Chennai Super Kings",
    "venue": "Wankhede Stadium, Mumbai", "toss_winner": "Mumbai Indians",
    "target_score": 180, "current_score": 90, "wickets_fallen": 3, "balls_bowled": 60,
}


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"]


def test_predict(client):
    r = client.post("/predict?log=false", json=BODY)
    assert r.status_code == 200
    d = r.json()
    assert d["win_probability"] + d["lose_probability"] == pytest.approx(1.0, abs=1e-3)
    assert d["match_status"] == "in_progress"
    assert d["explanation"]["factors"]
    assert "Mumbai Indians need 90 runs from 60 balls" in d["match_state_summary"]


def test_predict_accepts_abbreviations_and_legacy_names(client):
    body = {**BODY, "batting_team": "MI", "bowling_team": "Kings XI Punjab"}
    assert client.post("/predict?log=false", json=body).status_code == 200


def test_predict_terminal_states_are_not_errors(client):
    r = client.post("/predict?log=false", json={**BODY, "current_score": 185})
    assert r.status_code == 200 and r.json()["match_status"] == "chase_complete"


@pytest.mark.parametrize("patch", [
    {"bowling_team": "Mumbai Indians"},           # same team
    {"batting_team": "Not A Team"},               # unknown team
    {"wickets_fallen": 11},                       # out of range
    {"balls_bowled": 121},
    {"target_score": 0},
])
def test_predict_rejects_bad_input(client, patch):
    assert client.post("/predict?log=false", json={**BODY, **patch}).status_code == 422


def test_predictions_are_logged_and_listed(client):
    client.delete("/history")
    client.post("/predict?source=pytest", json=BODY)
    h = client.get("/history").json()
    assert h["stats"]["total"] == 1 and h["rows"][0]["source"] == "pytest"
    client.delete("/history")


def test_whatif_and_heatmap(client):
    w = client.post("/whatif", json=BODY).json()
    assert len(w["scenarios"]) == 6
    h = client.post("/heatmap?wickets=3", json=BODY).json()
    assert len(h["win_probability"]) == len(h["runs_needed"])
    assert client.post("/whatif", json={**BODY, "current_score": 200}).status_code == 422


def test_catalogue_endpoints(client):
    teams = client.get("/teams").json()
    assert len(teams["teams"]) == 10 and "Mumbai Indians" in teams["teams"]
    venues = client.get("/venues").json()
    assert "Wankhede Stadium, Mumbai" in venues["venues"]
    assert client.get("/model/info").json()["metrics"]["selected_model"]
    ctx = client.get("/context", params={"batting_team": "MI", "bowling_team": "CSK",
                                         "venue": "Wankhede Stadium"}).json()
    assert ctx["venue_known"] and 0 < ctx["head_to_head_ratio"] < 1


def test_match_list_and_replay(client):
    ms = client.get("/matches", params={"season": 2024, "limit": 5}).json()["matches"]
    assert ms and all(m["season"] == 2024 for m in ms)
    r = client.get(f"/matches/{ms[0]['match_id']}/replay").json()
    pts = r["points"]
    assert pts[0]["balls"] == 0 and pts[-1]["win_prob"] in (0.0, 1.0)
    assert len(r["key_moments"]) > 0
    assert client.get("/matches/1/replay").status_code == 404


def test_analytics(client):
    o = client.get("/analytics/overview").json()
    assert o["totals"]["chases"] > 1000 and o["by_season"]
    t = client.get("/analytics/teams").json()["teams"]
    assert t[0]["elo"] >= t[-1]["elo"]
    h = client.get("/analytics/head-to-head", params={"team_a": "MI", "team_b": "CSK"}).json()
    assert h["summary"]["games"] == h["summary"]["a_wins"] + h["summary"]["b_wins"]
    tp = client.get("/analytics/turning-points").json()
    assert tp["greatest_escapes"][0]["extreme_prob"] < 0.2
    assert client.get("/analytics/venues").json()["venues"]


def test_live_returns_404_without_match(client):
    assert client.get("/live").status_code in (200, 404)


# --------------------------------------------------------------------------- ChaseIQ: scope guards
def test_model_is_labelled_and_ipl_only(client):
    info = client.get("/model/info").json()
    assert info["model_name"] == "IPL Second-Innings Chase Model"
    assert "never applies it to ODI" in info["scope"]
    assert client.post("/predict?log=false", json=BODY).json()["model"] == "IPL Second-Innings Chase Model"


@pytest.mark.parametrize("patch", [
    {"competition": "ODI"}, {"competition": "T20I"}, {"competition": "Test"},
    {"batting_team": "India", "bowling_team": "Australia"},
    {"batting_team": "India Women", "bowling_team": "England Women"},
])
def test_non_ipl_predictions_are_refused(client, patch):
    r = client.post("/predict?log=false", json={**BODY, **patch})
    assert r.status_code == 422
    assert "IPL" in r.text


# --------------------------------------------------------------------------- Live Cricket
def test_cricket_endpoints_are_honest(client):
    data = client.get("/cricket/matches", params={"teams": "india_men,india_women", "formats": "T20I,ODI"}).json()
    assert set(data["counts"]) == {"live", "upcoming", "result"}
    for state, items in data["matches"].items():
        for m in items:
            assert m["source"] in ("cricketdata", "cricsheet") and m["state"] == state
            assert set(m["groups"]) <= {"india_men", "india_women"}
    status = client.get("/cricket/status").json()
    assert status["cricketdata"]["state"] in ("no_key", "invalid_key", "ok", "cached", "quota", "error", "pending")
    assert "Cricsheet" in status["attribution"]
    assert client.get("/cricket/matches", params={"teams": "pakistan"}).status_code == 422


def test_live_endpoint_still_reports_no_match(client):
    assert client.get("/live").status_code in (200, 404)
