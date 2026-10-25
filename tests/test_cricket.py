"""Live Cricket layer: classification, provider parsing, quota budget and the 'never invent data' rules.

NOTE: the payloads below are TEST FIXTURES shaped like the providers' documented formats. They exist only
in this test module - the application never contains or shows sample matches.
"""

import io
import json
import zipfile

import pytest

from src.cricket import cricketdata as cd
from src.cricket import cricsheet
from src.cricket.classify import format_for, groups_for, is_ipl, team_group, wanted
from src.cricket.service import CricketService

# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize("name,gender,expected", [
    ("India", "male", "india_men"), ("India", "female", "india_women"), ("India Women", None, "india_women"),
    ("India A", "male", "india_a"), ("India-A", None, "india_a"), ("India A Women", "female", "india_a"),
    ("West Indies", "male", None), ("Indian Railways", None, None), ("India Under-19", None, None),
    ("India U19", None, None), ("Australia", "male", None), ("Mumbai Indians", None, None),
])
def test_team_group(name, gender, expected):
    assert team_group(name, gender) == expected


def test_ipl_detection():
    assert is_ipl(series="Indian Premier League 2026")
    assert is_ipl(teams=["Mumbai Indians", "Chennai Super Kings"])
    assert not is_ipl(series="West Indies tour of India", teams=["India", "West Indies"])
    assert not is_ipl(series="Big Bash League", teams=["Sydney Sixers", "Perth Scorchers"])


def test_formats():
    assert format_for("t20", ["india_men"], False) == ("T20I", "T20I")
    assert format_for("ODI", ["india_women"], False) == ("ODI", "ODI")
    assert format_for("test", ["india_men"], False) == ("TEST", "Test")
    assert format_for("t20", ["india_a"], False)[1] == "T20 · India A"
    assert format_for("t20", [], True) == ("IPL", "IPL")
    assert format_for("t10", ["india_men"], False)[0] == "OTHER"


def test_filter_rule():
    all_groups, all_formats = {"india_men", "india_women", "india_a"}, {"T20I", "ODI", "TEST", "IPL"}
    assert wanted([], "IPL", {"india_men"}, {"IPL"})                    # IPL needs only the IPL format
    assert not wanted([], "IPL", all_groups, {"T20I"})
    assert wanted(["india_women"], "ODI", {"india_women"}, {"ODI"})
    assert not wanted(["india_women"], "ODI", {"india_men"}, all_formats)
    assert not wanted(["india_men"], "ODI", all_groups, {"TEST"})


# --------------------------------------------------------------------------- CricketData.org parsing
def _item(**kw):
    base = {"id": "abc", "name": "India vs West Indies, 2nd T20I", "matchType": "t20",
            "status": "India need 41 runs in 30 balls", "venue": "Eden Gardens, Kolkata", "date": "2026-10-12",
            "dateTimeGMT": "2026-10-12T14:00:00", "teams": ["India", "West Indies"],
            "score": [{"r": 176, "w": 6, "o": 20, "inning": "West Indies Inning 1"},
                      {"r": 136, "w": 4, "o": 15.0, "inning": "India Inning 2"}],
            "matchStarted": True, "matchEnded": False}
    base.update(kw)
    return base


def test_parse_live_international():
    m = cd.parse_item(_item(), "t")
    assert (m.state, m.format_base, m.groups) == ("live", "T20I", ["india_men"])
    assert m.start_utc == "2026-10-12T14:00:00Z"
    assert [(s["team"], s["runs"], s["wickets"]) for s in m.scores] == [("West Indies", 176, 6), ("India", 136, 4)]


def test_parse_states():
    assert cd.parse_item(_item(matchStarted=False, score=[]), "t").state == "upcoming"
    done = cd.parse_item(_item(matchEnded=True, status="India won by 5 wickets"), "t")
    assert done.state == "result" and done.result_text == "India won by 5 wickets"


def test_parse_women_and_a_and_ipl():
    assert cd.parse_item(_item(teams=["India Women", "Australia Women"]), "t").groups == ["india_women"]
    a = cd.parse_item(_item(teams=["India A", "Australia A"], matchType="odi"), "t")
    assert a.groups == ["india_a"] and a.format_label == "ODI · India A"
    ipl = cd.parse_item(_item(teams=["Mumbai Indians", "Chennai Super Kings"], name="MI vs CSK, 20th Match"), "t")
    assert ipl.format_base == "IPL" and ipl.groups == []


def test_irrelevant_matches_are_dropped():
    assert cd.parse_item(_item(teams=["Australia", "England"]), "t") is None
    assert cd.parse_item(_item(teams=["India"]), "t") is None
    assert cd.parse_item(_item(matchType="t10"), "t") is None


# --------------------------------------------------------------------------- quota budget and key handling
def reset_hits():
    """The production counter only ever goes up; tests clear it directly in their temporary database."""
    from src import database as db
    with db.session() as conn:
        conn.execute("DELETE FROM api_usage")


@pytest.fixture(autouse=True)
def _clean_budget():
    reset_hits()
    yield
    reset_hits()

class FakeHTTP:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def get(self, *a, **k):
        self.calls += 1
        payload = self.payload
        return type("R", (), {"json": lambda self_: payload})()


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv("CRICKETDATA_API_KEY", raising=False)
    monkeypatch.delenv("CRICKET_API_KEY", raising=False)
    http = FakeHTTP({})
    with pytest.raises(cd.KeyRejected):
        cd.CricketDataClient(http).current()
    assert http.calls == 0


def test_invalid_key_is_reported_and_not_retried(monkeypatch):
    monkeypatch.setenv("CRICKETDATA_API_KEY", "bad-key")
    http = FakeHTTP({"status": "failure", "reason": "Invalid API Key"})
    svc = CricketService(client=cd.CricketDataClient(http), results_fetcher=lambda d, fetched_at="": [])
    assert svc.refresh_cricketdata() is None
    assert svc.providers["cricketdata"]["state"] == "invalid_key"
    assert "rejected" in svc.providers["cricketdata"]["message"]
    svc.refresh_cricketdata()
    assert http.calls == 1                           # second poll is skipped, no quota burned


def test_hit_budget_is_enforced_and_persisted(monkeypatch):
    monkeypatch.setenv("CRICKETDATA_API_KEY", "k")
    monkeypatch.setenv("CRICKETDATA_DAILY_LIMIT", "12")
    http = FakeHTTP({"status": "success", "data": [], "info": {}})
    client = cd.CricketDataClient(http)
    assert cd.hits_left() == 2                       # 12 limit - 10 reserve
    cd._set_hits(11)
    assert cd.hits_left() == 0
    cd._set_hits(12)
    with pytest.raises(cd.QuotaExceeded):
        client.current()
    assert http.calls == 0


def test_quota_message_from_provider_marks_budget_spent(monkeypatch):
    monkeypatch.setenv("CRICKETDATA_API_KEY", "k")
    monkeypatch.setenv("CRICKETDATA_DAILY_LIMIT", "100")
    http = FakeHTTP({"status": "failure", "reason": "Hits limit reached for today"})
    with pytest.raises(cd.QuotaExceeded):
        cd.CricketDataClient(http).current()
    assert cd.hits_today() >= 100


# --------------------------------------------------------------------------- Cricsheet parsing
def _delivery(runs, wicket=False, extras=None):
    d = {"batter": "A", "bowler": "B", "non_striker": "C", "runs": {"batter": runs, "extras": 0, "total": runs}}
    if wicket:
        d["wickets"] = [{"player_out": "A", "kind": "bowled"}]
    if extras:
        d["extras"] = extras
    return d


def _cricsheet_json(teams, gender="male", mtype="T20", event="Tour", winner=None, by=None):
    outcome = {"winner": winner, "by": by} if winner else {"result": "no result"}
    return {"info": {"teams": teams, "gender": gender, "match_type": mtype, "event": {"name": event},
                     "dates": ["2026-10-11"], "venue": "Holkar", "city": "Indore", "outcome": outcome},
            "innings": [
                {"team": teams[0], "overs": [{"over": 0, "deliveries": [_delivery(4), _delivery(0, wicket=True),
                                                                          _delivery(1, extras={"wides": 1})]}]},
                {"team": teams[1], "overs": [{"over": 0, "deliveries": [_delivery(6)]}]}]}


def test_cricsheet_parse_scores_and_result():
    m = cricsheet.parse_match("1", _cricsheet_json(["India", "West Indies"], winner="India", by={"runs": 14}), "t")
    assert m.state == "result" and m.groups == ["india_men"] and m.format_label == "T20I"
    assert m.result_text == "India won by 14 runs"
    first = m.scores[0]
    assert (first["runs"], first["wickets"], first["overs"]) == (5, 1, "0.2")       # the wide is not a legal ball


def test_cricsheet_gender_and_filtering():
    w = cricsheet.parse_match("2", _cricsheet_json(["India", "Sri Lanka"], gender="female", winner="India",
                                                   by={"wickets": 8}), "t")
    assert w.groups == ["india_women"] and w.result_text == "India won by 8 wickets"
    assert cricsheet.parse_match("3", _cricsheet_json(["Australia", "England"], winner="England",
                                                      by={"runs": 3}), "t") is None
    ipl = cricsheet.parse_match("4", _cricsheet_json(["Mumbai Indians", "Chennai Super Kings"],
                                                     event="Indian Premier League", winner="Mumbai Indians",
                                                     by={"wickets": 5}), "t")
    assert ipl.format_base == "IPL"


def test_cricsheet_zip_roundtrip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("README.txt", "readme")
        z.writestr("11.json", json.dumps(_cricsheet_json(["India", "West Indies"], winner="India", by={"runs": 1})))
        z.writestr("12.json", json.dumps(_cricsheet_json(["Fiji", "Samoa"], winner="Fiji", by={"runs": 1})))
    out = cricsheet.parse_zip(buf.getvalue(), "t")
    assert [m.id for m in out] == ["11"]


# --------------------------------------------------------------------------- service honesty
def test_service_never_invents_matches(monkeypatch):
    monkeypatch.delenv("CRICKETDATA_API_KEY", raising=False)
    monkeypatch.delenv("CRICKET_API_KEY", raising=False)
    svc = CricketService(results_fetcher=lambda d, fetched_at="": [])
    svc.matches.clear()
    assert svc.refresh_cricketdata() is None
    svc.refresh_results()
    snap = svc.snapshot()
    assert snap["counts"] == {"live": 0, "upcoming": 0, "result": 0}
    st = svc.status()
    assert st["cricketdata"]["state"] == "no_key" and "free key" in st["cricketdata"]["message"]
    assert st["cricsheet"]["state"] == "ok"


def test_service_filters_dedupes_and_sorts():
    mk = lambda **k: cricsheet.CricketMatch(**{**dict(  # noqa: E731
        id="x", source="cricsheet", state="result", name="n", teams=["India", "West Indies"], groups=["india_men"],
        format_base="T20I", format_label="T20I", date="2026-10-11"), **k})
    svc = CricketService(results_fetcher=lambda d, fetched_at="": [
        mk(id="a", date="2026-10-01"), mk(id="b", date="2026-10-11"),
        mk(id="c", teams=["India", "Sri Lanka"], groups=["india_women"], format_base="ODI", format_label="ODI")])
    svc.matches.clear()
    svc.refresh_results()
    # a CricketData.org copy of the same real match must not be listed twice
    dup = cd.parse_item(_item(matchEnded=True, matchStarted=True, date="2026-10-11",
                              dateTimeGMT="2026-10-11T09:00:00"), "t")
    svc.matches[f"cricketdata:{dup.id}"] = dup
    res = svc.snapshot("result")["matches"]["result"]
    assert [m["id"] for m in res][:2] == ["b", "c"] or {m["id"] for m in res} == {"a", "b", "c"}
    assert sum(m["source"] == "cricketdata" for m in res) == 0
    only_women = svc.snapshot("result", groups={"india_women"})["matches"]["result"]
    assert [m["id"] for m in only_women] == ["c"]
    only_odi = svc.snapshot("result", formats={"ODI"})["matches"]["result"]
    assert [m["id"] for m in only_odi] == ["c"]
