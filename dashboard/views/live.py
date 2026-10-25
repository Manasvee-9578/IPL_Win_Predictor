"""Live Cricket: India Men / Women / A and IPL - live scores, upcoming fixtures and recent results.

Everything here comes from real providers (CricketData.org live + fixtures, Cricsheet results).
When a source is missing or unavailable the page says so; it never invents a score or a fixture.
"""

import os

import streamlit as st

from dashboard import api_client as api
from dashboard import components as ui

REFRESH = int(os.getenv("LIVE_REFRESH_SECONDS", "60"))     # re-reads the server cache; no extra API hits
TEAMS = {"India Men": "india_men", "India Women": "india_women", "India A": "india_a"}
FORMATS = {"T20I": "T20I", "ODI": "ODI", "Test": "TEST", "IPL": "IPL"}
TABS = {"Live now": "live", "Upcoming": "upcoming", "Recent results": "result"}


def _ipl_live_panel() -> None:
    """The IPL Second-Innings Chase Model's live view - shown only for a real live IPL chase."""
    data, err = api.live()
    if err:
        return
    live = data["data"]
    stamp = live["last_updated"][:19].replace("T", " ")
    ui.hero(live["batting_team"], live["bowling_team"], live["win_probability"], live["current_score"],
            live["wickets_fallen"], live["balls_bowled"], live["target_score"],
            f"{live['match_state_summary']}<br><span style='color:#97A9CF'>updated {stamp} UTC</span>",
            mode="live")
    ui.situation_strip(live["current_score"], live["wickets_fallen"], live["balls_bowled"], live["target_score"])
    if live.get("explanation"):
        ui.chart_head("Why this probability?", "From the IPL Second-Innings Chase Model.")
        ui.factor_chart(live["explanation"])


def _t(iso: str | None) -> str:
    return iso[:16].replace("T", " ") + " UTC" if iso else "never"


def _provider_notices(status: dict) -> None:
    cd = status["cricketdata"]
    if cd["state"] in ("no_key", "invalid_key", "quota", "error"):
        steps = ""
        if cd["state"] in ("no_key", "invalid_key"):
            steps = ("<ol class='bullets'><li>Create a free account at <b>cricketdata.org</b> (no card; "
                     "<b>100 requests per day</b>).</li><li>Copy the API key from the dashboard.</li>"
                     "<li>Put <code>CRICKETDATA_API_KEY=your-key</code> in <code>.env</code> and restart ChaseIQ.</li></ol>")
        ui.notice("Live scores and upcoming fixtures are unavailable", f"{cd['message']}{steps}"
                  "<div class='notice-n'>ChaseIQ never fills the gap with made-up data. Recent results below "
                  "come from Cricsheet and are unaffected.</div>", "warn")
    elif cd["state"] == "cached":
        ui.notice("Showing earlier data", cd["message"], "info")
    cs = status["cricsheet"]
    if cs["state"] == "error":
        ui.notice("Recent results are temporarily unavailable", cs["message"], "warn")


@st.fragment(run_every=REFRESH)
def _panel() -> None:
    status, err = api.cricket_status()
    if err:
        ui.empty("Cricket data unavailable", err)
        return
    _provider_notices(status)

    with st.container(key="filters"):
        c1, c2 = st.columns([3, 4])
        with c1:
            ui.panel_label("Teams")
            picked_teams = st.pills("Teams", list(TEAMS), selection_mode="multi", default=list(TEAMS),
                                    key="cq_teams", label_visibility="collapsed")
        with c2:
            ui.panel_label("Formats")
            picked_formats = st.pills("Formats", list(FORMATS), selection_mode="multi", default=list(FORMATS),
                                      key="cq_formats", label_visibility="collapsed")
    teams_q = ",".join(TEAMS[t] for t in (picked_teams or [])) or "none"
    formats_q = ",".join(FORMATS[f] for f in (picked_formats or [])) or "none"
    if teams_q == "none" and "IPL" not in (picked_formats or []):
        ui.empty("Nothing selected", "Pick at least one team group or the IPL format.")
        return
    # an empty team list is not valid for the API; use the first group with IPL-only formats
    data, err = api.cricket("all", teams_q if teams_q != "none" else "india_men",
                            formats_q if formats_q != "none" else "IPL")
    if err:
        ui.empty("Cricket data unavailable", err)
        return
    counts = data["counts"]

    if "cq_tab" not in st.session_state:               # first visit: open on the first tab that has content
        st.session_state.cq_tab = ("Live now" if counts["live"] else "Upcoming" if counts["upcoming"]
                                   else "Recent results" if counts["result"] else "Live now")
    with st.container(key="subnav"):
        tab = st.segmented_control("View", list(TABS), key="cq_tab", label_visibility="collapsed")
    tab_key = TABS[tab or "Live now"]
    ui.facts([("Live now", str(counts["live"])), ("Upcoming", str(counts["upcoming"])),
              ("Recent results", str(counts["result"]))])
    items = data["matches"][tab_key]

    if tab_key == "live":
        _ipl_live_panel()
    if not items:
        what = {"live": "No live matches match your filters right now.",
                "upcoming": "No upcoming fixtures are listed for your filters.",
                "result": "No recent results match your filters."}[tab_key]
        extra = ""
        if tab_key in ("live", "upcoming") and status["cricketdata"]["state"] != "ok":
            extra = " Live and fixture data needs the free CricketData.org key described above."
        if tab_key == "result" and status["cricsheet"]["data_through"]:
            extra = (f" Cricsheet's completed-match data runs through {status['cricsheet']['data_through']}; "
                     "India A matches appear only if Cricsheet carries them.")
        ui.empty("Nothing to show", what + extra)
    else:
        cols = st.columns(2, gap="medium")
        for i, m in enumerate(items):
            with cols[i % 2]:
                ui.match_card(m)

    cd, cs = status["cricketdata"], status["cricsheet"]
    ui.html(f"""<div class="srcline">
      <b>Live &amp; fixtures</b> · CricketData.org free tier · {cd['hits_today']}/{cd['hits_limit']} requests used today
      · last update {_t(cd['last_success'])}<br>
      <b>Results</b> · Cricsheet.org (ODC-BY) · {cs['matches']} matches in the last 30 days
      · data through {cs['data_through'] or 'n/a'} · last fetch {_t(cs['last_success'])}</div>""")


def render() -> None:
    ui.section("01", "Live cricket", "India Men, India Women, India A and the IPL across T20I, ODI, Test and IPL "
               "formats. Real provider data only - live scores are a few minutes behind real time.")
    ui.html(f'<div class="bug-tags" style="justify-content:flex-start;margin:-4px 0 12px">{ui.mode_tag("live")}'
            f'{ui.mode_tag("upcoming")}{ui.mode_tag("historical")}</div>')
    ui.notice("About win probabilities",
              f"ChaseIQ's win-probability model is the <b>{ui.MODEL_NAME}</b>. It is used only for live IPL chases. "
              "International (T20I, ODI, Test) and India A matches are shown with scores only - ChaseIQ never "
              "predicts them.", "info")
    _panel()
