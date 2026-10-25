"""Simulator: set up any chase, then bowl it ball by ball."""

import json

import streamlit as st

from dashboard import api_client as api
from dashboard import components as ui

DEFAULTS = {"sim_balls": 60, "sim_score": 90, "sim_wkts": 3, "sim_target": 180}
DELIVERIES = [("ball_dot", "•", {"runs": 0}), ("ball_1", "1", {"runs": 1}), ("ball_2", "2", {"runs": 2}),
              ("ball_4", "4", {"runs": 4}), ("ball_6", "6", {"runs": 6}), ("ball_W", "W", {"wicket": True})]


# -- callbacks run *before* the widgets are re-created, so they may edit widget state --
def _deliver(runs: int = 0, wicket: bool = False) -> None:
    s = st.session_state
    if s.sim_balls >= 120 or s.sim_wkts >= 10 or s.sim_score >= s.sim_target:
        return
    s.sim_balls += 1
    s.sim_score = min(500, s.sim_score + runs)
    s.sim_wkts = min(10, s.sim_wkts + int(wicket))
    s.sim_log.append("W" if wicket else str(runs))


def _reset_chase() -> None:
    s = st.session_state
    s.sim_balls, s.sim_score, s.sim_wkts = 0, 0, 0
    s.sim_hist, s.sim_log = {}, []


def _clear_trail() -> None:
    st.session_state.sim_hist = {}


def render() -> None:
    s = st.session_state
    for key, val in DEFAULTS.items():
        s.setdefault(key, val)
    s.setdefault("sim_hist", {})
    s.setdefault("sim_log", [])

    teams_data, err = api.teams()
    venues_data, err2 = api.venues()
    if err or err2:
        ui.empty("API unavailable", err or err2)
        return
    team_names, venue_names = teams_data["teams"], venues_data["venues"]

    hero_slot = st.container()                       # filled once the prediction is known
    ui.section("01", "Set the scene", "A hypothetical IPL chase: choose the fixture, then bowl it ball by ball. "
               "Everything here is a what-if scored by the IPL Second-Innings Chase Model, not a real match.")

    left, right = st.columns([5, 7], gap="large")
    with left:
        with st.container(key="panel_fixture"):
            ui.panel_label("Fixture")
            c1, c2 = st.columns(2)
            bat = c1.selectbox("Chasing", team_names,
                               index=team_names.index("Mumbai Indians") if "Mumbai Indians" in team_names else 0)
            bowl = c2.selectbox("Defending", [t for t in team_names if t != bat], index=0)
            venue = st.selectbox("Venue", venue_names, index=0)
            c3, c4 = st.columns([3, 2])
            toss = c3.selectbox("Toss won by", [bat, bowl])
            target = c4.number_input("Target", 1, 400, key="sim_target")

        sig = (bat, bowl, venue, toss, target)
        if s.get("sim_sig") != sig:                   # new fixture -> fresh probability trail
            s.sim_sig = sig
            s.sim_hist = {}

        with st.container(key="panel_chase"):
            ui.panel_label("Chase situation")
            balls = st.slider("Balls bowled", 0, 120, key="sim_balls")
            c1, c2 = st.columns(2)
            score = c1.number_input("Score", 0, 500, key="sim_score")
            wickets = c2.number_input("Wickets lost", 0, 10, key="sim_wkts")

            ui.panel_label("Bowl the next ball")
            with st.container(key="pad"):
                cols = st.columns(6)
                for col, (key, label, kw) in zip(cols, DELIVERIES):
                    col.button(label, key=key, on_click=_deliver, kwargs=kw)
            ui.tracker(s.sim_log)
            b1, b2 = st.columns(2)
            b1.button("Restart chase", on_click=_reset_chase, use_container_width=True)
            b2.button("Clear trail", on_click=_clear_trail, use_container_width=True)

            with st.expander("Override recent form / head-to-head"):
                use_form = st.checkbox("Override team form")
                form = st.slider("Batting team form (win ratio, last 5)", 0.0, 1.0, 0.5, 0.05,
                                 disabled=not use_form)
                use_h2h = st.checkbox("Override head-to-head")
                h2h = st.slider("Head-to-head win ratio", 0.0, 1.0, 0.5, 0.05, disabled=not use_h2h)

    payload = {
        "batting_team": bat, "bowling_team": bowl, "venue": venue, "toss_winner": toss,
        "target_score": int(target), "current_score": int(score),
        "wickets_fallen": int(wickets), "balls_bowled": int(balls),
        "batting_team_form": form if use_form else None,
        "head_to_head_ratio": h2h if use_h2h else None,
    }
    result, error = api.predict(json.dumps(payload, sort_keys=True))
    if error:
        with hero_slot:
            ui.empty("Cannot predict this situation", error)
        return

    win, status = result["win_probability"], result["match_status"]
    s.sim_hist[int(balls)] = {"balls": int(balls), "runs": int(score), "wickets": int(wickets), "win_prob": win}
    badge = {"chase_complete": ui.tag("Target chased", "good"), "chase_failed": ui.tag("Chase failed", "bad"),
             "tied": ui.tag("Scores level", "warn")}.get(status, "")

    with hero_slot:
        ui.hero(bat, bowl, win, int(score), int(wickets), int(balls), int(target),
                result["match_state_summary"], badge, mode="hypothetical")
        ui.situation_strip(int(score), int(wickets), int(balls), int(target))

    with right:
        ui.chart_head("Win probability, ball by ball", "Your chase so far. Blue = chasing side ahead, white = "
                      "defending side ahead.")
        ui.trend_chart(sorted(s.sim_hist.values(), key=lambda p: p["balls"]), bat, bowl, marker_ball=int(balls))
        ctx, meta = result["context"], ui.team_meta()
        ab = lambda n: meta.get(n, {}).get("abbr", n[:3].upper())  # noqa: E731
        ui.facts([(f"Elo · {ab(bat)}", f"{ctx['elo_batting']:.0f}"),
                  (f"Elo · {ab(bowl)}", f"{ctx['elo_bowling']:.0f}"),
                  ("Recent form", f"{ctx['batting_team_form']:.2f}"),
                  ("Head-to-head", f"{ctx['head_to_head_ratio']:.2f}"),
                  ("Ground avg 1st inns", f"{ctx['venue_avg_score']:.0f}")])

    if status != "in_progress":
        st.info("The chase is decided - restart it or move the sliders back to explore more scenarios.")
        return

    with right:
        base = result["explanation"]["baseline_win_probability"] * 100
        ui.chart_head("Why this probability?", f"Effect of each group of factors, compared with an average chase "
                      f"({base:.0f}%).")
        ui.factor_chart(result["explanation"])

    ui.section("02", "What happens next", "How the next over could swing it, and where this chase sits on the "
               f"pressure map ({int(wickets)} wickets down - the ring marks the current position).")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        w, werr = api.whatif(json.dumps(payload, sort_keys=True))
        if werr:
            st.warning(werr)
        else:
            ui.chart_head(f"If the next {w['balls_ahead']} balls go...", "Win probability after a maiden, a par "
                          "over, a big over or a wicket.")
            ui.whatif_chart(w)
    with c2:
        flat = {**payload, "current_score": 0, "balls_bowled": 0, "batting_team_form": None,
                "head_to_head_ratio": None}
        hm, herr = api.heatmap(json.dumps(flat, sort_keys=True), int(wickets))
        ui.chart_head("Pressure map", "Win probability for every mix of runs needed and overs bowled.")
        if not herr:
            ui.heatmap_chart(hm, int(target - score), int(balls))
