"""Replay: re-watch any historical chase with the model's win probability."""

import streamlit as st

from dashboard import api_client as api
from dashboard import components as ui

TICK = "0.45s"          # one ball per tick while playing


# ---- state transitions (callbacks run before the next script run) ---------------
def _go(pos: int | None = None, play: bool | None = None) -> None:
    s = st.session_state
    if pos is not None:
        s.rp_pos = pos
    if play is not None:
        s.rp_play = play
    s.rp_epoch += 1                     # new slider key -> slider re-reads the new position


def _toggle(total: int) -> None:
    s = st.session_state
    if not s.rp_play and s.rp_pos >= total:
        s.rp_pos = 0
    s.rp_play = not s.rp_play
    s.rp_epoch += 1


def _on_slide(epoch: int) -> None:
    st.session_state.rp_pos = st.session_state[f"rp_slider_{epoch}"]


def _moments_html(moments: list[dict], bat: str, bowl: str) -> None:
    peak = max(abs(m["delta_pct"]) for m in moments) or 1
    rows = []
    for m in moments:
        up = m["delta_pct"] > 0
        width = abs(m["delta_pct"]) / peak * 50
        bar = (f'<i style="left:50%;width:{width:.1f}%;background:#1677D2"></i>' if up
               else f'<i style="right:50%;width:{width:.1f}%;background:#FFFFFF"></i>')
        ev = "Wicket" if m["wickets"] else ("Boundary" if m["runs"] >= 4 else f"{m['runs']} run(s)")
        rows.append(
            f'<div class="moment"><div class="moment-over">{m["over"]}<small>OVER</small></div>'
            f'<div class="moment-ev"><b style="color:#FFFFFF">{ev}</b><br>{m["score"]}</div>'
            f'<div class="moment-bar">{bar}</div>'
            f'<div class="moment-delta {"up" if up else "down"}">{m["delta_pct"]:+.1f}</div></div>')
    ui.html(f'<div class="moments">{"".join(rows)}</div>')


def render() -> None:
    ui.section("01", "Choose a match", "Pick any IPL chase since 2008 and watch the model's win probability move "
               "ball by ball.")
    seasons, err = api.seasons()
    teams, err2 = api.teams()
    if err or err2:
        ui.empty("API unavailable", err or err2)
        return

    with st.container(key="panel_pick"):
        c1, c2, c3 = st.columns([1, 1.6, 4])
        season = c1.selectbox("Season", ["All"] + seasons["seasons"], index=1)
        team = c2.selectbox("Team", ["Any"] + teams["teams"])
        data, err = api.matches(None if season == "All" else season, None if team == "Any" else team)
        if err or not data["matches"]:
            c3.warning(err or "No matches for this filter")
            return
        labels = {m["match_id"]: f"{m['match_date'][:10]}  ·  {m['team1']} vs {m['team2']}  ·  {m['result_text']}"
                  for m in data["matches"]}
        match_id = c3.selectbox("Match", list(labels), format_func=labels.get)

    rep, err = api.replay(match_id)
    if err:
        st.error(err)
        return
    meta, pts = rep["match"], rep["points"]
    total = pts[-1]["balls"]
    bat, bowl = meta["chasing_team"], meta["first_batting_team"]

    s = st.session_state
    s.setdefault("rp_epoch", 0)
    if s.get("rp_match") != match_id:                # new match -> show the finished chase
        s.rp_match, s.rp_pos, s.rp_play = match_id, total, False
        s.rp_epoch += 1

    ui.section("02", "The chase", None)
    with st.container(key="transport"):
        b1, b2, b3, b4 = st.columns([1.2, 1, 1, 5])
        b1.button("Pause" if s.rp_play else "Play", key="play", on_click=_toggle, args=(total,),
                  use_container_width=True)
        b2.button("Start", on_click=_go, kwargs={"pos": 0, "play": False}, use_container_width=True)
        b3.button("End", on_click=_go, kwargs={"pos": total, "play": False}, use_container_width=True)
        with b4:
            ui.html('<div style="padding-top:8px">' + (
                ui.tag("Hold-out match · unseen by the model", "good") if meta["split"] == "test"
                else ui.tag("Training-period match · in-sample", "warn")) + "</div>")
    if not s.rp_play:
        st.slider("Ball", 0, total, value=s.rp_pos, key=f"rp_slider_{s.rp_epoch}",
                  on_change=_on_slide, args=(s.rp_epoch,))

    def panel() -> None:
        if s.rp_play:
            s.rp_pos = min(total, s.rp_pos + 1)
            if s.rp_pos >= total:                    # finished: drop the timer via a full rerun
                s.rp_play = False
                s.rp_epoch += 1
                st.rerun()
        ball = s.rp_pos
        upto = [p for p in pts if p["balls"] <= ball]
        cur = upto[-1]
        summary = (f"<b>{bat}</b> chasing {rep['target']} against {bowl} at {meta['venue']}" if ball < total
                   else f"<b>{meta['result_text']}</b> · {bat} {cur['runs']}/{cur['wickets']} chasing {rep['target']}")
        ui.hero(bat, bowl, cur["win_prob"], cur["runs"], cur["wickets"], cur["balls"], rep["target"], summary,
                ui.tag(f"Season {meta['season']}"), mode="historical")
        ui.situation_strip(cur["runs"], cur["wickets"], cur["balls"], rep["target"])
        shown = [m for m in rep["key_moments"] if m["balls"] <= ball]
        ui.chart_head("Win probability, ball by ball", "Diamonds mark the deliveries that swung the match "
                      "(diamond = wicket, circle = boundary).")
        ui.trend_chart(upto, bat, bowl, marker_ball=ball, moments=shown)

    # a fragment re-runs only itself on each tick, which keeps the animation smooth
    st.fragment(panel, run_every=TICK if s.rp_play else None)()

    ui.section("03", "Turning points", "The six deliveries that moved the win probability most. "
               "Blue favoured the chasing side, white the defence.")
    _moments_html(rep["key_moments"], bat, bowl)
    st.caption(rep["note"])
