"""Analytics: seasons, teams, venues, head-to-head and turning points (all SQL-backed)."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import components as ui
from dashboard.components import BLUE, BLUE_HI, FONT_NUM, INK, INK2, INK3, WHITE, chart_layout, show


def render() -> None:
    overview, err = api.analytics("overview")
    if err:
        ui.empty("Analytics unavailable", err)
        return
    t = overview["totals"]
    ui.section("01", "The IPL in numbers", "Everything on this page is a SQL query over the match database.")
    ui.strip([("Matches", f"{t['matches']:,}", "", "2008 - 2025"), ("Analysed chases", f"{t['chases']:,}", "hi",
              "clean 2nd innings"), ("Seasons", str(t["seasons"]), "", f"{t['venues']} grounds"),
              ("Ball-by-ball states", f"{t['ball_states']:,}", "", "scored by the model")])

    def _keep_sub() -> None:
        s = st.session_state
        if s.analytics_sub is None:
            s.analytics_sub = s.get("_sub_last", "Seasons")
        s._sub_last = s.analytics_sub

    st.session_state.setdefault("analytics_sub", "Seasons")
    with st.container(key="subnav"):
        st.segmented_control("Analysis", ["Seasons", "Teams", "Venues", "Head-to-head", "Turning points"],
                             key="analytics_sub", label_visibility="collapsed", on_change=_keep_sub)
    sub = st.session_state.analytics_sub or st.session_state.get("_sub_last", "Seasons")
    st.session_state._sub_last = sub
    {"Seasons": lambda: _seasons(overview), "Teams": _teams, "Venues": _venues,
     "Head-to-head": _head_to_head, "Turning points": _turning_points}[sub]()


def _seasons(ov: dict) -> None:
    df = pd.DataFrame(ov["by_season"])
    ui.chart_head("Chasing sides win about half the time", "Blue line: share of matches won by the chasing "
                  "team. Grey bars: average first-innings score - scoring has climbed sharply since 2023.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["season"], y=df["avg_first_innings"], name="Avg 1st-innings score", yaxis="y2",
                         marker=dict(color="rgba(255,255,255,.16)"), hovertemplate="%{x}: %{y} avg score<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["season"], y=df["chase_win_pct"], name="Chasing side wins %", mode="lines+markers",
                             line=dict(color=BLUE, width=3), marker=dict(size=8, color=BLUE, line=dict(color=INK, width=1.5)),
                             hovertemplate="%{x}: chasing side won %{y}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,.3)")
    fig.update_layout(yaxis=dict(range=[20, 80], ticksuffix="%", title_text="Chasing side wins"),
                      yaxis2=dict(overlaying="y", side="right", range=[0, 230], showgrid=False,
                                  title_text="Avg score", tickfont=dict(family=FONT_NUM, size=13, color=INK3)),
                      margin=dict(l=54, r=54, t=14, b=38))
    fig.update_xaxes(dtick=2)
    show(chart_layout(fig, 400, legend=True))
    toss = ov["toss"]
    st.caption(f"Toss winners won {toss['toss_winner_wins_pct']}% of matches - barely better than a coin flip, "
               f"which is why the model gives the toss almost no weight. {toss['chose_to_field_pct']}% of toss "
               f"winners chose to field first.")


def _teams() -> None:
    data, err = api.analytics("teams")
    if err:
        st.error(err)
        return
    df = pd.DataFrame(data["teams"])
    only_active = st.toggle("Current franchises only", value=True)
    if only_active:
        df = df[df["active"] == 1]
    df = df.reset_index(drop=True)
    lo, hi = 1400, max(1600, df["elo"].max() + 10)
    best = {c: df[c].max() for c in ("win_pct", "chase_win_pct", "defend_win_pct")}

    def num(row, col):
        v = row[col]
        if pd.isna(v):
            return '<div class="lb-num">-</div>'
        return f'<div class="lb-num {"best" if v == best[col] else ""}">{v:.0f}%</div>'

    rows = ['<div class="lb-row head"><div>#</div><div></div><div>Team</div><div>Elo strength rating</div>'
            '<div style="text-align:right">Win</div><div style="text-align:right">Chase</div>'
            '<div style="text-align:right">Defend</div></div>']
    for i, r in df.iterrows():
        pct = max(0, min(100, (r["elo"] - lo) / (hi - lo) * 100))
        rows.append(
            f'<div class="lb-row"><div class="lb-rank">{i + 1}</div><div>{ui.crest(r["name"], 40)}</div>'
            f'<div class="lb-name">{r["name"]}<small>{int(r["matches"])} matches</small></div>'
            f'<div><div class="lb-bar"><i style="width:{pct:.0f}%"></i></div>'
            f'<div class="lb-elo">{r["elo"]:.0f}</div></div>'
            f'{num(r, "win_pct")}{num(r, "chase_win_pct")}{num(r, "defend_win_pct")}</div>')
    ui.chart_head("Team strength", "Elo rating after the latest match. Win = all matches, Chase / Defend = win rate "
                  "when batting second / first. Orange = best in column.")
    ui.html(f'<div class="lb">{"".join(rows)}</div>')


def _venues() -> None:
    data, err = api.analytics("venues")
    if err:
        st.error(err)
        return
    df = pd.DataFrame(data["venues"])
    ui.chart_head("Grounds: high-scoring or chase-friendly?", "Bubble size = matches played (grounds with at least 5). "
                  "Right = more runs; above the line = chasing side wins more often.")
    # label only the busier grounds, by city, so the cloud stays legible; the rest label on hover
    city = df["name"].str.split(",").str[-1].str.strip()
    short = [c if n >= 14 else "" for c, n in zip(city, df["matches"])]
    fig = go.Figure(go.Scatter(
        x=df["avg_first_innings"], y=df["chase_win_pct"], mode="markers+text", text=short, textposition="top center",
        textfont=dict(family="Manrope", size=11, color=INK2),
        marker=dict(size=df["matches"].clip(upper=90) / 3 + 10, color="rgba(22,119,210,.35)",
                    line=dict(width=2, color=BLUE_HI)),
        customdata=df[["name", "matches", "highest_total"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} matches · highest total %{customdata[2]}"
                      "<br>avg 1st innings %{x} · chasing side wins %{y}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,.3)")
    fig.update_xaxes(title_text="Average first-innings score")
    fig.update_yaxes(title_text="Chasing side wins", ticksuffix="%")
    show(chart_layout(fig, 500))


def _head_to_head() -> None:
    teams, err = api.teams()
    if err:
        st.error(err)
        return
    names = teams["teams"]
    c1, c2 = st.columns(2)
    a = c1.selectbox("Team A", names, index=names.index("Mumbai Indians") if "Mumbai Indians" in names else 0)
    rest = [n for n in names if n != a]
    b = c2.selectbox("Team B", rest, index=rest.index("Chennai Super Kings") if "Chennai Super Kings" in rest else 0)
    data, err = api.analytics("head-to-head", json.dumps({"team_a": a, "team_b": b}))
    if err:
        st.error(err)
        return
    s = data["summary"]
    aw, bw = int(s["a_wins"] or 0), int(s["b_wins"] or 0)
    if not s["games"]:
        ui.empty("No meetings", "These two franchises never played each other in the database.")
        return
    pct = aw / (aw + bw) * 100 if aw + bw else 50
    ui.html(f"""
    <section class="bug" style="--p:{pct:.1f}%">
      <div class="bug-top">
        <div class="bug-side">{ui.crest(a, 54)}<div><div class="bug-team">{a}</div><div class="bug-role role-chase">Team A</div></div></div>
        <div class="bug-centre"><div class="bug-score">{s['games']}</div><div class="bug-over">matches</div></div>
        <div class="bug-side right">{ui.crest(b, 54)}<div><div class="bug-team">{b}</div><div class="bug-role role-defend">Team B</div></div></div>
      </div>
      <div class="bug-prob"><div class="prob-num chase">{aw}</div><div class="prob-label">Wins</div><div class="prob-num defend">{bw}</div></div>
      <div class="meter"><i class="fill" style="width:{pct:.1f}%"></i><i class="mid"></i></div>
      <div class="bug-foot"><div class="bug-summary">Average first-innings score in this fixture: <b>{s['avg_first_innings']}</b></div></div>
    </section>""")
    ui.section("", "Most recent meetings", None)
    meta = ui.team_meta()
    chips = []
    for r in data["recent"]:
        w = r["winner"]
        chips.append(f'<div class="fact"><span>{r["match_date"][:10]}</span>'
                     f'<b style="color:{BLUE_HI if w == a else WHITE}">{meta.get(w, {}).get("abbr", w)}</b></div>')
    ui.html('<div class="facts">' + "".join(chips) + "</div>")


def _turning_points() -> None:
    data, err = api.analytics("turning-points", json.dumps({"limit": 8}))
    if err:
        st.error(err)
        return
    ui.chart_head("Greatest escapes and biggest collapses", "One SQL aggregate over 125k ball states: the worst "
                  "probability the eventual winner ever had, and the best the eventual loser ever had. "
                  "Model-rated, not official records.")
    c1, c2 = st.columns(2, gap="large")
    for col, key, title, color, verb in [(c1, "greatest_escapes", "Greatest escapes", BLUE_HI, "won from"),
                                         (c2, "biggest_collapses", "Biggest collapses", WHITE, "lost from")]:
        with col:
            ui.panel_label(title)
            for r in data[key]:
                rep, _ = api.replay(r["match_id"])
                curve = [(p["balls"], p["win_prob"]) for p in (rep or {}).get("points", [])]
                chance = r["extreme_prob"] * 100
                ui.html(f"""
                <div class="tp"><div>
                  <div class="tp-title">{r['chasing_team']} <span style="color:#97A9CF;font-weight:500">v {r['opponent']}</span></div>
                  <div class="tp-meta">{r['season']} · chasing {r['target']} · {verb} <span class="tp-pct" style="color:{color}">{chance:.1f}%</span>
                  {'· hold-out' if r['split'] == 'test' else ''}<br>{r['venue']}</div></div>
                  {ui.spark(curve, color)}</div>""")
