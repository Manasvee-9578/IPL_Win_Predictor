"""
Design-system components: HTML building blocks (styled by styles.css) and the
shared Plotly theme.  Colour meaning is fixed across the whole product:

    BLUE    = the chasing side, primary actions         WHITE = the defending side
    SAFFRON = turning point / hypothetical / key figure (used sparingly)
"""

import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api

BLUE, BLUE_HI, SAFFRON, WHITE = "#1677D2", "#4DA3F5", "#FF9933", "#FFFFFF"
NAVY = "#0B1F4D"
INK, INK2, INK3 = "#FFFFFF", "#D3DCEE", "#97A9CF"
LINE, SURFACE = "#2B4A8C", "#102A63"
FONT_UI = "Manrope, system-ui, sans-serif"
FONT_NUM = "Barlow Condensed, Arial Narrow, sans-serif"


# ---------------------------------------------------------------------------
# HTML plumbing
# ---------------------------------------------------------------------------
def html(markup: str) -> None:
    """Render HTML. Every line is stripped and joined: Markdown treats lines indented by
    4+ spaces as a code block, which would print the raw tags instead of the design."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def _esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


BRAND_MARK = """
<svg viewBox="0 0 40 40" aria-hidden="true">
  <circle cx="20" cy="20" r="17.5" fill="#102A63" stroke="#1677D2" stroke-width="2.4"/>
  <path d="M9 10.5c6.5 5.5 6.5 14.5 0 19" fill="none" stroke="#FF9933" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M31 10.5c-6.5 5.5-6.5 14.5 0 19" fill="none" stroke="#FF9933" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M11.2 14l2.2 1M11.9 18l2.6.2M12 22l2.6-.2M11.2 26l2.2-1M28.8 14l-2.2 1M28.1 18l-2.6.2M28 22l-2.6-.2M28.8 26l-2.2-1"
        stroke="#FFFFFF" stroke-width="1.1" stroke-linecap="round"/>
</svg>"""

STUMPS = """
<svg viewBox="0 0 64 64" aria-hidden="true">
  <g stroke="#1677D2" stroke-width="3" stroke-linecap="round">
    <path d="M20 20v34M32 20v34M44 20v34"/>
  </g>
  <path d="M18 14h12M34 14h12" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
  <path d="M8 56h48" stroke="#2B4A8C" stroke-width="3" stroke-linecap="round"/>
</svg>"""


# ---------------------------------------------------------------------------
# Team identity
# ---------------------------------------------------------------------------
def team_meta() -> dict:
    data, _ = api.teams()
    return {t["name"]: t for t in (data or {}).get("details", [])}


def crest(team_name: str, size: int = 52) -> str:
    meta = team_meta().get(team_name, {"abbr": team_name[:3].upper(), "color": "#3A5A9A"})
    c = meta["color"]
    return (f'<span class="crest" style="width:{size}px;height:{size}px;font-size:{int(size * .4)}px;'
            f'background:linear-gradient(150deg,{c},{c}b8)">{meta["abbr"]}</span>')


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
def topbar(health: dict | None, test: dict | None) -> None:
    chips = []
    if health:
        chips.append('<span class="chip"><span class="dot"></span>API online</span>')
        if health.get("live_match_active"):
            chips.append('<span class="chip"><span class="dot live"></span>IPL match live</span>')
        if test:
            chips.append(f'<span class="chip">IPL chase model <b>{test["accuracy"]:.1%}</b> hold-out accuracy</span>')
    else:
        chips.append('<span class="chip"><span class="dot off"></span>API offline</span>')
    html(f"""
    <header class="topbar">
      <div class="brand">{BRAND_MARK}
        <div><div class="brand-name">Chase<em>IQ</em></div>
        <div class="brand-tag">Inside Every Chase.</div></div>
      </div>
      <div class="sys">{''.join(chips)}</div>
    </header>""")


def section(no: str, title: str, sub: str | None = "") -> None:
    html(f'<div class="sec">' + (f'<span class="sec-no">{no}</span>' if no else '') + f'<span class="sec-title">{title}</span>'
         + (f'<div class="sec-sub">{sub}</div>' if sub else "") + "</div>")


def chart_head(title: str, sub: str = "") -> None:
    html(f'<div class="chart-head"><div class="chart-title">{title}</div>'
         + (f'<div class="chart-sub">{sub}</div>' if sub else "") + "</div>")


def panel_label(text: str) -> None:
    html(f'<div class="panel-label">{text}</div>')


def tag(text: str, kind: str = "") -> str:
    return f'<span class="tag tag-{kind}">{text}</span>' if kind else f'<span class="tag">{text}</span>'


MODEL_NAME = "IPL Second-Innings Chase Model"


def mode_tag(kind: str) -> str:
    """Every number says what it is: LIVE data, HISTORICAL record, or a HYPOTHETICAL what-if."""
    return {
        "live": '<span class="tag tag-live"><span class="dot live"></span>&nbsp;Live</span>',
        "historical": '<span class="tag tag-hist">Historical</span>',
        "hypothetical": '<span class="tag tag-hypo">Hypothetical</span>',
        "upcoming": '<span class="tag tag-sched">Scheduled</span>',
    }[kind]


def model_tag() -> str:
    return f'<span class="tag tag-model">{MODEL_NAME}</span>'


def notice(title: str, body: str, kind: str = "info") -> None:
    html(f'<div class="notice notice-{kind}"><div class="notice-t">{title}</div><div class="notice-b">{body}</div></div>')


def empty(title: str, body: str) -> None:
    html(f'<div class="empty">{STUMPS}<div class="empty-t">{title}</div><p>{body}</p></div>')


# ---------------------------------------------------------------------------
# Data blocks
# ---------------------------------------------------------------------------
def hero(bat: str, bowl: str, win_prob: float, score: int, wickets: int, balls: int, target: int,
         summary: str, badge: str = "", mode: str = "hypothetical") -> None:
    """The scorebug: teams, score, big probability numerals and the split win-meter."""
    win, lose = win_prob * 100, (1 - win_prob) * 100
    html(f"""
    <section class="bug" style="--p:{win:.2f}%">
      <div class="bug-top">
        <div class="bug-side">{crest(bat, 54)}
          <div><div class="bug-team">{_esc(bat)}</div><div class="bug-role role-chase">Chasing {target}</div></div></div>
        <div class="bug-centre"><div class="bug-score">{score}<span>/{wickets}</span></div>
          <div class="bug-over">{balls // 6}.{balls % 6} overs</div></div>
        <div class="bug-side right">{crest(bowl, 54)}
          <div><div class="bug-team">{_esc(bowl)}</div><div class="bug-role role-defend">Defending</div></div></div>
      </div>
      <div class="bug-prob">
        <div class="prob-num chase">{win:.1f}<small>%</small></div>
        <div class="prob-label">Win probability</div>
        <div class="prob-num defend">{lose:.1f}<small>%</small></div>
      </div>
      <div class="meter"><i class="fill" style="width:{win:.2f}%"></i><i class="mid"></i></div>
      <div class="bug-foot"><div class="bug-summary">{summary}</div>
        <div class="bug-tags">{badge}{mode_tag(mode)}{model_tag()}</div></div>
    </section>""")


def strip(cells: list[tuple], cols: int | None = None) -> None:
    """cells = [(label, value, colour, sub)] rendered as one scoreboard strip."""
    n = cols or len(cells)
    body = "".join(
        f'<div class="cell"><div class="cell-label">{label}</div>'
        f'<div class="cell-value {color}">{value}</div>'
        + (f'<div class="cell-sub">{sub}</div>' if sub else "") + "</div>"
        for label, value, color, sub in cells)
    html(f'<div class="strip n{n}">{body}</div>')


def situation_strip(score: int, wickets: int, balls: int, target: int) -> None:
    left = 120 - balls
    need = max(target - score, 0)
    rrr = need / (left / 6) if left > 0 else 0.0
    crr = score / (balls / 6) if balls else 0.0
    strip([
        ("Runs needed", str(need), "hi", f"off {left} balls"),
        ("Required rate", f"{rrr:.2f}", "", "runs per over"),
        ("Current rate", f"{crr:.2f}", "", "runs per over"),
        ("Wickets in hand", str(10 - wickets), "hi" if wickets >= 7 else "", f"{wickets} down"),
    ])


def facts(items: list[tuple[str, str]]) -> None:
    html('<div class="facts">' + "".join(
        f'<div class="fact"><span>{k}</span><b>{v}</b></div>' for k, v in items) + "</div>")


def tracker(log: list[str]) -> None:
    """Broadcast-style recent deliveries: a dot/number per ball, a divider between overs."""
    if not log:
        html('<div class="tracker"><span class="tracker-lab">Recent deliveries</span>'
             '<span class="tracker-empty">Bowl a ball to start the chase.</span></div>')
        return
    out = []
    for i, ev in enumerate(log[-18:]):
        cls = {"W": "bW", "4": "b4", "6": "b6", "0": "b0"}.get(ev, "")
        out.append(f'<span class="ball {cls}">{"•" if ev == "0" else ev}</span>')
    html('<div class="tracker"><span class="tracker-lab">Recent deliveries</span>' + "".join(out) + "</div>")


def spark(points: list[tuple[float, float]], color: str) -> str:
    """Tiny inline-SVG win-probability curve for cards."""
    if len(points) < 2:
        return ""
    w, h = 120, 44
    path = " ".join(f"{x / 120 * w:.1f},{h - y * (h - 4) - 2:.1f}" for x, y in points)
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><line x1="0" x2="{w}" y1="{h / 2}" '
            f'y2="{h / 2}" stroke="#2B4A8C" stroke-dasharray="3 3"/><polyline points="{path}" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/></svg>')


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------
def chart_layout(fig: go.Figure, height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_UI, color=INK3, size=12), height=height, showlegend=legend,
        margin=dict(l=44, r=16, t=14, b=38), bargap=.35,
        hoverlabel=dict(bgcolor="#163577", bordercolor=LINE, font=dict(family=FONT_UI, color=INK, size=13)),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK2)),
    )
    fig.update_xaxes(gridcolor="rgba(77,163,245,.10)", zeroline=False, linecolor=LINE, ticks="",
                     tickfont=dict(family=FONT_NUM, size=13, color=INK3), title_font=dict(size=12, color=INK3))
    fig.update_yaxes(gridcolor="rgba(77,163,245,.16)", zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
                     tickfont=dict(family=FONT_NUM, size=13, color=INK3), title_font=dict(size=12, color=INK3))
    return fig


def _over_axis(fig: go.Figure) -> None:
    fig.update_xaxes(range=[0, 120], tickvals=list(range(0, 121, 12)),
                     ticktext=[str(i) for i in range(0, 21, 2)], title_text="Overs", title_standoff=6)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def trend_chart(points: list[dict], bat: str, bowl: str, marker_ball: int | None = None,
                moments: list[dict] | None = None, height: int = 400) -> None:
    """Win-probability 'worm': blue fill above 50% (chase ahead), white below (defence ahead)."""
    x = [p["balls"] for p in points]
    y = [p["win_prob"] * 100 for p in points]
    fig = go.Figure()
    base = dict(x=x, y=[50] * len(x), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_trace(go.Scatter(**base))
    fig.add_trace(go.Scatter(x=x, y=[max(v, 50) for v in y], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(22,119,210,.38)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(**base))
    fig.add_trace(go.Scatter(x=x, y=[min(v, 50) for v in y], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(255,255,255,.12)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=INK, width=2.6), showlegend=False,
        customdata=[[f"{p['runs']}/{p['wickets']}", f"{p['balls'] // 6}.{p['balls'] % 6}"] for p in points],
        hovertemplate=f"<b>{_esc(bat)}</b> %{{y:.1f}}%<br>%{{customdata[0]}} after %{{customdata[1]}} ov<extra></extra>"))
    if x:
        fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(size=11, color=BLUE_HI if y[-1] >= 50 else WHITE,
                                             line=dict(color=INK, width=2))))
    if moments:
        fig.add_trace(go.Scatter(
            x=[m["balls"] for m in moments], y=[m["win_prob"] * 100 for m in moments], mode="markers",
            showlegend=False,
            marker=dict(size=12, symbol=["diamond" if m["wickets"] else "circle" for m in moments],
                        line=dict(color=NAVY, width=1.5), color=SAFFRON),
            customdata=[[m["over"], m["event"], m["delta_pct"]] for m in moments],
            hovertemplate="Over %{customdata[0]} · %{customdata[1]}<br>swing %{customdata[2]:+.1f} pts<extra></extra>"))
    if len(points) < 2:
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
                           text="Bowl a ball or move the slider<br>to draw the win-probability line",
                           font=dict(family=FONT_UI, size=13, color=INK3))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,.32)", line_width=1)
    if marker_ball is not None:
        fig.add_vline(x=marker_ball, line_dash="dot", line_color="rgba(255,255,255,.3)", line_width=1)
    _over_axis(fig)
    fig.update_yaxes(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    fig.add_annotation(x=0.005, y=0.98, xref="paper", yref="paper", text=f"<b>{_esc(bat)}</b> ahead",
                       showarrow=False, font=dict(color=BLUE_HI, size=12, family=FONT_NUM), xanchor="left", yanchor="top")
    fig.add_annotation(x=0.005, y=0.02, xref="paper", yref="paper", text=f"<b>{_esc(bowl)}</b> ahead",
                       showarrow=False, font=dict(color=WHITE, size=12, family=FONT_NUM), xanchor="left", yanchor="bottom")
    show(chart_layout(fig, height))


def factor_chart(expl: dict) -> None:
    factors = sorted(expl["factors"], key=lambda f: f["impact_pct"])
    fig = go.Figure(go.Bar(
        x=[f["impact_pct"] for f in factors], y=[f["factor"] for f in factors], orientation="h",
        marker=dict(color=[BLUE if f["impact_pct"] >= 0 else "#C9D6EE" for f in factors]),
        text=[f"{f['impact_pct']:+.1f}" for f in factors], textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT_NUM, size=15, color=INK), hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
    span = max(8, max(abs(f["impact_pct"]) for f in factors) * 1.5)
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor="rgba(255,255,255,.4)",
                     title_text="Percentage points vs. an average chase")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(family=FONT_UI, size=12, color=INK2))
    fig.update_layout(margin=dict(l=150, r=30, t=8, b=40))
    show(chart_layout(fig, 280))


def whatif_chart(w: dict) -> None:
    sc = w["scenarios"]
    cur = w["current_win_probability"] * 100
    fig = go.Figure(go.Bar(
        x=[s["scenario"].split(" (")[0].replace(" over", "").replace(" + ", "<br>+ ").replace("Two wickets", "2 wickets") for s in sc], y=[s["win_probability"] * 100 for s in sc],
        marker=dict(color=[BLUE if s["change_pct"] >= 0 else "#C9D6EE" for s in sc]),
        text=[f"{s['win_probability'] * 100:.0f}%<br><span style='font-size:11px'>{s['change_pct']:+.0f}</span>" for s in sc],
        textposition="outside", cliponaxis=False, textfont=dict(family=FONT_NUM, size=16, color=INK),
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=cur, line_dash="dash", line_color=SAFFRON, line_width=1.4,
                  annotation_text=f"now {cur:.0f}%", annotation_position="top right",
                  annotation_font=dict(color=SAFFRON, family=FONT_NUM, size=13))
    fig.update_yaxes(range=[0, 118], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    fig.update_xaxes(tickfont=dict(family=FONT_UI, size=11, color=INK2), tickangle=0)
    show(chart_layout(fig, 330))


def heatmap_chart(h: dict, runs_needed_now: int, balls_bowled_now: int) -> None:
    z = [[v * 100 for v in row[::-1]] for row in h["win_probability"]]
    x = [120 - b for b in h["balls_left"]][::-1]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=h["runs_needed"], zmin=0, zmax=100, zsmooth="best",
        colorscale=[[0, "#C3D2EC"], [.28, "#6E8FC4"], [.5, "#12306F"], [.74, "#1565B8"], [1, "#3D9BF0"]],
        colorbar=dict(ticksuffix="%", thickness=10, len=.9,
                      outlinewidth=0, tickfont=dict(family=FONT_NUM, size=12, color=INK3)),
        hovertemplate="after %{x} balls · %{y} runs needed<br>win %{z:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[balls_bowled_now], y=[runs_needed_now], mode="markers", hoverinfo="skip",
                             showlegend=False, marker=dict(size=16, color="rgba(0,0,0,0)", symbol="circle",
                                                           line=dict(width=3, color=SAFFRON))))
    _over_axis(fig)
    fig.update_yaxes(title_text="Runs still needed")
    show(chart_layout(fig, 330))


# ---------------------------------------------------------------------------
# Live Cricket: match cards
# ---------------------------------------------------------------------------
def _abbr(name: str) -> str:
    meta = team_meta().get(name)
    if meta:
        return meta["abbr"]
    low = name.lower()
    if low.startswith("india"):
        if "women" in low:
            return "IND W"
        if low.replace("-", " ").startswith("india a"):
            return "IND A"
        return "IND"
    words = [w for w in name.replace("-", " ").split() if w]
    return (words[0][:3] if len(words) == 1 else "".join(w[0] for w in words)[:3]).upper() or name[:3].upper()


def _team_score(m: dict, team: str) -> str:
    parts = []
    for sc in m.get("scores", []):
        if sc.get("team", "").lower() != team.lower() or sc.get("runs") is None:
            continue
        w = sc.get("wickets")
        parts.append(f"{sc['runs']}" + (f"/{w}" if w not in (None, 10) else ""))
    return " &amp; ".join(parts)


def _fmt_when(m: dict) -> str:
    from datetime import datetime, timedelta
    if m.get("start_utc"):
        try:
            utc = datetime.fromisoformat(m["start_utc"].replace("Z", "+00:00"))
            ist = utc + timedelta(hours=5, minutes=30)
            return f"{ist:%a %d %b, %H:%M} IST · {utc:%H:%M} UTC"
        except ValueError:
            pass
    return m.get("date") or ""


GROUP_LABEL = {"india_men": "India Men", "india_women": "India Women", "india_a": "India A"}


def match_card(m: dict) -> None:
    mode = {"live": "live", "upcoming": "upcoming", "result": "historical"}[m["state"]]
    rows = []
    for team in m["teams"]:
        sc = _team_score(m, team)
        rows.append(f'<div class="mc-row"><span class="mc-abbr">{_esc(_abbr(team))}</span>'
                    f'<span class="mc-name">{_esc(team)}</span><span class="mc-score">{sc or "&nbsp;"}</span></div>')
    status = m.get("result_text") or m.get("status_text") or ""
    group_tags = "".join(f'<span class="tag tag-model">{GROUP_LABEL[g]}</span>' for g in m.get("groups", []))
    meta_bits = [b for b in (_esc(m.get("venue") or ""), _fmt_when(m)) if b]
    src = {"cricketdata": "CricketData.org", "cricsheet": "Cricsheet.org"}.get(m["source"], m["source"])
    html(f"""
    <div class="mcard mcard-{m['state']}">
      <div class="mc-top"><span class="tag">{_esc(m['format_label'])}</span>{group_tags}
        <span class="mc-series">{_esc(m.get('series') or '')}</span>{mode_tag(mode)}</div>
      {''.join(rows)}
      <div class="mc-status">{_esc(status)}</div>
      <div class="mc-meta">{' · '.join(meta_bits)}<span class="mc-src">Source: {src}</span></div>
    </div>""")
