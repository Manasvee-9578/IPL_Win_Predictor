"""
IPL Win Probability Predictor — Premium Dashboard
===================================================
Premium glassmorphic dark theme with live match intelligence.

Usage:
    streamlit run dashboard/app.py
"""

import logging
import os
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
)

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
LIVE_REFRESH_SECONDS = int(os.getenv("LIVE_REFRESH_SECONDS", "30"))

# ---------------------------------------------------------------------------
# Team metadata — initials + brand colours for logo circles
# ---------------------------------------------------------------------------
TEAM_META = {
    "Chennai Super Kings":         {"abbr": "CSK", "color": "#f9cd05"},
    "Delhi Capitals":              {"abbr": "DC",  "color": "#004c93"},
    "Gujarat Titans":              {"abbr": "GT",  "color": "#1c1c2b"},
    "Kolkata Knight Riders":       {"abbr": "KKR", "color": "#3a225d"},
    "Lucknow Super Giants":       {"abbr": "LSG", "color": "#a72056"},
    "Mumbai Indians":              {"abbr": "MI",  "color": "#004ba0"},
    "Punjab Kings":                {"abbr": "PBKS","color": "#ed1b24"},
    "Rajasthan Royals":            {"abbr": "RR",  "color": "#ea1a85"},
    "Royal Challengers Bengaluru": {"abbr": "RCB", "color": "#d4213d"},
    "Sunrisers Hyderabad":         {"abbr": "SRH", "color": "#ff822a"},
}

def _team_logo_html(team_name: str, size: int = 44) -> str:
    meta = TEAM_META.get(team_name, {"abbr": team_name[:3].upper(), "color": "#555"})
    return (
        f'<div style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:{size}px;height:{size}px;border-radius:50%;'
        f'background:linear-gradient(135deg, {meta["color"]}, {meta["color"]}99);'
        f'color:#fff;font-weight:800;font-size:{size//3}px;letter-spacing:.5px;'
        f'box-shadow:0 0 18px {meta["color"]}55, inset 0 0 12px rgba(255,255,255,.12);'
        f'flex-shrink:0;border:2px solid {meta["color"]}88;">'
        f'{meta["abbr"]}</div>'
    )


# ---------------------------------------------------------------------------
# Load external CSS + Google Fonts
# ---------------------------------------------------------------------------
_css_path = Path(__file__).parent / "styles.css"
_css_content = _css_path.read_text(encoding="utf-8") if _css_path.is_file() else ""

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown("<style>" + _css_content + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hdr-bar">
    <div class="hdr-title">🏏 IPL Win Probability Predictor</div>
    <div class="hdr-sub">Live Match Intelligence · Powered by XGBoost</div>
    <div class="hdr-badge">⚡ REAL-TIME ANALYSIS</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_teams() -> list[str]:
    """Fetch available teams from the API, with local fallback."""
    try:
        resp = requests.get(f"{API_BASE}/teams", timeout=3)
        resp.raise_for_status()
        return resp.json()["teams"]
    except Exception as exc:
        logger.warning("Failed to fetch teams from API: %s", exc)
        return list(TEAM_META.keys())

@st.cache_data(ttl=300)
def fetch_venues() -> list[str]:
    """Fetch available venues from the API, with local fallback."""
    try:
        resp = requests.get(f"{API_BASE}/venues", timeout=3)
        resp.raise_for_status()
        return resp.json()["venues"]
    except Exception as exc:
        logger.warning("Failed to fetch venues from API: %s", exc)
        return ["Wankhede Stadium, Mumbai", "M Chinnaswamy Stadium"]

def call_predict(payload: dict) -> dict | None:
    """Send a prediction request. Returns response dict or None on failure."""
    try:
        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Predict returned %d: %s", resp.status_code, resp.text)
        return None
    except Exception as exc:
        logger.warning("Predict request failed: %s", exc)
        return None

def call_live() -> dict | None:
    """Fetch the latest live match data. Returns dict or None."""
    try:
        resp = requests.get(f"{API_BASE}/live", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Handle structured response wrapper
            return data.get("data", data)
        return None
    except Exception as exc:
        logger.warning("Live request failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "live_history" not in st.session_state:
    st.session_state.live_history = []
if "sim_balls" not in st.session_state:
    st.session_state.sim_balls = 30
if "sim_score" not in st.session_state:
    st.session_state.sim_score = 45
if "sim_wickets" not in st.session_state:
    st.session_state.sim_wickets = 2


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------
def render_scorecard(score, wickets, balls, target):
    """Render the four-card scorecard strip."""
    overs_bowled = balls / 6.0
    overs_remaining = (120 - balls) / 6.0
    runs_remaining = target - score
    crr = score / overs_bowled if overs_bowled > 0 else 0.0
    rrr = runs_remaining / overs_remaining if overs_remaining > 0 else 0.0
    overs_str = f"{balls // 6}.{balls % 6}"

    st.markdown(f"""
    <div class="sc-row">
        <div class="sc-card">
            <div class="sc-value clr-cyan">{score}/{wickets}</div>
            <div class="sc-label">Score · {overs_str} ov</div>
        </div>
        <div class="sc-card">
            <div class="sc-value clr-red">{wickets}</div>
            <div class="sc-label">Wickets Fallen</div>
        </div>
        <div class="sc-card">
            <div class="sc-value clr-amber">{rrr:.2f}</div>
            <div class="sc-label">Required Run Rate</div>
        </div>
        <div class="sc-card">
            <div class="sc-value clr-green">{crr:.2f}</div>
            <div class="sc-label">Current Run Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_match_state(bat_team, bowl_team, win_prob, lose_prob, summary):
    """Render the head-to-head probability bar."""
    win_pct = win_prob * 100
    lose_pct = lose_prob * 100
    st.markdown(f"""
    <div class="ms-bar">
        <div class="ms-teams">
            <div class="ms-team">
                {_team_logo_html(bat_team, 52)}
                <div class="ms-prob clr-cyan">{win_pct:.1f}%</div>
                <div class="ms-name clr-cyan">{bat_team}</div>
            </div>
            <div class="ms-vs clr-grey">VS</div>
            <div class="ms-team">
                {_team_logo_html(bowl_team, 52)}
                <div class="ms-prob clr-red">{lose_pct:.1f}%</div>
                <div class="ms-name clr-red">{bowl_team}</div>
            </div>
        </div>
        <div class="ms-progress">
            <div class="ms-fill-bat" style="width:{win_pct}%;"></div>
            <div class="ms-fill-bowl" style="width:{lose_pct}%;"></div>
        </div>
        <div class="ms-summary">{summary}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chart(history, bat_team, bowl_team, current_ball):
    """Render the win probability trend chart."""
    balls_x = [h[0] for h in history]
    win_y = [h[1] for h in history]
    lose_y = [h[2] for h in history]

    fig = go.Figure()

    # Batting team — cyan gradient fill
    fig.add_trace(go.Scatter(
        x=balls_x, y=win_y, mode="lines", name=bat_team,
        line=dict(color="#22d3ee", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(34,211,238,0.08)",
    ))
    # Bowling team — red
    fig.add_trace(go.Scatter(
        x=balls_x, y=lose_y, mode="lines", name=bowl_team,
        line=dict(color="#f43f5e", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(244,63,94,0.05)",
    ))

    # Current ball marker
    fig.add_vline(
        x=current_ball, line_dash="dash", line_color="rgba(255,255,255,.25)",
        line_width=1.5,
    )
    # 50% reference line
    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(255,255,255,.08)", line_width=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.6)",
        font=dict(family="Inter, sans-serif"),
        title=dict(
            text="Win Probability Trend",
            font=dict(size=15, color="#8b8ba0", family="Outfit, sans-serif"),
            x=0.01,
        ),
        xaxis=dict(
            title=None, range=[0, 120], dtick=12,
            showgrid=True,
            gridcolor="rgba(99,102,241,0.06)",
            zeroline=False,
            tickfont=dict(color="#6b7280", size=11),
            linecolor="rgba(99,102,241,.12)",
        ),
        yaxis=dict(
            title=None, range=[0, 100], dtick=25,
            showgrid=True,
            gridcolor="rgba(99,102,241,0.06)",
            zeroline=False,
            tickfont=dict(color="#6b7280", size=11),
            ticksuffix="%",
            linecolor="rgba(99,102,241,.12)",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03,
            xanchor="center", x=0.5,
            font=dict(color="#8b8ba0", size=12, family="Inter, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=42, r=16, t=50, b=32),
        height=400,
        hoverlabel=dict(
            bgcolor="rgba(25,25,50,.9)",
            font_color="#fff",
            bordercolor="rgba(99,102,241,.3)",
            font=dict(family="Inter, sans-serif"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
teams  = fetch_teams()
venues = fetch_venues()

with st.sidebar:
    st.markdown('<div class="sidebar-head"><span class="dot"></span>Match Setup</div>',
                unsafe_allow_html=True)

    batting_team = st.selectbox("Batting Team", teams,
        index=teams.index("Mumbai Indians") if "Mumbai Indians" in teams else 0)
    bowling_team = st.selectbox("Bowling Team",
        [t for t in teams if t != batting_team], index=0)

    # Team badges
    st.markdown(f"""
    <div class="team-badge">
        {_team_logo_html(batting_team, 38)}
        <div>
            <div class="team-badge-name">{batting_team}</div>
            <div class="team-badge-role">Batting</div>
        </div>
    </div>
    <div class="team-badge">
        {_team_logo_html(bowling_team, 38)}
        <div>
            <div class="team-badge-name">{bowling_team}</div>
            <div class="team-badge-role">Bowling</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    venue       = st.selectbox("Venue", venues, index=0)
    toss_winner = st.selectbox("Toss Winner", [batting_team, bowling_team], index=0)
    target_score = st.number_input("Target Score", min_value=1, max_value=350,
                                    value=180, step=1)

    st.markdown("---")
    batting_team_form  = st.slider("Team Form", 0.0, 1.0, 0.5, 0.05)
    head_to_head_ratio = st.slider("Head-to-Head", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    if st.button("🔄 Reset History", use_container_width=True):
        st.session_state.history = []
        st.session_state.live_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════
tab_sim, tab_live = st.tabs(["🏏 Simulator", "📡 Live Match"])


# ─────────────── TAB 1 — SIMULATOR ───────────────
with tab_sim:

    # --- Controls ---
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    with c1:
        st.session_state.sim_balls = st.slider(
            "Balls Bowled", 0, 120, st.session_state.sim_balls, 1,
            key="slider_balls",
        )
    with c2:
        st.session_state.sim_score = st.number_input(
            "Score", 0, 500, st.session_state.sim_score, 1, key="inp_score",
        )
    with c3:
        st.session_state.sim_wickets = st.number_input(
            "Wickets", 0, 10, st.session_state.sim_wickets, 1, key="inp_wkt",
        )
    with c4:
        st.write("")   # spacer for alignment
        st.write("")
        if st.button("⚾ +1 Ball", use_container_width=True, key="btn_ball"):
            if st.session_state.sim_balls < 120:
                st.session_state.sim_balls += 1
                st.rerun()
    with c5:
        st.write("")
        st.write("")
        if st.button("🚨 +1 Wicket", use_container_width=True, key="btn_wkt"):
            if st.session_state.sim_wickets < 10:
                st.session_state.sim_wickets += 1
                st.rerun()

    balls_bowled   = st.session_state.sim_balls
    current_score  = st.session_state.sim_score
    wickets_fallen = st.session_state.sim_wickets

    payload = {
        "batting_team": batting_team, "bowling_team": bowling_team,
        "venue": venue, "toss_winner": toss_winner,
        "target_score": target_score,
        "current_score": current_score,
        "wickets_fallen": wickets_fallen,
        "balls_bowled": balls_bowled,
        "batting_team_form": batting_team_form,
        "head_to_head_ratio": head_to_head_ratio,
    }
    result = call_predict(payload)

    if result is None:
        st.markdown(
            '<div class="no-match"><div class="icon">⚠️</div>'
            '<h3>API Unavailable</h3>'
            '<p>Start the FastAPI server with '
            '<code>uvicorn api.main:app --reload</code></p></div>',
            unsafe_allow_html=True,
        )
    else:
        win_p = result["win_probability"]
        lose_p = result["lose_probability"]
        summary = result["match_state_summary"]

        # Accumulate history
        if not st.session_state.history or st.session_state.history[-1][0] != balls_bowled:
            st.session_state.history.append((balls_bowled, win_p * 100, lose_p * 100))
            st.session_state.history.sort(key=lambda x: x[0])

        render_match_state(batting_team, bowling_team, win_p, lose_p, summary)
        render_scorecard(current_score, wickets_fallen, balls_bowled, target_score)
        render_chart(st.session_state.history, batting_team, bowling_team, balls_bowled)


# ─────────────── TAB 2 — LIVE MATCH ───────────────
with tab_live:
    live_data = call_live()

    if live_data is None:
        st.markdown("""
        <div class="no-match">
            <div class="icon">📡</div>
            <h3>No Live IPL Match</h3>
            <p>
                Waiting for a live 2nd-innings chase to begin.<br>
                Auto-refreshes every 30 seconds.<br><br>
                <span style="font-size:.72rem;color:#6b7280;">
                    Set <code>CRICKET_API_KEY</code> in <code>.env</code>
                    and restart the server.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        updated = live_data.get("last_updated", "")
        st.markdown(
            f'<div class="live-chip"><span class="dot"></span>'
            f'LIVE &nbsp;·&nbsp; {updated[:19].replace("T", " ")} UTC</div>',
            unsafe_allow_html=True,
        )

        live_bat_team = live_data.get("batting_team", "")
        live_bowl_team = live_data.get("bowling_team", "")
        live_win_prob = live_data["win_probability"]
        live_lose_prob = live_data["lose_probability"]
        live_summary = live_data["match_state_summary"]
        live_score = live_data.get("current_score", 0)
        live_wickets = live_data.get("wickets_fallen", 0)
        live_balls = live_data.get("balls_bowled", 0)
        live_target = live_data.get("target_score", 0)

        render_match_state(live_bat_team, live_bowl_team, live_win_prob, live_lose_prob, live_summary)
        render_scorecard(live_score, live_wickets, live_balls, live_target)

        if not st.session_state.live_history or st.session_state.live_history[-1][0] != live_balls:
            st.session_state.live_history.append((live_balls, live_win_prob * 100, live_lose_prob * 100))
            st.session_state.live_history.sort(key=lambda x: x[0])

        render_chart(st.session_state.live_history, live_bat_team, live_bowl_team, live_balls)
