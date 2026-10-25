"""
ChaseIQ - Inside Every Chase. Streamlit dashboard
=======================================
    streamlit run dashboard/app.py

A pure client: every number comes from the FastAPI backend (set API_BASE).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow `streamlit run dashboard/app.py`

from dashboard import api_client as api  # noqa: E402
from dashboard import components as ui  # noqa: E402
from dashboard.views import analytics, insights, live, replay, simulator  # noqa: E402

st.set_page_config(page_title="ChaseIQ - Inside Every Chase", page_icon="🏏", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700'
    '&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400'
    '&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True)
st.markdown(f"<style>{(Path(__file__).parent / 'styles.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

# ---- header: brand + system status ---------------------------------------------------
health, err = api.health()
test = None
if health:
    info, _ = api.model_info()
    m = (info or {}).get("metrics", {})
    test = m.get("candidates", {}).get(m.get("selected_model", ""), {}).get("test")
ui.topbar(health, test)

# ---- navigation -------------------------------------------------------------------------
VIEWS = {
    "Live Cricket": live.render,
    "Simulator": simulator.render,
    "Replay": replay.render,
    "Analytics": analytics.render,
    "Model": insights.render,
}
DEFAULT_VIEW = "Live Cricket"
def _keep_nav() -> None:
    """Clicking the active tab would deselect it - restore the previous choice instead."""
    s = st.session_state
    if s.nav_view is None:
        s.nav_view = s.get("_nav_last", DEFAULT_VIEW)
    s._nav_last = s.nav_view


st.session_state.setdefault("nav_view", DEFAULT_VIEW)   # no `default=`: the callback below also writes the key
with st.container(key="nav"):
    st.segmented_control("Section", list(VIEWS), key="nav_view", label_visibility="collapsed",
                         on_change=_keep_nav)
choice = st.session_state.nav_view or st.session_state.get("_nav_last", DEFAULT_VIEW)
st.session_state._nav_last = choice

if err:
    ui.empty("The API is not reachable", f"{err}<br><br>Start the backend with "
             "<code>uvicorn api.main:app --reload</code> (or <code>python run.py</code>).")
else:
    VIEWS[choice]()      # only the selected view runs -> no wasted API calls

# ---- footer ---------------------------------------------------------------------------
ui.html(f"""<div class="foot"><span>ChaseIQ · IPL Second-Innings Chase Model (IPL 2008-2025, never used for ODI/T20I/Test) ·
    Results data: Cricsheet.org (ODC-BY) · Live &amp; fixtures: CricketData.org ·
    <a href="{api.API_BASE}/docs" target="_blank">API ↗</a></span>
    <span>Inside Every Chase.</span></div>""")
if st.button("Refresh data", key="refresh"):
    st.cache_data.clear()
    st.rerun()
