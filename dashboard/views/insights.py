"""Model: how it was built, tested, and where it can (and cannot) be trusted."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import api_client as api
from dashboard import components as ui
from dashboard.components import BLUE, BLUE_HI, INK, chart_layout, show

LABELS = {"runs_remaining": "Runs still needed", "balls_remaining": "Balls left", "wickets_remaining": "Wickets in hand",
          "required_run_rate": "Required run rate", "current_run_rate": "Current run rate",
          "run_rate_diff": "Run-rate gap", "target_score": "Target", "target_vs_venue": "Target vs ground avg",
          "elo_diff": "Elo gap", "batting_team_form": "Recent form", "head_to_head_ratio": "Head-to-head",
          "venue_chase_bias": "Ground chase bias", "toss_advantage": "Toss"}


def render() -> None:
    info, err = api.model_info()
    if err:
        ui.empty("Model information unavailable", err)
        return
    m = info["metrics"]
    best = m["candidates"][m["selected_model"]]["test"]

    ui.html(f'<div class="bug-tags" style="justify-content:flex-start;margin:18px 0 0">{ui.model_tag()}</div>')
    ui.notice("Scope", info.get("scope", ""), "info")
    ui.section("01", "How good is it, honestly?",
               f"{m['selected_model']}, scored on {m['rows']['test_matches']} IPL matches from 2023-25 "
               f"({m['rows']['test']:,} ball-by-ball situations) that were never used for training or tuning.")
    ui.strip([("Accuracy", f"{best['accuracy']:.1%}", "hi", f"baseline {m['baseline_accuracy']:.1%}"),
              ("ROC-AUC", f"{best['roc_auc']:.3f}", "", "ranking quality"),
              ("Brier score", f"{best['brier']:.3f}", "", "lower is better"),
              ("Log-loss", f"{best['log_loss']:.3f}", "", "lower is better")])

    ui.section("02", "How it was built")
    ui.html("""<div class="steps">
      <div class="step"><div class="step-no">STEP 1</div><div class="step-t">Clean</div>
        <div class="step-d">Teams mapped to franchises, 59 venue spellings merged into 37 grounds, ties, rain-hit and
        super-over data removed.</div></div>
      <div class="step"><div class="step-no">STEP 2</div><div class="step-t">Engineer</div>
        <div class="step-d">Required rate, wickets in hand, target vs ground, plus Elo, form and head-to-head
        computed only from earlier matches.</div></div>
      <div class="step"><div class="step-no">STEP 3</div><div class="step-t">Select</div>
        <div class="step-d">Five candidates compared on 2020-22 after training on seasons up to 2019. Lowest
        log-loss wins.</div></div>
      <div class="step"><div class="step-no">STEP 4</div><div class="step-t">Test</div>
        <div class="step-d">Refit up to 2022, then score once on 2023-25. Splits are chronological, never random.</div></div>
    </div>""")

    ui.section("03", "Candidates", "Highlighted row is the model in production. Orange marks the best value per column.")
    cols = [("Val log-loss", "validation", "log_loss", False), ("Test accuracy", "test", "accuracy", True),
            ("Test AUC", "test", "roc_auc", True), ("Test Brier", "test", "brier", False),
            ("Test log-loss", "test", "log_loss", False)]
    ranges = {c[0]: [r[c[1]][c[2]] for r in m["candidates"].values()] for c in cols}
    body = []
    for name, r in m["candidates"].items():
        cells = []
        for label, split, key, higher in cols:
            v = r[split][key]
            is_best = v == (max if higher else min)(ranges[label])
            txt = f"{v:.1%}" if key == "accuracy" else f"{v:.4f}" if key != "roc_auc" else f"{v:.3f}"
            cells.append(f'<td class="{"best" if is_best else ""}">{txt}</td>')
        body.append(f'<tr class="{"sel" if name == m["selected_model"] else ""}"><td>{name}</td>{"".join(cells)}</tr>')
    ui.html('<table class="mt"><thead><tr><th>Model</th>' + "".join(f"<th>{c[0]}</th>" for c in cols)
            + f'</tr></thead><tbody>{"".join(body)}</tbody></table>')
    st.caption("With ~110 correlated rows per match but one outcome, deep trees mostly memorise matches: the simple "
               "model is competitive, and regularising plus blending wins.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.chart_head("Calibration", "When the model says 70%, do teams win 70% of the time? On the diagonal = yes.")
        cal = pd.DataFrame(m["calibration"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="rgba(255,255,255,.45)"),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers", showlegend=False,
                                 line=dict(color=BLUE, width=3),
                                 marker=dict(size=cal["count"] ** .5 / 3.5 + 6, color=BLUE, line=dict(color=INK, width=1.5)),
                                 hovertemplate="predicted %{x:.0%}<br>actually won %{y:.0%}<extra></extra>"))
        fig.update_xaxes(title_text="Predicted win probability", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title_text="Observed win rate", tickformat=".0%", range=[0, 1])
        show(chart_layout(fig, 360))
    with c2:
        ui.chart_head("Accuracy by stage", "It sharpens as the chase unfolds. Dotted line = majority-class baseline.")
        st_df = pd.DataFrame(m["test_by_stage"])
        fig = go.Figure(go.Bar(x=st_df["stage"], y=st_df["accuracy"], marker=dict(color=BLUE),
                               text=[f"{a:.0%}" for a in st_df["accuracy"]], textposition="outside", cliponaxis=False,
                               textfont=dict(family="Barlow Condensed", size=18, color=INK)))
        fig.add_hline(y=m["baseline_accuracy"], line_dash="dot", line_color="rgba(255,255,255,.5)")
        fig.update_yaxes(range=[0.4, 1.0], tickformat=".0%")
        show(chart_layout(fig, 360))

    fi = m.get("feature_importance")
    c1, c2 = st.columns([7, 5], gap="large")
    with c1:
        if fi:
            ui.chart_head("What the model pays attention to", "Share of importance, blended across both components.")
            items = sorted(fi.items(), key=lambda kv: kv[1])
            fig = go.Figure(go.Bar(x=[v for _, v in items], y=[LABELS.get(k, k) for k, _ in items], orientation="h",
                                   marker=dict(color=BLUE)))
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(family="Manrope", size=12, color="#D3DCEE"))
            fig.update_layout(margin=dict(l=140, r=16, t=8, b=30))
            show(chart_layout(fig, 400))
    with c2:
        ck = m.get("test_by_checkpoint", {})
        ui.chart_head("Where to trust it - and where not to")
        early = f" ({ck['after_0_overs']['accuracy']:.0%} before a ball is bowled)" if "after_0_overs" in ck else ""
        ui.html(f"""<ul class="bullets">
          <li><b>Strong late:</b> {m['test_by_stage'][-1]['accuracy']:.0%} accuracy in the last five overs.</li>
          <li><b>Near a coin-flip early</b>{early}, tilted by target size and team strength.</li>
          <li><b>Scoreboard only:</b> no players, pitch, dew or in-over momentum.</li>
          <li><b>Team strength</b> is a season-regressed Elo, so a newly rebuilt squad can surprise it.</li>
          <li><b>Replay caveat:</b> matches before 2023 are in-sample for the model.</li></ul>""")
