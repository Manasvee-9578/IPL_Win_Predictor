"""Model & prediction-engine behaviour: the properties a cricket fan would expect."""

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import make_features
from src.predictor import Situation
from src.reference import canonical_venue, team_id


@pytest.fixture()
def base(predictor):
    return dict(
        batting_team=predictor.resolve_team("MI"),
        bowling_team=predictor.resolve_team("CSK"),
        venue="Wankhede Stadium, Mumbai",
        toss_winner=predictor.resolve_team("MI"),
        target_score=180, current_score=90, wickets_fallen=3, balls_bowled=60,
    )


def prob(predictor, base, **override):
    return predictor.predict(Situation(**{**base, **override}), explain=False)["win_probability"]


def test_probability_is_valid(predictor, base):
    assert 0.0 < prob(predictor, base) < 1.0


def test_more_runs_scored_helps(predictor, base):
    p = [prob(predictor, base, current_score=s) for s in range(60, 170, 20)]
    assert p == sorted(p)


def test_more_wickets_lost_hurts(predictor, base):
    p = [prob(predictor, base, wickets_fallen=w) for w in range(0, 10)]
    assert p == sorted(p, reverse=True)


def test_same_score_later_is_worse(predictor, base):
    early = prob(predictor, base, current_score=60, balls_bowled=60)
    late = prob(predictor, base, current_score=60, balls_bowled=90)
    assert late < early


def test_terminal_states(predictor, base):
    cases = {
        "chase_complete": dict(current_score=180),
        "chase_failed": dict(wickets_fallen=10),
        "tied": dict(current_score=179, balls_bowled=120),
    }
    for status, override in cases.items():
        out = predictor.predict(Situation(**{**base, **override}))
        assert out["status"] == status
    assert predictor.predict(Situation(**{**base, "current_score": 180}))["win_probability"] == 1.0
    assert predictor.predict(Situation(**{**base, "wickets_fallen": 10}))["win_probability"] == 0.0


def test_explanation_adds_up(predictor, base):
    """base log-odds + sum of contributions == the model's log-odds (exact additivity)."""
    s = Situation(**base)
    ctx = predictor.context(s.batting_team, s.bowling_team, s.venue)
    X = predictor._features(s, ctx)
    parts, b = predictor.model.contributions(X)
    logit_from_parts = float(b[0] + parts.sum(axis=1).iloc[0])
    p = float(predictor.model.predict_proba(X)[0, 1])
    assert logit_from_parts == pytest.approx(np.log(p / (1 - p)), abs=1e-3)


def test_what_if_ordering(predictor, base):
    out = predictor.what_if(Situation(**base))
    probs = {r["scenario"]: r["win_probability"] for r in out["scenarios"]}
    assert probs["Big over (15 runs)"] > probs["Par over (9 runs)"] > probs["Maiden over (0 runs)"]
    assert probs["Two wickets + 4 runs"] < probs["Wicket + 5 runs"]


def test_heatmap_shape_and_trend(predictor, base):
    h = predictor.heatmap(Situation(**base), wickets=3)
    grid = np.array(h["win_probability"])
    assert grid.shape == (len(h["runs_needed"]), len(h["balls_left"]))
    assert (grid[0] >= grid[-1]).all()          # fewer runs needed -> never worse


def test_no_train_serve_skew(predictor):
    """The feature table used for training is reproduced exactly by make_features."""
    df = pd.read_csv(C.FEATURES_CSV).sample(200, random_state=0)
    again = make_features(
        target_score=df["target_score"], current_score=df["target_score"] - df["runs_remaining"],
        wickets_fallen=10 - df["wickets_remaining"], balls_bowled=df["balls_bowled"],
        elo_diff=df["elo_diff"], batting_team_form=df["batting_team_form"],
        head_to_head_ratio=df["head_to_head_ratio"],
        venue_avg_score=df["target_score"] - df["target_vs_venue"],
        venue_chase_bias=df["venue_chase_bias"], toss_advantage=df["toss_advantage"])
    np.testing.assert_allclose(again.to_numpy(), df[C.FEATURE_COLS].to_numpy(), atol=1e-6)


def test_reference_data():
    assert team_id("Royal Challengers Bangalore") == team_id("RCB") == 1
    assert team_id("Delhi Daredevils") == team_id("Delhi Capitals")
    assert team_id("nonsense") is None
    assert canonical_venue("Wankhede Stadium") == canonical_venue("Wankhede Stadium, Mumbai")


def test_venue_resolution(predictor):
    assert predictor.resolve_venue("Wankhede Stadium Mumbai") == ("Wankhede Stadium, Mumbai", True)
    assert predictor.resolve_venue("Somewhere Unknown Ground")[1] is False


def test_selected_model_beats_baseline(predictor):
    m = predictor.metrics
    best = m["candidates"][m["selected_model"]]["test"]
    assert best["accuracy"] > m["baseline_accuracy"] + 0.15
    assert best["roc_auc"] > 0.85
