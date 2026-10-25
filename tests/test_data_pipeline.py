"""Checks on the processed tables (skipped when the raw data / pipeline output is absent)."""

import pandas as pd
import pytest

from src import config as C

pytestmark = pytest.mark.skipif(not C.CHASE_STATES_CSV.exists(),
                                reason="run `python -m src.run_pipeline` first")


@pytest.fixture(scope="module")
def states():
    return pd.read_csv(C.CHASE_STATES_CSV)


def test_no_terminal_states(states):
    assert (states["cum_runs"] < states["target_score"]).all()
    assert (states["cum_wickets"] < 10).all()
    assert (states["balls_bowled"] < 120).all()


def test_every_chase_starts_at_ball_zero(states):
    first = states.groupby("match_id")["balls_bowled"].min()
    assert (first == 0).all()


def test_runs_and_wickets_never_decrease(states):
    s = states.sort_values(["match_id", "balls_bowled"])
    assert (s.groupby("match_id")["cum_runs"].diff().fillna(0) >= 0).all()
    assert (s.groupby("match_id")["cum_wickets"].diff().fillna(0) >= 0).all()


def test_label_is_constant_within_a_match(states):
    assert (states.groupby("match_id")["batting_team_wins"].nunique() == 1).all()


def test_canonical_venues_collapse_duplicates():
    matches = pd.read_csv(C.MATCHES_CLEAN_CSV)
    assert matches["venue"].nunique() < 40
    assert not any("Wankhede" in v and v == "Wankhede Stadium" for v in matches["venue"].unique())


def test_features_have_no_missing_values():
    df = pd.read_csv(C.FEATURES_CSV)
    assert not df[C.FEATURE_COLS + [C.TARGET]].isna().any().any()
