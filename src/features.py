"""
Single source of truth for turning a match situation into model features.

Both the training pipeline (vectorised over ~125k rows) and the live API
(one row) call ``make_features`` - so there can be no train/serve skew.
"""

import numpy as np
import pandas as pd

from src import config as C


def make_features(
    target_score,
    current_score,
    wickets_fallen,
    balls_bowled,
    elo_diff,
    batting_team_form,
    head_to_head_ratio,
    venue_avg_score,
    venue_chase_bias,
    toss_advantage,
) -> pd.DataFrame:
    """Build the model's feature frame. All arguments broadcast (scalar or array)."""
    (target, score, wkts, balls, elo, form, h2h, v_avg, v_bias, toss) = np.broadcast_arrays(
        *[np.atleast_1d(np.asarray(a, dtype=float)) for a in (
            target_score, current_score, wickets_fallen, balls_bowled, elo_diff,
            batting_team_form, head_to_head_ratio, venue_avg_score, venue_chase_bias,
            toss_advantage)]
    )

    runs_remaining = target - score
    balls_remaining = C.TOTAL_BALLS - balls
    wickets_remaining = C.TOTAL_WICKETS - wkts

    overs_bowled = balls / 6.0
    overs_remaining = balls_remaining / 6.0
    # Shrunk run rate: after 1 ball "4 runs / 0.17 overs = 24 rpo" is noise, so blend in
    # a prior of PRIOR_RUN_RATE worth PRIOR_OVERS overs. Converges to the true rate later.
    crr = (score + C.PRIOR_RUN_RATE * C.PRIOR_OVERS) / (overs_bowled + C.PRIOR_OVERS)
    rrr = np.divide(runs_remaining, overs_remaining, out=np.zeros_like(score),
                    where=overs_remaining > 0)

    frame = pd.DataFrame({
        "runs_remaining": runs_remaining,
        "balls_remaining": balls_remaining,
        "wickets_remaining": wickets_remaining,
        "required_run_rate": rrr,
        "current_run_rate": crr,
        "run_rate_diff": crr - rrr,
        "target_score": target,
        "target_vs_venue": target - v_avg,
        "elo_diff": elo,
        "batting_team_form": form,
        "head_to_head_ratio": h2h,
        "venue_chase_bias": v_bias,
        "toss_advantage": toss,
    })
    return frame[C.FEATURE_COLS]
