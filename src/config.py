"""Central configuration: paths and modelling constants shared by every module."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "database" / "ipl.db"

MATCHES_CSV = RAW_DIR / "matches.csv"
DELIVERIES_CSV = RAW_DIR / "deliveries.csv"
MATCHES_CLEAN_CSV = PROCESSED_DIR / "matches_clean.csv"
CHASE_STATES_CSV = PROCESSED_DIR / "chase_states.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

MODEL_PKL = MODELS_DIR / "ipl_model.pkl"
FEATURES_JSON = MODELS_DIR / "feature_columns.json"
METRICS_JSON = MODELS_DIR / "metrics.json"

TOTAL_BALLS = 120
TOTAL_WICKETS = 10

# Model features, in the exact order the model expects them.
FEATURE_COLS = [
    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "required_run_rate",
    "current_run_rate",
    "run_rate_diff",
    "target_score",
    "target_vs_venue",
    "elo_diff",
    "batting_team_form",
    "head_to_head_ratio",
    "venue_chase_bias",
    "toss_advantage",
]
TARGET = "batting_team_wins"

# Season-based evaluation protocol (no random shuffling -> no future leakage).
TRAIN_END = 2022          # final model is trained on seasons <= 2022
VALID_START = 2020        # model selection: fit <= 2019, validate 2020-2022
TEST_START = 2023         # honest hold-out: 2023-2025

# Rating system defaults
ELO_START = 1500.0
ELO_K = 24.0
ELO_SEASON_REGRESSION = 0.25   # pull 25% towards the mean each new season
FORM_WINDOW = 5
PRIOR_FIRST_INNINGS = 165.0
PRIOR_RUN_RATE = 8.0           # league-typical runs per over, used to shrink early run rates
PRIOR_OVERS = 1.0
