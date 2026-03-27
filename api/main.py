"""
IPL Win Probability Predictor — FastAPI Backend
================================================
Serves real-time win-probability predictions via a REST API.
Includes a /live endpoint with background CricAPI polling.

Usage:
    uvicorn api.main:app --reload
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import MatchState, PredictionResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (relative to project root, resolved from this file's location)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
MODEL_PKL     = MODELS_DIR / "ipl_model.pkl"
VENUE_ENC_PKL = MODELS_DIR / "venue_encoder.pkl"
FEAT_COLS_PKL = MODELS_DIR / "feature_columns.pkl"

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
model = None
venue_encoder = None
feature_columns = None

# Cached result from the latest CricAPI poll
live_cache: dict | None = None


# ---------------------------------------------------------------------------
# Lifespan — load model artefacts + start live polling
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated @app.on_event."""
    global model, venue_encoder, feature_columns

    # --- Startup ---
    for path, label in [
        (MODEL_PKL, "model"),
        (VENUE_ENC_PKL, "venue encoder"),
        (FEAT_COLS_PKL, "feature columns"),
    ]:
        if not path.is_file():
            raise RuntimeError(
                f"{label} not found at {path}. "
                "Run `python src/train_model.py` first."
            )

    model           = joblib.load(MODEL_PKL)
    venue_encoder   = joblib.load(VENUE_ENC_PKL)
    feature_columns = joblib.load(FEAT_COLS_PKL)

    logger.info("Model loaded          → %s", MODEL_PKL)
    logger.info("Venue encoder loaded  → %s", VENUE_ENC_PKL)
    logger.info("Feature columns       → %s", feature_columns)

    # Launch background live-polling task
    poll_task = asyncio.create_task(_poll_live_data())

    yield  # App is running

    # --- Shutdown ---
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="IPL Win Probability Predictor",
    description="Real-time win probability predictions for IPL matches.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background live polling
# ---------------------------------------------------------------------------
async def _poll_live_data():
    """Infinite loop: poll CricAPI → run prediction → cache result."""
    global live_cache

    # Import here to avoid circular imports at module level
    from src.live_data import fetch_live_match

    api_key = os.getenv("CRICKET_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        logger.warning("CRICKET_API_KEY not set — live polling disabled")
        return

    logger.info("Live polling started (every %ds)", POLL_INTERVAL)

    while True:
        try:
            # Run the blocking HTTP call in a thread so we don't block the event loop
            loop = asyncio.get_running_loop()
            match_data = await loop.run_in_executor(None, fetch_live_match)

            if match_data is not None:
                # Build a MatchState and run prediction
                state = MatchState(**{
                    k: match_data[k]
                    for k in MatchState.model_fields
                })
                features = _build_features(state)
                probabilities = model.predict_proba(features)[0]

                win_prob  = round(float(probabilities[1]), 4)
                lose_prob = round(float(probabilities[0]), 4)

                overs_bowled = state.balls_bowled // 6
                balls_in_over = state.balls_bowled % 6
                overs_str = f"{overs_bowled}.{balls_in_over}"

                summary = (
                    f"{state.batting_team} need "
                    f"{state.target_score - state.current_score} runs from "
                    f"{120 - state.balls_bowled} balls "
                    f"({state.current_score}/{state.wickets_fallen} after "
                    f"{overs_str} overs) vs {state.bowling_team} at {state.venue}"
                )

                live_cache = {
                    "win_probability": win_prob,
                    "lose_probability": lose_prob,
                    "match_state_summary": summary,
                    "is_live": True,
                    "last_updated": match_data.get(
                        "last_updated",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    # Extra fields for the dashboard
                    "batting_team": state.batting_team,
                    "bowling_team": state.bowling_team,
                    "venue": state.venue,
                    "target_score": state.target_score,
                    "current_score": state.current_score,
                    "wickets_fallen": state.wickets_fallen,
                    "balls_bowled": state.balls_bowled,
                }
                logger.info(
                    "Live update: %s %d/%d → %.1f%%",
                    state.batting_team,
                    state.current_score,
                    state.wickets_fallen,
                    win_prob * 100,
                )
            else:
                live_cache = None

        except Exception as exc:
            logger.error("Live poll error: %s", exc)

        await asyncio.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Helper — build feature vector
# ---------------------------------------------------------------------------
def _build_features(state: MatchState) -> np.ndarray:
    """
    Transform a MatchState into the 10-feature vector the model expects:
        runs_remaining, balls_remaining, wickets_remaining,
        required_run_rate, current_run_rate, run_rate_diff,
        toss_advantage, venue_encoded, batting_team_form, head_to_head_ratio
    """
    runs_remaining   = state.target_score - state.current_score
    balls_remaining  = 120 - state.balls_bowled
    wickets_remaining = 10 - state.wickets_fallen

    overs_bowled     = state.balls_bowled / 6.0
    overs_remaining  = balls_remaining / 6.0

    current_run_rate = (
        state.current_score / overs_bowled if overs_bowled > 0 else 0.0
    )
    required_run_rate = (
        runs_remaining / overs_remaining if overs_remaining > 0 else 0.0
    )
    run_rate_diff = current_run_rate - required_run_rate

    toss_advantage = 1 if state.batting_team == state.toss_winner else 0

    # Venue encoding — unseen venues fall back to code -1
    try:
        venue_encoded = int(
            venue_encoder.transform([state.venue])[0]
        )
    except (ValueError, KeyError):
        venue_encoded = -1

    features = np.array(
        [[
            runs_remaining,
            balls_remaining,
            wickets_remaining,
            required_run_rate,
            current_run_rate,
            run_rate_diff,
            toss_advantage,
            venue_encoded,
            state.batting_team_form,
            state.head_to_head_ratio,
        ]],
        dtype=np.float64,
    )
    return features


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_match_state(state: MatchState) -> None:
    """Raise HTTPException if business-logic constraints are violated."""
    errors = []

    if state.current_score >= state.target_score:
        errors.append(
            f"current_score ({state.current_score}) must be less than "
            f"target_score ({state.target_score}) — chase is already won"
        )

    if state.balls_bowled >= 120 and state.wickets_fallen < 10:
        errors.append("balls_bowled is 120 (innings over) with wickets remaining")

    if state.wickets_fallen >= 10 and state.balls_bowled < 120:
        errors.append("All 10 wickets fallen — innings is over")

    if state.batting_team == state.bowling_team:
        errors.append("batting_team and bowling_team cannot be the same")

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "error",
                "message": "Input validation failed",
                "errors": errors,
            },
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API docs — useful for deployment health checks."""
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictionResponse)
def predict(state: MatchState):
    """Return win/lose probabilities for the current match state."""
    _validate_match_state(state)

    try:
        features = _build_features(state)
        probabilities = model.predict_proba(features)[0]
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Prediction failed: {exc}",
            },
        )

    win_prob  = round(float(probabilities[1]), 4)
    lose_prob = round(float(probabilities[0]), 4)

    overs_bowled = state.balls_bowled // 6
    balls_in_over = state.balls_bowled % 6
    overs_str = f"{overs_bowled}.{balls_in_over}"

    summary = (
        f"{state.batting_team} need "
        f"{state.target_score - state.current_score} runs from "
        f"{120 - state.balls_bowled} balls "
        f"({state.current_score}/{state.wickets_fallen} after {overs_str} overs) "
        f"vs {state.bowling_team} at {state.venue}"
    )

    return PredictionResponse(
        status="success",
        win_probability=win_prob,
        lose_probability=lose_prob,
        match_state_summary=summary,
    )


@app.get("/live")
def live():
    """Return the latest cached live-match prediction, or 404 if none."""
    if live_cache is None:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": "No live IPL match currently in progress",
            },
        )
    return {"status": "success", "data": live_cache}


@app.get("/health")
def health():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "live_match_active": live_cache is not None,
    }


@app.get("/teams")
def teams():
    """Return the list of IPL teams the model was trained on."""
    ipl_teams = [
        "Chennai Super Kings",
        "Delhi Capitals",
        "Gujarat Titans",
        "Kolkata Knight Riders",
        "Lucknow Super Giants",
        "Mumbai Indians",
        "Punjab Kings",
        "Rajasthan Royals",
        "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    ]
    return {"status": "success", "teams": ipl_teams}


@app.get("/venues")
def venues():
    """Return the list of venues the model knows about."""
    try:
        known_venues = venue_encoder.classes_.tolist()
    except Exception as exc:
        logger.warning("Could not retrieve venues: %s", exc)
        known_venues = []
    return {"status": "success", "venues": known_venues}
