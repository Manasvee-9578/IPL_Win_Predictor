"""
ChaseIQ - FastAPI backend (v4)
====================================================
    uvicorn api.main:app --reload       ->  http://127.0.0.1:8000/docs

Layers
------
    api/routes/*     HTTP endpoints (thin: validation + serialisation)
    api/state.py     objects shared between requests
    src/predictor.py model + ratings + explanations   (business logic)
    src/database.py  SQLite persistence
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routes import analytics, cricket, live, matches, meta, predict
from api.state import state
from src import config as C
from src import database as db
from src.cricket import cricketdata
from src.cricket.service import CricketService
from src.predictor import Predictor

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in (C.MODEL_PKL, C.DB_PATH):
        if not path.is_file():
            raise RuntimeError(f"{path.name} not found - run `python -m src.run_pipeline` first.")

    db.ensure_schema()
    state.predictor = Predictor()
    state.cricket = CricketService()
    logger.info("Model '%s' loaded; %d teams, %d venues in database",
                state.predictor.metrics.get("selected_model", "?"),
                len(state.predictor.team_rows), len(state.predictor.venue_rows))

    tasks = [asyncio.create_task(live.results_forever())]
    state.live_enabled = bool(cricketdata.configured_key())
    if state.live_enabled:
        tasks.append(asyncio.create_task(live.poll_forever()))
    else:
        logger.warning("No CricketData.org key - live scores/fixtures disabled "
                       "(results, simulator, replay and analytics still work)")

    yield

    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ChaseIQ",
    description="ChaseIQ - Inside Every Chase. Ball-by-ball win probabilities for IPL second-innings chases "
                "(IPL Second-Innings Chase Model), live cricket and analytics.",
    version="4.0.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

for module in (predict, matches, analytics, live, cricket, meta):
    app.include_router(module.router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
