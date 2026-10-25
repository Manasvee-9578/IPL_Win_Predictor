"""
SQLite persistence layer.

Why SQLite?  Zero-install, a single portable file (``database/ipl.db``) that
ships with the repo, and full SQL - ideal for a project that must run on any
examiner's laptop *and* on a free cloud host.

Entity-relationship overview
----------------------------
    teams ──< matches >── venues
      │          │
      │          └──< chase_states        (win probability after every ball)
      └──< head_to_head
    match_context       (pre-match ratings of each historical chase, for what-if re-scoring)
    external_matches    (cache of live / fixture / result data from free cricket providers)
    api_usage           (daily hit counter for free-tier quotas)
    prediction_log      (every scenario evaluated through the API)
    live_snapshots      (history of the live match polled from CricAPI)
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,
    abbr      TEXT NOT NULL,
    color     TEXT NOT NULL,
    active    INTEGER NOT NULL,
    elo       REAL NOT NULL,
    form      REAL NOT NULL,
    matches   INTEGER NOT NULL DEFAULT 0,
    wins      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL UNIQUE,
    city               TEXT,
    matches            INTEGER NOT NULL DEFAULT 0,
    avg_first_innings  REAL NOT NULL,
    chase_bias         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,
    season          INTEGER NOT NULL,
    match_date      TEXT NOT NULL,
    venue_id        INTEGER NOT NULL REFERENCES venues(venue_id),
    team1           INTEGER NOT NULL REFERENCES teams(team_id),
    team2           INTEGER NOT NULL REFERENCES teams(team_id),
    toss_winner     INTEGER REFERENCES teams(team_id),
    toss_decision   TEXT,
    winner          INTEGER REFERENCES teams(team_id),
    result          TEXT,
    first_team      INTEGER REFERENCES teams(team_id),
    first_runs      INTEGER,
    chase_team      INTEGER REFERENCES teams(team_id),
    chase_runs      INTEGER,
    chase_wickets   INTEGER,
    chase_balls     INTEGER,
    win_by_runs     REAL,
    win_by_wickets  REAL,
    valid_chase     INTEGER NOT NULL,
    split           TEXT NOT NULL            -- 'train' (<=2022) or 'test' (2023+)
);
CREATE INDEX IF NOT EXISTS ix_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS ix_matches_venue  ON matches(venue_id);

CREATE TABLE IF NOT EXISTS chase_states (
    match_id  INTEGER NOT NULL REFERENCES matches(match_id),
    balls     INTEGER NOT NULL,               -- legal balls bowled in the chase
    runs      INTEGER NOT NULL,
    wickets   INTEGER NOT NULL,
    win_prob  REAL NOT NULL,                  -- chasing team's win probability
    PRIMARY KEY (match_id, balls)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS head_to_head (
    team_a  INTEGER NOT NULL REFERENCES teams(team_id),
    team_b  INTEGER NOT NULL REFERENCES teams(team_id),
    a_wins  INTEGER NOT NULL,
    games   INTEGER NOT NULL,
    PRIMARY KEY (team_a, team_b)
);

-- pre-match context of every historical chase (needed to re-score hypothetical "what if" balls)
CREATE TABLE IF NOT EXISTS match_context (
    match_id   INTEGER PRIMARY KEY REFERENCES matches(match_id),
    target     INTEGER NOT NULL,
    elo_diff   REAL NOT NULL,
    form       REAL NOT NULL,
    h2h        REAL NOT NULL,
    venue_avg  REAL NOT NULL,
    venue_bias REAL NOT NULL,
    toss_adv   INTEGER NOT NULL
);

-- last good copy of every match fetched from an external cricket provider (survives restarts,
-- so the API never has to burn free-tier quota just to refill its cache)
CREATE TABLE IF NOT EXISTS external_matches (
    provider    TEXT NOT NULL,
    match_key   TEXT NOT NULL,
    state       TEXT NOT NULL,                -- live | upcoming | result
    start_utc   TEXT,
    payload     TEXT NOT NULL,                -- normalised JSON
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (provider, match_key)
);

-- free-tier quota bookkeeping (CricketData.org allows 100 hits / day)
CREATE TABLE IF NOT EXISTS api_usage (
    day       TEXT NOT NULL,
    provider  TEXT NOT NULL,
    hits      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, provider)
);

CREATE TABLE IF NOT EXISTS prediction_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    source           TEXT NOT NULL,
    batting_team     TEXT NOT NULL,
    bowling_team     TEXT NOT NULL,
    venue            TEXT NOT NULL,
    target           INTEGER NOT NULL,
    score            INTEGER NOT NULL,
    wickets          INTEGER NOT NULL,
    balls            INTEGER NOT NULL,
    win_probability  REAL NOT NULL,
    latency_ms       REAL
);

CREATE TABLE IF NOT EXISTS live_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    batting_team     TEXT NOT NULL,
    bowling_team     TEXT NOT NULL,
    venue            TEXT,
    target           INTEGER NOT NULL,
    score            INTEGER NOT NULL,
    wickets          INTEGER NOT NULL,
    balls            INTEGER NOT NULL,
    win_probability  REAL NOT NULL
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or C.DB_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | str | None = None):
    """Short-lived connection: commit on success, always close."""
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def ensure_schema(path: Path | str | None = None) -> None:
    """Create any missing tables in an existing database (safe to call on every start)."""
    with session(path) as conn:
        init_schema(conn)


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
