"""Thin HTTP client for the FastAPI backend. Every call returns (data, error)."""

import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("API_TIMEOUT", "20"))     # generous: free hosts cold-start slowly

_session = requests.Session()


class _ApiError(Exception):
    """Raised inside cached functions so that st.cache_data does not store a failure."""


def cached(ttl: int):
    """Like st.cache_data(ttl) for functions returning (data, error), but errors are never cached:
    a request that fails while the API is still starting must not stay 'offline' for minutes."""
    def deco(fn):
        # `name` is part of the cache key: every endpoint shares this inner function's code,
        # so without it teams() / venues() / health() would collide on the same empty-args key.
        @st.cache_data(ttl=ttl, show_spinner=False)
        def inner(name, *args):
            data, err = fn(*args)
            if err:
                raise _ApiError(err)
            return data

        def wrapper(*args):
            try:
                return inner(fn.__name__, *args), None
            except _ApiError as exc:
                return None, str(exc)
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


def _call(method: str, path: str, *, params=None, body=None):
    try:
        resp = _session.request(method, f"{API_BASE}{path}", params=params, json=body,
                                timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_BASE} ({exc.__class__.__name__})"
    if resp.status_code == 200:
        return resp.json(), None
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("errors") or detail
    if isinstance(detail, list):                       # FastAPI validation errors
        detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}"
                           for d in detail)
    return None, f"{resp.status_code}: {detail}"


# ---- cached reference data ---------------------------------------------------
@cached(20)
def health():
    return _call("GET", "/health")


@cached(600)
def teams():
    return _call("GET", "/teams")


@cached(600)
def venues():
    return _call("GET", "/venues")


@cached(600)
def model_info():
    return _call("GET", "/model/info")


@cached(600)
def analytics(name: str, params_json: str = "{}"):
    return _call("GET", f"/analytics/{name}", params=json.loads(params_json))


@cached(600)
def seasons():
    return _call("GET", "/matches/seasons")


@cached(600)
def matches(season: int | None, team: str | None):
    params = {"limit": 300}
    if season:
        params["season"] = season
    if team:
        params["team"] = team
    return _call("GET", "/matches", params=params)


@cached(3600)
def replay(match_id: int):
    return _call("GET", f"/matches/{match_id}/replay")


# ---- prediction calls (short cache so reruns of the same state are instant) ------
@cached(60)
def predict(payload_json: str, explain: bool = True):
    return _call("POST", "/predict", params={"log": "false", "explain": str(explain).lower()},
                 body=json.loads(payload_json))


@cached(60)
def whatif(payload_json: str):
    return _call("POST", "/whatif", body=json.loads(payload_json))


@cached(300)
def heatmap(payload_json: str, wickets: int):
    return _call("POST", "/heatmap", params={"wickets": wickets}, body=json.loads(payload_json))


# ---- uncached (always fresh) ----------------------------------------------------
def live():
    return _call("GET", "/live")


def live_history():
    return _call("GET", "/live/history")


# ---- live cricket (providers cache on the server; these are cheap reads) ------------
@cached(20)
def cricket(state: str, teams: str, formats: str):
    return _call("GET", "/cricket/matches", params={"state": state, "teams": teams, "formats": formats})


@cached(20)
def cricket_status():
    return _call("GET", "/cricket/status")
