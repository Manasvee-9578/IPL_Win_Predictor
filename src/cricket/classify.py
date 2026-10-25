"""Which side is which: India Men / India Women / India A, and which competition format."""

import re

from src.reference import team_id

GROUPS = {"india_men": "India Men", "india_women": "India Women", "india_a": "India A"}
FORMATS = ["T20I", "ODI", "TEST", "IPL"]

_INDIA = re.compile(r"(?<![a-z])india(?![a-z])", re.I)          # matches "India", not "Indies"/"Indian"
_INDIA_A = re.compile(r"(?<![a-z])india[\s-]+(a|emerging)(?![a-z])", re.I)
_YOUTH = re.compile(r"under[\s-]*\d+|(?<![a-z])u[\s-]?\d{2}(?![a-z])", re.I)


def team_group(name: str, gender: str | None = None) -> str | None:
    """Classify a team name; None for anyone who is not an India senior/A side."""
    if not name or not _INDIA.search(name):
        return None
    if _YOUTH.search(name):
        return None
    if _INDIA_A.search(name):
        return "india_a"
    if "women" in name.lower() or (gender or "").lower() in ("female", "women"):
        return "india_women"
    return "india_men"


def groups_for(teams: list[str], gender: str | None = None) -> list[str]:
    seen = []
    for t in teams:
        g = team_group(t, gender)
        if g and g not in seen:
            seen.append(g)
    return seen


def is_ipl(series: str = "", name: str = "", teams: list[str] | None = None) -> bool:
    """IPL = the series says so, or both sides are IPL franchises."""
    text = f"{series} {name}".lower()
    if "indian premier league" in text or re.search(r"(?<![a-z])ipl(?![a-z])", text):
        return True
    teams = teams or []
    return len(teams) == 2 and all(team_id(t) is not None for t in teams)


def format_for(match_type: str, groups: list[str], ipl: bool) -> tuple[str, str]:
    """(format_base, label). T20/ODI/Test for India national sides are the international formats."""
    if ipl:
        return "IPL", "IPL"
    mt = (match_type or "").lower()
    base = {"t20": "T20I", "t20i": "T20I", "odi": "ODI", "odm": "ODI", "test": "TEST", "mdm": "TEST"}.get(mt)
    if base is None:
        return "OTHER", (match_type or "Match").upper()
    if "india_a" in groups:
        return base, {"T20I": "T20 · India A", "ODI": "ODI · India A", "TEST": "Multi-day · India A"}[base]
    return base, {"T20I": "T20I", "ODI": "ODI", "TEST": "Test"}[base]


def wanted(match_groups: list[str], format_base: str, groups: set[str], formats: set[str]) -> bool:
    """Filter rule: an IPL game shows when IPL is selected; otherwise both the India side and
    the format must be selected."""
    if format_base == "IPL":
        return "IPL" in formats
    return bool(set(match_groups) & groups) and format_base in formats
