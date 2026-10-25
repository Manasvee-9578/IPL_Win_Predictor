"""Reference data: IPL franchises and canonical venue names.

The raw Kaggle-style CSVs identify teams by numeric id and spell the same
ground in several ways ("Wankhede Stadium" / "Wankhede Stadium, Mumbai").
This module is the single source of truth that fixes both problems.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Team:
    team_id: int
    name: str
    abbr: str
    color: str
    active: bool
    aliases: tuple = field(default_factory=tuple)


TEAMS: list[Team] = [
    Team(129, "Chennai Super Kings", "CSK", "#f9cd05", True, ("Chennai Super Kings",)),
    Team(252, "Delhi Capitals", "DC", "#2d6fd4", True, ("Delhi Daredevils",)),
    Team(615, "Gujarat Titans", "GT", "#3b4a7a", True, ()),
    Team(6, "Kolkata Knight Riders", "KKR", "#7b4bc4", True, ()),
    Team(614, "Lucknow Super Giants", "LSG", "#00a8e0", True, ("Lucknow Super Giant",)),
    Team(3, "Mumbai Indians", "MI", "#1d6fd1", True, ()),
    Team(494, "Punjab Kings", "PBKS", "#ed1b24", True, ("Kings XI Punjab",)),
    Team(134, "Rajasthan Royals", "RR", "#ea1a85", True, ()),
    Team(1, "Royal Challengers Bengaluru", "RCB", "#d4213d", True,
         ("Royal Challengers Bangalore",)),
    Team(2, "Sunrisers Hyderabad", "SRH", "#ff822a", True, ("Deccan Chargers",)),
    # Defunct franchises (kept so historical matches stay explorable)
    Team(4, "Rising Pune Supergiant", "RPS", "#9b59b6", False,
         ("Rising Pune Supergiants",)),
    Team(5, "Gujarat Lions", "GL", "#e67e22", False, ()),
    Team(1414, "Kochi Tuskers Kerala", "KTK", "#8e44ad", False, ()),
    Team(1419, "Pune Warriors India", "PWI", "#3E6FB0", False, ("Pune Warriors",)),
]

TEAM_BY_ID = {t.team_id: t for t in TEAMS}
_NAME_TO_ID: dict[str, int] = {}
for _t in TEAMS:
    for _key in (_t.name, _t.abbr, *_t.aliases):
        _NAME_TO_ID[_key.strip().lower()] = _t.team_id


def team_id(name: str | int) -> int | None:
    """Resolve a team name / abbreviation / legacy alias / id to its numeric id."""
    if isinstance(name, int):
        return name if name in TEAM_BY_ID else None
    return _NAME_TO_ID.get(str(name).strip().lower())


def team_name(tid: int) -> str:
    return TEAM_BY_ID[tid].name


# ---------------------------------------------------------------------------
# Venues: many raw spellings -> one canonical name
# ---------------------------------------------------------------------------
_VENUE_GROUPS: dict[str, list[str]] = {
    "Arun Jaitley Stadium, Delhi": [
        "Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Brabourne Stadium, Mumbai": ["Brabourne Stadium"],
    "Dr DY Patil Sports Academy, Mumbai": ["Dr DY Patil Sports Academy"],
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": [
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium"],
    "Eden Gardens, Kolkata": ["Eden Gardens"],
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": [
        "Himachal Pradesh Cricket Association Stadium"],
    "M Chinnaswamy Stadium, Bengaluru": [
        "M Chinnaswamy Stadium", "M.Chinnaswamy Stadium"],
    "MA Chidambaram Stadium, Chennai": [
        "MA Chidambaram Stadium", "MA Chidambaram Stadium, Chepauk",
        "MA Chidambaram Stadium, Chepauk, Chennai"],
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": [
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh"],
    "Maharashtra Cricket Association Stadium, Pune": [
        "Maharashtra Cricket Association Stadium"],
    "Narendra Modi Stadium, Ahmedabad": ["Sardar Patel Stadium, Motera"],
    "Punjab Cricket Association IS Bindra Stadium, Mohali": [
        "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"],
    "Rajiv Gandhi International Stadium, Hyderabad": [
        "Rajiv Gandhi International Stadium",
        "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi International Stadium, Uppal, Hyderabad"],
    "Sawai Mansingh Stadium, Jaipur": ["Sawai Mansingh Stadium"],
    "Sheikh Zayed Stadium, Abu Dhabi": [
        "Sheikh Zayed Stadium", "Zayed Cricket Stadium, Abu Dhabi"],
    "Wankhede Stadium, Mumbai": ["Wankhede Stadium"],
}
VENUE_ALIASES = {
    alias.lower(): canon for canon, aliases in _VENUE_GROUPS.items() for alias in aliases
}


def canonical_venue(name: str) -> str:
    """Return the canonical venue name for any raw spelling."""
    name = str(name).strip()
    return VENUE_ALIASES.get(name.lower(), name)
