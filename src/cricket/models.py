"""Provider-independent match record used by the Live Cricket page."""

from dataclasses import asdict, dataclass, field


@dataclass
class CricketMatch:
    id: str
    source: str                       # "cricketdata" | "cricsheet"
    state: str                        # "live" | "upcoming" | "result"
    name: str
    teams: list[str]
    groups: list[str]                 # subset of india_men / india_women / india_a ([] for IPL clubs)
    format_base: str                  # IPL | T20I | ODI | TEST | OTHER
    format_label: str                 # human label, e.g. "T20I" or "ODI · India A"
    series: str = ""
    venue: str | None = None
    date: str | None = None           # local match date YYYY-MM-DD (may be the only time information)
    start_utc: str | None = None      # ISO timestamp when the provider gives one
    status_text: str = ""
    result_text: str | None = None
    scores: list[dict] = field(default_factory=list)   # [{team, runs, wickets, overs, innings}]
    gender: str | None = None
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CricketMatch":
        return cls(**d)
