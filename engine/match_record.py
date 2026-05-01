"""Canonical normalized match records for engine replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

SKIP_STATUSES = {"forfeit", "abandoned", "postponed", "upcoming", "bye"}
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
)


def shorten_team_name(raw: str) -> str:
    """Strip the league/grade suffix that Dribl appends to team names."""
    suffixes = [
        " Premier League Men First Grade",
        " Premier League Men Reserve Grade",
        " Premier League First Grade",
        " Premier League Reserve Grade",
        " Premier League Men",
        " Premier League",
        " First Grade",
        " Reserve Grade",
    ]
    for suffix in suffixes:
        if raw.endswith(suffix):
            return raw[: -len(suffix)].strip()
    return raw.strip()


def parse_match_datetime(raw: str | datetime | None) -> datetime:
    """Parse a match date string, falling back safely when absent."""
    if not raw:
        return datetime.min

    if isinstance(raw, datetime):
        return raw

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    return datetime.min


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """Normalized match record used by engine replay and backtests."""

    home_team: str
    away_team: str
    home_score: int
    away_score: int
    match_date: str | None = None
    kickoff: datetime = datetime.min
    round_label: str | None = None
    round_number: int | None = None
    season: str | None = None
    grade: str | None = None
    status: str = "played"
    source: str = "unknown"

    @classmethod
    def from_api_dict(cls, match: dict[str, Any]) -> MatchRecord | None:
        attrs = match.get("attributes", {})
        if attrs.get("bye_flag", 0):
            return None

        status = attrs.get("status", "played").lower()
        if status in SKIP_STATUSES:
            return None

        home_score = attrs.get("home_score")
        away_score = attrs.get("away_score")
        if home_score is None or away_score is None:
            return None

        match_date = attrs.get("date")
        return cls(
            home_team=shorten_team_name(attrs["home_team_name"]),
            away_team=shorten_team_name(attrs["away_team_name"]),
            home_score=int(home_score),
            away_score=int(away_score),
            match_date=match_date,
            kickoff=parse_match_datetime(match_date),
            round_label=attrs.get("full_round"),
            status=status,
            source="api",
        )

    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> MatchRecord | None:
        status = str(row.get("status", "complete")).lower()
        if status != "complete":
            return None

        home_score = row.get("home_goals")
        away_score = row.get("away_goals")
        if home_score in (None, "") or away_score in (None, ""):
            return None

        match_date = row.get("match_date")
        round_number = row.get("round")
        return cls(
            home_team=str(row["home_team_id"]),
            away_team=str(row["away_team_id"]),
            home_score=int(home_score),
            away_score=int(away_score),
            match_date=match_date,
            kickoff=parse_match_datetime(match_date),
            round_label=row.get("full_round"),
            round_number=int(round_number) if round_number not in (None, "") else None,
            season=row.get("season"),
            grade=row.get("grade"),
            status=status,
            source="csv",
        )

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> MatchRecord | None:
        status = str(row.get("status", "played")).lower()
        if status in SKIP_STATUSES:
            return None

        home_score = row.get("home_score")
        away_score = row.get("away_score")
        if home_score is None or away_score is None:
            return None

        match_date = row.get("match_date")
        return cls(
            home_team=shorten_team_name(str(row["home_team"])),
            away_team=shorten_team_name(str(row["away_team"])),
            home_score=int(home_score),
            away_score=int(away_score),
            match_date=match_date,
            kickoff=parse_match_datetime(match_date),
            round_label=row.get("round_label"),
            status=status,
            source="db",
        )


def normalize_match_records(raw_items: Iterable[MatchRecord | dict[str, Any]]) -> tuple[list[MatchRecord], int]:
    """Coerce mixed raw inputs into sorted, normalized match records."""
    records: list[MatchRecord] = []
    skipped = 0

    for item in raw_items:
        record: MatchRecord | None
        if isinstance(item, MatchRecord):
            record = item
        elif isinstance(item, dict) and "attributes" in item:
            record = MatchRecord.from_api_dict(item)
        elif isinstance(item, dict) and "home_team_id" in item:
            record = MatchRecord.from_csv_row(item)
        elif isinstance(item, dict) and "home_team" in item:
            record = MatchRecord.from_db_row(item)
        else:
            raise TypeError(f"Unsupported match input: {type(item)!r}")

        if record is None:
            skipped += 1
            continue

        records.append(record)

    records.sort(key=lambda record: (record.kickoff, record.match_date or "", record.home_team, record.away_team))
    return records, skipped
