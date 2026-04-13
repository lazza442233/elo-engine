"""Shared helper functions for dashboard components."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from zoneinfo import ZoneInfo


def team_form(team_name: str, match_log: list[dict], n: int = 5) -> list[dict]:
    """Return last *n* results for a team with score context, most recent first."""
    results: list[dict] = []
    for entry in reversed(match_log):
        if team_name not in (entry["home"], entry["away"]):
            continue
        hs, as_ = entry["home_score"], entry["away_score"]
        if team_name == entry["home"]:
            res = "W" if hs > as_ else ("D" if hs == as_ else "L")
            opponent = entry["away"]
            score = f"{hs}-{as_}"
        else:
            res = "W" if as_ > hs else ("D" if hs == as_ else "L")
            opponent = entry["home"]
            score = f"{as_}-{hs}"
        results.append({"result": res, "score": score, "opponent": opponent})
        if len(results) >= n:
            break
    return results


def form_dots(form: list[dict]) -> str:
    """Render form as coloured dots HTML with CSS hover tooltips."""
    colours = {"W": "#22c55e", "D": "#94a3b8", "L": "#ef4444"}
    labels = {"W": "Win", "D": "Draw", "L": "Loss"}
    dots = []
    for entry in form:
        r = entry["result"]
        c = colours[r]
        tip = escape(f'{labels[r]} {entry["score"]} vs {entry["opponent"]}')
        dots.append(
            f'<span class="dot-wrap" data-tip="{tip}">'
            f'<span style="display:inline-block; width:14px; height:14px; '
            f'border-radius:50%; background:{c}"></span></span>'
        )
    for _ in range(5 - len(dots)):
        dots.append(
            '<span style="display:inline-block; width:14px; height:14px; '
            'border-radius:50%; background:#e5e7eb; margin:0 2px"></span>'
        )
    return "".join(dots)


def confidence_label(pct: float) -> tuple[str, str]:
    """Return (label, colour) for a win probability."""
    if pct >= 0.80:
        return "Strong", "#1a7f37"
    if pct >= 0.60:
        return "Likely", "#2563eb"
    if pct >= 0.45:
        return "Lean", "#d97706"
    return "Toss-up", "#94a3b8"


def parse_fixture_datetime(raw_date: str) -> datetime:
    """Parse a Dribl fixture date string to a UTC-aware datetime.

    Handles both '%Y-%m-%dT%H:%M:%S.%fZ' and '%Y-%m-%dT%H:%M:%SZ' formats.
    Returns datetime.min (UTC) on unparseable input.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(raw_date, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def parse_fixture_dt(attrs: dict) -> str:
    """Parse fixture date to 'Sat 12 Apr  15:00 AEST' display string."""
    raw = attrs.get("date", "")
    utc = parse_fixture_datetime(raw)
    if utc == datetime.min.replace(tzinfo=timezone.utc):
        return ""
    local = utc.astimezone(ZoneInfo("Australia/Sydney"))
    return local.strftime("%a %d %b  %H:%M AEST")


def closeness_score(hw: float, aw: float) -> float:
    """0 = perfectly even, 1 = total blowout. For sorting."""
    return abs(hw - aw)
