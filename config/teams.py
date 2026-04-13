"""
Team identity configuration — colors, short names, abbreviations.

Keys must match the shortened team names produced by GrassrootsEloEngine._shorten_name().
Fill in primary/secondary hex colors and abbreviations for each team.
"""

import base64
import os
from functools import lru_cache
from html import escape as _esc

TEAMS = {
    "Castle Hill United Football Club": {
        "short": "Castle Hill",
        "abbr": "CHU",
        "primary": "#0AA265",
        "secondary": "#FFFFFF",
        "badge": "assets/badges/castle_hill.png",
    },
    "Eastwood St Andrews FC": {
        "short": "Eastwood",
        "abbr": "STA",
        "primary": "#0057B8",
        "secondary": "#FFD700",
        "badge": "assets/badges/eastwood_sa.png",
    },
    "Epping Eastwood FC": {
        "short": "Epping Eastwood",
        "abbr": "EEW",
        "primary": "#F5A623",
        "secondary": "#000000",
        "badge": "assets/badges/epping_eastwood.png",
    },
    "Epping FC": {
        "short": "Epping",
        "abbr": "EPP",
        "primary": "#2B4CC4",
        "secondary": "#FFFFFF",
        "badge": "assets/badges/epping.png",
    },
    "Gladesville Ravens SC": {
        "short": "Gladesville",
        "abbr": "GVR",
        "primary": "#000000",
        "secondary": "#FFFFFF",
        "badge": "assets/badges/gladesville.png",
    },
    "Homenetmen Antranig FC": {
        "short": "Homenetmen",
        "abbr": "HOM",
        "primary": "#3F3A8C",
        "secondary": "#E5692C",
        "badge": "assets/badges/homenetmen.png",
    },
    "Kellyville Kolts Soccer Club": {
        "short": "Kellyville Kolts",
        "abbr": "KVK",
        "primary": "#D50032",
        "secondary": "#000000",
        "badge": "assets/badges/kellyville.png",
    },
    "Macquarie Dragons FC": {
        "short": "Macquarie Dragons",
        "abbr": "MAC",
        "primary": "#C41E3A",
        "secondary": "#FFFFFF",
        "badge": "assets/badges/macquarie.png",
    },
    "Pennant Hills FC": {
        "short": "Pennant Hills",
        "abbr": "PEN",
        "primary": "#1B5E20",
        "secondary": "#FFD600",
        "badge": "assets/badges/pennant_hills.png",
    },
    "Putney Rangers FC": {
        "short": "Putney Rangers",
        "abbr": "PUT",
        "primary": "#D50000",
        "secondary": "#000000",
        "badge": "assets/badges/putney.png",
    },
    "West Pennant Hills Cherrybrook FC": {
        "short": "WPH Cherrybrook",
        "abbr": "WPH",
        "primary": "#0033A0",
        "secondary": "#E4002B",
        "badge": "assets/badges/wph_cherrybrook.png",
    },
    "West Ryde Rovers FC": {
        "short": "West Ryde",
        "abbr": "WRR",
        "primary": "#1B2240",
        "secondary": "#FFFFFF",
        "badge": "assets/badges/west_ryde.png",
    },
}


_warned_teams: set[str] = set()


def _warn_unknown(name: str) -> None:
    """Log a one-time warning when a team has no config entry."""
    if name not in _warned_teams:
        _warned_teams.add(name)
        import sys
        print(f"[teams] Unknown team: {name!r} — using defaults. "
              f"Add to config/teams.py for badge/colour support.",
              file=sys.stderr)


def team_color(name: str) -> str:
    """Return the primary color for a team, or a default grey."""
    cfg = TEAMS.get(name)
    if not cfg:
        _warn_unknown(name)
    return cfg["primary"] if cfg else "#64748b"


def team_short(name: str) -> str:
    """Return the short display name for a team."""
    cfg = TEAMS.get(name)
    return cfg["short"] if cfg else name


def team_abbr(name: str) -> str:
    """Return the 3-letter abbreviation for a team."""
    cfg = TEAMS.get(name)
    return cfg["abbr"] if cfg else name[:3].upper()


def team_badge(name: str) -> str:
    """Return the badge image path/URL for a team, or empty string."""
    cfg = TEAMS.get(name)
    return cfg["badge"] if cfg else ""


@lru_cache(maxsize=64)
def _badge_data_uri(path: str) -> str:
    """Read a PNG badge and return a data:image URI, or empty string."""
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _fallback_circle(name: str, size: int) -> str:
    """SVG circle with the team abbreviation as a fallback badge."""
    color = team_color(name)
    abbr = _esc(team_abbr(name)[:2])
    fs = max(size * 0.42, 8)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{size // 2}" fill="{color}"/>'
        f'<text x="50%" y="50%" dominant-baseline="central" text-anchor="middle" '
        f'fill="#fff" font-size="{fs:.0f}" font-weight="700" font-family="sans-serif">'
        f'{abbr}</text></svg>'
    )


def team_badge_html(name: str, size: int = 20) -> str:
    """Return an <img> tag for the team badge, or an SVG fallback circle."""
    badge_path = team_badge(name)
    uri = _badge_data_uri(badge_path)
    if uri:
        return (
            f'<img src="{uri}" width="{size}" height="{size}" '
            f'style="vertical-align:middle; border-radius:2px; object-fit:contain" '
            f'alt="{_esc(team_abbr(name))}">'
        )
    svg = _fallback_circle(name, size)
    b64 = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return (
        f'<img src="{b64}" width="{size}" height="{size}" '
        f'style="vertical-align:middle" alt="{_esc(team_abbr(name))}">'
    )


def team_badge_data_uri(name: str) -> str:
    """Return a data URI for the badge (PNG or SVG fallback). Used by Altair."""
    badge_path = team_badge(name)
    uri = _badge_data_uri(badge_path)
    if uri:
        return uri
    svg = _fallback_circle(name, 16)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
