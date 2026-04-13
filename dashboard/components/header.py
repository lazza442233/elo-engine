"""Header stats bar component."""

from __future__ import annotations

import streamlit as st

from engine.elo import GrassrootsEloEngine
from config.teams import team_short
from models.team import Team


def _biggest_swing(engine: GrassrootsEloEngine) -> dict | None:
    """Return details of the match with the largest Elo exchange in the most recent round."""
    ml = engine.match_log
    hist = engine.elo_history
    if len(ml) < 2 or len(hist) < 2:
        return None

    last_round = ml[-1].get("round")
    if not last_round:
        return None

    last_round_start = len(ml) - 1
    while last_round_start > 0 and ml[last_round_start - 1].get("round") == last_round:
        last_round_start -= 1

    if last_round_start == 0:
        return None

    best_swing = 0.0
    best_match = None
    before = hist[last_round_start - 1]
    for i in range(last_round_start, len(ml)):
        after_snap = hist[i]
        home, away = ml[i]["home"], ml[i]["away"]
        home_delta = after_snap.get(home, 0) - before.get(home, 0)
        swing = abs(home_delta)
        if swing > best_swing:
            best_swing = swing
            hs, as_ = ml[i]["home_score"], ml[i]["away_score"]
            best_match = {
                "home": home, "away": away,
                "home_score": hs, "away_score": as_,
                "swing": swing,
                "winner_delta": home_delta if hs > as_ else -home_delta,
            }
        before = {
            **before,
            home: after_snap.get(home, before.get(home, 0)),
            away: after_snap.get(away, before.get(away, 0)),
        }

    return best_match


def _closest_upcoming(
    engine: GrassrootsEloEngine,
    raw_fixtures: list[dict],
) -> dict | None:
    """Find the tightest predicted matchup from upcoming fixtures."""
    if not raw_fixtures:
        return None
    best = None
    best_gap = 2.0
    for fix in raw_fixtures:
        attrs = fix["attributes"]
        if attrs.get("bye_flag"):
            continue
        home = GrassrootsEloEngine._shorten_name(attrs["home_team_name"])
        away = GrassrootsEloEngine._shorten_name(attrs["away_team_name"])
        pred = engine.predict_match(home, away)
        gap = abs(pred["home_win"] - pred["away_win"])
        if gap < best_gap:
            best_gap = gap
            best = {"home": home, "away": away, "draw_pct": pred["draw"]}
    return best


def render_header(
    engine: GrassrootsEloEngine,
    league_table: list[Team],
    raw_fixtures: list[dict],
    detected_round: int,
    league_name: str,
) -> None:
    """Render the page title and compact stats header bar."""
    st.title(league_name)

    leader = league_table[0] if league_table else None
    total_goals = sum(t.gf for t in league_table)
    avg_gpg = total_goals / max(engine.processed_matches, 1)

    swing = _biggest_swing(engine)
    closest = _closest_upcoming(engine, raw_fixtures)

    # Inline style tokens
    cell = "display:flex; flex-direction:column; gap:1px"
    label = (
        "font-size:0.65rem; color:#94a3b8; text-transform:uppercase; "
        "letter-spacing:0.5px; font-weight:600"
    )
    value = "font-size:0.88rem; font-weight:600; line-height:1.3"

    # Responsive CSS for the header grid
    hdr = '''<style>
    .hdr-grid {
        display:grid; grid-template-columns:repeat(5, 1fr);
        gap:0 24px; padding:12px 0 16px;
        border-bottom:1px solid #e2e8f0; margin-bottom:8px;
    }
    .hdr-grid .hdr-cell { display:flex; flex-direction:column; gap:1px; }
    .hdr-grid .hdr-secondary { }
    .hdr-grid .hdr-name-full { display: inline; }
    .hdr-grid .hdr-name-short { display: none; }
    @media (max-width: 640px) {
        .hdr-grid {
            grid-template-columns: 1fr 1fr;
            gap: 12px 16px;
            padding: 8px 0 12px;
        }
        .hdr-grid .hdr-secondary { display: none; }
        .hdr-grid .hdr-name-full { display: none; }
        .hdr-grid .hdr-name-short { display: inline; }
    }
    </style>
    <div class="hdr-grid">'''

    # Primary KPIs (always visible on mobile): Leader + Closest matchup
    if leader:
        leader_short = team_short(leader.name)
        hdr += f'''<div class="hdr-cell">
    <span style="{label}">Leader</span>
    <span style="{value}"><span class="hdr-name-full">{leader.name}</span><span class="hdr-name-short">{leader_short}</span></span>
    <span style="font-size:0.72rem; color:#64748b">{leader.points} pts &middot; {leader.elo:.0f} Elo</span>
</div>'''

    if closest:
        home_short = team_short(closest["home"])
        away_short = team_short(closest["away"])
        hdr += f'''<div class="hdr-cell">
    <span style="{label}">Closest matchup Rd {detected_round}</span>
    <span style="{value}"><span class="hdr-name-full">{closest["home"]} vs {closest["away"]}</span><span class="hdr-name-short">{home_short} vs {away_short}</span></span>
    <span style="font-size:0.72rem; color:#64748b">{closest["draw_pct"]*100:.0f}% draw probability</span>
</div>'''

    # Secondary KPIs (hidden on mobile)
    hdr += f'''<div class="hdr-cell hdr-secondary">
    <span style="{label}">Next round</span>
    <span style="{value}">Round {detected_round}</span>
</div>'''

    if swing:
        hdr += f'''<div class="hdr-cell hdr-secondary">
    <span style="{label}">Biggest swing last round</span>
    <span style="{value}">{swing["home"]} {swing["home_score"]}&ndash;{swing["away_score"]} {swing["away"]}</span>
    <span style="font-size:0.72rem; color:#64748b">&plusmn;{swing["swing"]:.0f} Elo exchanged</span>
</div>'''

    hdr += f'''<div class="hdr-cell hdr-secondary">
    <span style="{label}">Goals per game</span>
    <span style="{value}">{avg_gpg:.1f}</span>
</div>'''

    hdr += "</div>"
    st.html(hdr)
