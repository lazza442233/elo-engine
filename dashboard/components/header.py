"""Header stats bar component."""

from __future__ import annotations

import streamlit as st

from engine.elo import GrassrootsEloEngine
from config.teams import team_short
from models.team import Team


def _biggest_swing(engine: GrassrootsEloEngine, target_round: str | None = None) -> dict | None:
    """Return details of the biggest upset or largest Elo exchange in a given round.

    If *target_round* is supplied (e.g. ``"Round 5"``), only matches from that
    round are considered.  Otherwise falls back to the most recent round label
    found in the match log.

    An upset is when the lower-rated team wins. When an upset occurred,
    that match is returned with ``is_upset=True``. Otherwise falls back to
    the match with the largest raw Elo exchange.
    """
    ml = engine.match_log
    hist = engine.elo_history
    if len(ml) < 2 or len(hist) < 2:
        return None

    if target_round is None:
        target_round = ml[-1].get("round")
    if not target_round:
        return None

    # Collect indices for matches in the target round
    round_indices = [i for i, m in enumerate(ml) if m.get("round") == target_round]
    if not round_indices or round_indices[0] == 0:
        return None

    best_upset_swing = 0.0
    best_upset: dict | None = None
    best_swing = 0.0
    best_match: dict | None = None

    for i in round_indices:
        # Use the snapshot *before* this match as the baseline
        before = hist[i - 1]
        after_snap = hist[i]
        home, away = ml[i]["home"], ml[i]["away"]
        home_delta = after_snap.get(home, 0) - before.get(home, 0)
        swing = abs(home_delta)
        hs, as_ = ml[i]["home_score"], ml[i]["away_score"]

        # Determine pre-match favourite (home gets HFA implicitly in Elo)
        home_pre = before.get(home, 1500)
        away_pre = before.get(away, 1500)
        home_favoured = home_pre >= away_pre

        # Was this an upset? (lower-rated team won)
        is_upset = False
        if hs > as_ and not home_favoured:
            is_upset = True
        elif as_ > hs and home_favoured:
            is_upset = True

        match_info = {
            "home": home, "away": away,
            "home_score": hs, "away_score": as_,
            "swing": swing,
            "is_upset": is_upset,
        }

        if is_upset and swing > best_upset_swing:
            best_upset_swing = swing
            best_upset = match_info

        if swing > best_swing:
            best_swing = swing
            best_match = match_info

    # Prefer upset; fall back to biggest swing
    return best_upset if best_upset else best_match


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
    st.markdown(
        f'<h2 style="margin:0 0 4px; font-size:1.4rem; font-weight:700; line-height:1.3">{league_name}</h2>',
        unsafe_allow_html=True,
    )

    leader = league_table[0] if league_table else None
    total_goals = sum(t.gf for t in league_table)
    avg_gpg = total_goals / max(engine.processed_matches, 1)

    # Use the most recently completed full round, not the last-processed match
    last_completed_round = max(detected_round - 1, 1)
    swing = _biggest_swing(engine, target_round=f"Round {last_completed_round}")
    swing_round_label = last_completed_round
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
        if swing.get("is_upset"):
            swing_label = f"Shock result Rd {swing_round_label}"
            swing_sub = f'&plusmn;{swing["swing"]:.0f} Elo exchanged'
        else:
            swing_label = f"Biggest swing Rd {swing_round_label}"
            swing_sub = f'&plusmn;{swing["swing"]:.0f} Elo exchanged'
        home_short_sw = team_short(swing["home"])
        away_short_sw = team_short(swing["away"])
        hdr += f'''<div class="hdr-cell hdr-secondary">
    <span style="{label}">{swing_label}</span>
    <span style="{value}"><span class="hdr-name-full">{swing["home"]} {swing["home_score"]}&ndash;{swing["away_score"]} {swing["away"]}</span><span class="hdr-name-short">{home_short_sw} {swing["home_score"]}&ndash;{swing["away_score"]} {away_short_sw}</span></span>
    <span style="font-size:0.72rem; color:#64748b">{swing_sub}</span>
</div>'''

    hdr += f'''<div class="hdr-cell hdr-secondary">
    <span style="{label}">Goals per game</span>
    <span style="{value}">{avg_gpg:.1f}</span>
</div>'''

    hdr += "</div>"
    st.html(hdr)
