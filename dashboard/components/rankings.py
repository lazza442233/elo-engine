"""Rankings tab component."""

from __future__ import annotations

import streamlit as st

from dashboard.helpers import form_dots, team_form
from config.teams import team_badge_html, team_color, team_short
from models.team import Team

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

FINALS_THRESHOLD = 6

# Column count is derived from the header row (<th> elements) — see _COLUMNS.
_COLUMNS = [
    "#", "DRIFT", "Team", "Elo", "Form",
    "MP", "W", "D", "L", "GF", "GA", "GD", "Pts",
]
NUM_COLS = len(_COLUMNS)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------

_RANKINGS_CSS = """<style>
/* ── Theme tokens ─────────────────────────────────────────────── */
:root {
    --bg: #ffffff; --fg: #262730; --muted: #64748b;
    --border: #f1f5f9; --th-border: #e2e8f0;
    --bar-bg: #f1f5f9; --tip-bg: #1e293b; --tip-fg: #ffffff;
    --clr-win: #22c55e; --clr-draw: #94a3b8; --clr-loss: #ef4444;
    --drift-up-bg: #dcfce7; --drift-up-fg: #16a34a;
    --drift-dn-bg: #fee2e2; --drift-dn-fg: #dc2626;
    --drift-eq-bg: #f1f5f9; --drift-eq-fg: #475569;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0e1117; --fg: #fafafa; --muted: #94a3b8;
        --border: #1e293b; --th-border: #334155;
        --bar-bg: #1e293b; --tip-bg: #f1f5f9; --tip-fg: #0e1117;
        --drift-up-bg: #064e3b; --drift-up-fg: #4ade80;
        --drift-dn-bg: #7f1d1d; --drift-dn-fg: #f87171;
        --drift-eq-bg: #334155; --drift-eq-fg: #cbd5e1;
    }
}

/* ── Base layout ──────────────────────────────────────────────── */
body { font-family: "Source Sans Pro", "Segoe UI", sans-serif; margin:0; padding:4px 0; background:transparent; color:var(--fg); -webkit-tap-highlight-color:transparent; }
table { width:100%; border-collapse:collapse; font-size:0.88rem; table-layout:fixed; }
th   { text-align:center; padding:10px 8px; border-bottom:2px solid var(--th-border); color:var(--muted); font-weight:600; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; }
td   { padding:10px 8px; vertical-align:middle; text-align:center; }
tr   { -webkit-user-select:none; user-select:none; -webkit-tap-highlight-color:transparent; }
tr:hover  td { background: color-mix(in srgb, var(--border) 50%, transparent); }
tr:active td { background:inherit !important; }

/* ── Row semantics ────────────────────────────────────────────── */
.sep td          { border-bottom:none; }
.finals-label td { padding:2px 8px; border-bottom:2px dashed #cbd5e1; }
.finals-text     { font-size:0.68rem; color:var(--muted); font-weight:500; letter-spacing:0.3px; }

/* ── Cell classes (replaces per-cell inline styles) ───────────── */
.rank-cell   { font-weight:700; color:var(--muted); }
.drift-cell  { font-size:0.8rem; }
.team-cell   { text-align:left; font-weight:600; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.elo-cell    { font-weight:700; font-size:0.95rem; }
.form-cell   { white-space:nowrap; }
.stat-cell   { color:var(--muted); font-size:0.82rem; }
.stat-w      { color:var(--clr-win); font-size:0.82rem; }
.stat-d      { color:var(--clr-draw); font-size:0.82rem; }
.stat-l      { color:var(--clr-loss); font-size:0.82rem; }
.gd-cell     { font-weight:600; }
.gd-pos      { color:var(--clr-win); }
.gd-neg      { color:var(--clr-loss); }
.gd-zero     { color:var(--muted); }
.pts-cell    { font-weight:700; font-size:1rem; }

/* ── Drift chips ──────────────────────────────────────────────── */
.drift-chip     { display:inline-block; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; line-height:1.4; text-align:center; min-width: 20px; }
.drift-chip--up { background:var(--drift-up-bg); color:var(--drift-up-fg); }
.drift-chip--dn { background:var(--drift-dn-bg); color:var(--drift-dn-fg); }
.drift-chip--eq { background:var(--drift-eq-bg); color:var(--drift-eq-fg); }

/* ── Shared tooltip base ──────────────────────────────────────── */
.has-tip, .dot-wrap { position:relative; cursor:default; }
.dot-wrap { display:inline-block; margin:0 2px; }
.has-tip::after, .dot-wrap::after {
    content: attr(data-tip);
    position:absolute; left:50%; transform:translateX(-50%);
    background:var(--tip-bg); color:var(--tip-fg); border-radius:6px;
    font-size:0.72rem; z-index:9999; pointer-events:none;
    box-shadow:0 2px 8px rgba(0,0,0,0.2); display:none;
}
.has-tip::before, .dot-wrap::before {
    content:""; position:absolute; left:50%; transform:translateX(-50%);
    border:4px solid transparent; z-index:9999; display:none;
}
.has-tip:hover::after, .has-tip:hover::before,
.dot-wrap:hover::after, .dot-wrap:hover::before { display:block; }

/* Tooltip above (form dots) */
.tip-above::after, .dot-wrap::after  { bottom:calc(100% + 6px); padding:5px 12px; white-space:nowrap; }
.tip-above::before, .dot-wrap::before { bottom:calc(100% + 2px); border-top-color:var(--tip-bg); }

/* Tooltip below (header labels) */
.tip-below         { cursor:help; border-bottom:1px dashed var(--muted); padding-bottom:1px; }
.tip-below::after  { top:calc(100% + 8px); padding:8px 14px; white-space:pre; width:max-content; max-width:280px; text-transform:none; letter-spacing:0; font-weight:400; line-height:1.6; }
.tip-below::before { top:calc(100% + 4px); border-bottom-color:var(--tip-bg); }

/* ── Name toggle ──────────────────────────────────────────────── */
.name-short { display:none; }
.name-full  { display:inline; }
.name-short, .name-full { vertical-align:middle; }

/* ── Mobile ───────────────────────────────────────────────────── */
@media (max-width: 640px) {
    table      { font-size:0.88rem; table-layout:auto; width:100%; max-width:100%; }
    th, td     { padding:12px 4px; width:auto !important; }
    .col-secondary { display:none; }
    .name-full  { display:none; }
    .name-short { display:inline; }
    .elo-cell   { background:none !important; font-size:0.92rem !important; font-weight:600 !important; }
    .pts-cell   { font-size:0.92rem !important; font-weight:600 !important; }
    .gd-cell    { font-weight:500 !important; font-size:0.88rem !important; }
    body        { margin:0; padding:0; overflow-x:hidden; }
}
</style>"""

# -----------------------------------------------------------------------------
# Table skeleton
# -----------------------------------------------------------------------------

_DRIFT_TIP = (
    "How Elo ranks compare to the ladder.&#10;&#10;"
    "Green chip: Underrated.&#10;Elo thinks they're better than&#10;their win-loss record suggests.&#10;&#10;"
    "Red chip: Overrated.&#10;Elo thinks they're worse than&#10;their win-loss record suggests.&#10;&#10;"
    "Grey chip: Spot on.&#10;Elo and ladder agree."
)
_ELO_TIP = (
    "Elo reflects predictive strength,&#10;not just ladder points.&#10;&#10;"
    "Higher Elo = the model thinks&#10;the team is more likely to win&#10;their next match."
)
_FORM_LEGEND = (
    '<span style="font-size:0.65rem; font-weight:400; text-transform:none; '
    'letter-spacing:0; color:var(--muted)">'
    '<span style="color:var(--clr-win)">&bull;</span>&thinsp;W&ensp;'
    '<span style="color:var(--muted)">&bull;</span>&thinsp;D&ensp;'
    '<span style="color:var(--clr-loss)">&bull;</span>&thinsp;L</span>'
)

_TABLE_OPEN = f"""<table><thead><tr>
    <th style="width:28px">#</th>
    <th class="col-secondary" style="width:40px; font-size:0.65rem"><span class="has-tip tip-below" data-tip="{_DRIFT_TIP}">DRIFT</span></th>
    <th style="text-align:left">Team</th>
    <th style="width:55px"><span class="has-tip tip-below" data-tip="{_ELO_TIP}">Elo</span></th>
    <th style="width:110px">Form<br>{_FORM_LEGEND}</th>
    <th class="col-secondary" style="width:30px">MP</th>
    <th class="col-secondary" style="width:30px">W</th>
    <th class="col-secondary" style="width:30px">D</th>
    <th class="col-secondary" style="width:30px">L</th>
    <th class="col-secondary" style="width:36px">GF</th>
    <th class="col-secondary" style="width:36px">GA</th>
    <th style="width:42px">GD</th>
    <th style="width:36px">Pts</th>
</tr></thead><tbody>"""

_TABLE_CLOSE = """</tbody></table>
<script>
new ResizeObserver(() => {
    const h = document.documentElement.scrollHeight;
    window.frameElement && (window.frameElement.style.height = h + "px");
}).observe(document.body);
</script>"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _drift_chip(rank: int, elo_pos: int) -> str:
    """HTML chip showing difference between ladder rank and Elo rank."""
    diff = rank - elo_pos
    if diff > 0:
        return f'<span class="drift-chip drift-chip--up has-tip tip-above" data-tip="Underrated: Elo thinks they are {diff} spot(s) better">+{diff}</span>'
    if diff < 0:
        return f'<span class="drift-chip drift-chip--dn has-tip tip-above" data-tip="Overrated: Elo thinks they are {-diff} spot(s) worse">{diff}</span>'
    return '<span class="drift-chip drift-chip--eq has-tip tip-above" data-tip="Spot on: Elo and ladder agree">&pm;0</span>'


def _gd_html(gd: int) -> str:
    """Formatted goal-difference value (without the <td> wrapper)."""
    if gd > 0:
        return f'<span class="gd-pos">+{gd}</span>'
    if gd < 0:
        return f'<span class="gd-neg">{gd}</span>'
    return '<span class="gd-zero">0</span>'


def _elo_range(league_table: list[Team]) -> tuple[float, float]:
    """Return (floor, span) of Elo scores for bar-background scaling."""
    elos = [t.elo for t in league_table]
    floor = min(elos)
    span = max(elos) - floor
    return floor, span if span > 0 else 1.0


def _build_row(
    rank: int,
    team: Team,
    elo_floor: float,
    elo_span: float,
    elo_rank_map: dict[str, int],
    match_log: list[dict],
) -> str:
    """Build a single <tr> for the league table."""
    bar_pct = max(8.0, (team.elo - elo_floor) / elo_span * 100)
    hex_clr = team_color(team.name)
    elo_bg = f"linear-gradient(to right,{hex_clr}20 {bar_pct:.0f}%,transparent {bar_pct:.0f}%)"
    tr_cls = ' class="sep"' if rank == FINALS_THRESHOLD else ""

    return f"""<tr{tr_cls}>
        <td class="rank-cell">{rank}</td>
        <td class="col-secondary drift-cell">{_drift_chip(rank, elo_rank_map[team.name])}</td>
        <td class="team-cell">
            {team_badge_html(team.name, size=22)}
            <span class="name-full">{team.name}</span>
            <span class="name-short">{team_short(team.name)}</span>
        </td>
        <td class="elo-cell" style="background:{elo_bg}">{team.elo:.0f}</td>
        <td class="form-cell">{form_dots(team_form(team.name, match_log))}</td>
        <td class="col-secondary stat-cell">{team.played}</td>
        <td class="col-secondary stat-w">{team.wins}</td>
        <td class="col-secondary stat-d">{team.draws}</td>
        <td class="col-secondary stat-l">{team.losses}</td>
        <td class="col-secondary stat-cell">{team.gf}</td>
        <td class="col-secondary stat-cell">{team.ga}</td>
        <td class="gd-cell">{_gd_html(team.gd)}</td>
        <td class="pts-cell">{team.points}</td>
    </tr>"""


# -----------------------------------------------------------------------------
# Public render
# -----------------------------------------------------------------------------

def render_rankings_tab(
    league_table: list[Team],
    elo_rank_map: dict[str, int],
    match_log: list[dict],
) -> None:
    """Render the full Rankings HTML table via st.html()."""
    if not league_table:
        st.info("No teams to display.")
        return

    elo_floor, elo_span = _elo_range(league_table)
    rows: list[str] = []

    for rank, team in enumerate(league_table, start=1):
        if rank == FINALS_THRESHOLD + 1:
            rows.append(
                f'<tr class="finals-label"><td colspan="{NUM_COLS}">'
                f'<span class="finals-text">Top {FINALS_THRESHOLD} &mdash; Finals Threshold</span>'
                f'</td></tr>'
            )
        rows.append(_build_row(rank, team, elo_floor, elo_span, elo_rank_map, match_log))

    st.html(_RANKINGS_CSS + _TABLE_OPEN + "".join(rows) + _TABLE_CLOSE)
