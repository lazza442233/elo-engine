"""Rankings tab component."""

from __future__ import annotations

import streamlit as st

from dashboard.helpers import form_dots, team_form
from config.teams import team_badge_html, team_color, team_short
from models.team import Team


def render_rankings_tab(
    league_table: list[Team],
    elo_rank_map: dict[str, int],
    match_log: list[dict],
) -> None:
    """Render the full Rankings HTML table via st.html()."""
    if not league_table:
        st.info("No teams to display.")
        return

    style = '''<style>
    :root { --bg: #ffffff; --fg: #262730; --muted: #64748b; --border: #f1f5f9; --th-border: #e2e8f0; --bar-bg: #f1f5f9; --tip-bg: #1e293b; --tip-fg: #ffffff; }
    @media (prefers-color-scheme: dark) {
        :root { --bg: #0e1117; --fg: #fafafa; --muted: #94a3b8; --border: #1e293b; --th-border: #334155; --bar-bg: #1e293b; --tip-bg: #f1f5f9; --tip-fg: #0e1117; }
    }
    body { font-family: "Source Sans Pro", "Segoe UI", sans-serif; margin:0; padding:4px 0; background:transparent; color:var(--fg); }
    table { width:100%; border-collapse:collapse; font-size:0.88rem; }
    th { text-align:center; padding:10px 8px; border-bottom:2px solid var(--th-border); color:var(--muted); font-weight:600; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.5px; }
    td { padding:10px 8px; vertical-align:middle; }
    tr:hover td { background: color-mix(in srgb, var(--border) 50%, transparent); }
    .sep td { border-bottom:2px dashed #cbd5e1; }
    .dot-wrap { position:relative; display:inline-block; margin:0 2px; cursor:default; }
    .dot-wrap:hover::after {
        content: attr(data-tip);
        position: absolute; bottom: calc(100% + 6px); left: 50%; transform: translateX(-50%);
        background: var(--tip-bg); color: var(--tip-fg); padding: 5px 12px; border-radius: 6px;
        font-size: 0.72rem; white-space: nowrap; z-index: 9999;
        pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .dot-wrap:hover::before {
        content: ""; position: absolute; bottom: calc(100% + 2px); left: 50%; transform: translateX(-50%);
        border: 4px solid transparent; border-top-color: var(--tip-bg); z-index: 9999;
    }
    .drift-chip { display:inline-block; padding:1px 5px; border-radius:6px; font-size:0.65rem; font-weight:600; line-height:1.4; }
    .drift-hdr { position:relative; cursor:help; border-bottom:1px dashed var(--muted); padding-bottom:1px; }
    .drift-hdr:hover::after {
        content: attr(data-tip);
        position: absolute; top: calc(100% + 8px); left: 50%; transform: translateX(-50%);
        background: var(--tip-bg); color: var(--tip-fg); padding: 8px 14px; border-radius: 6px;
        font-size: 0.72rem; white-space: pre; z-index: 9999; width:max-content; max-width:280px;
        pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-transform:none; letter-spacing:0; font-weight:400; line-height:1.6;
    }
    .drift-hdr:hover::before {
        content: ""; position: absolute; top: calc(100% + 4px); left: 50%; transform: translateX(-50%);
        border: 4px solid transparent; border-bottom-color: var(--tip-bg); z-index: 9999;
    }
    /* Mobile: hide secondary columns */
    .col-secondary { }
    .mobile-detail { display: none; }
    .name-short { display: none; }
    .name-full { display: inline; }
    @media (max-width: 640px) {
        table { font-size: 0.82rem; }
        th, td { padding: 8px 4px; }
        .col-secondary { display: none; }
        .name-full { display: none; }
        .name-short { display: inline; }
        .elo-cell { background: none !important; font-size: 0.88rem !important; font-weight: 600 !important; }
        .pts-cell { font-size: 0.88rem !important; font-weight: 600 !important; }
        .gd-cell { font-weight: 500 !important; font-size: 0.82rem !important; }
    }
    </style>'''

    header = '''<table>
    <thead><tr>
        <th style="width:28px">#</th>
        <th class="col-secondary" style="width:40px; font-size:0.65rem"><span class="drift-hdr" data-tip="How Elo ranks compare to the ladder.&#10;&#10;Green chip: Underrated.&#10;Elo thinks they're better than&#10;their win-loss record suggests.&#10;&#10;Red chip: Overrated.&#10;Elo thinks they're worse than&#10;their win-loss record suggests.&#10;&#10;Grey chip: Spot on.&#10;Elo and ladder agree.">DRIFT</span></th>
        <th style="text-align:left">Team</th>
        <th style="width:55px">Elo</th>
        <th style="width:110px">Form</th>
        <th class="col-secondary" style="width:30px">MP</th>
        <th class="col-secondary" style="width:30px">W</th>
        <th class="col-secondary" style="width:30px">D</th>
        <th class="col-secondary" style="width:30px">L</th>
        <th class="col-secondary" style="width:36px">GF</th>
        <th class="col-secondary" style="width:36px">GA</th>
        <th style="width:42px">GD</th>
        <th style="width:36px">Pts</th>
    </tr></thead><tbody>
    '''

    elo_values = [t.elo for t in league_table]
    elo_floor = min(elo_values)
    elo_ceil = max(elo_values)
    elo_span = elo_ceil - elo_floor if elo_ceil > elo_floor else 1

    rows_html: list[str] = []
    for rank, team in enumerate(league_table, 1):
        form = team_form(team.name, match_log)
        form_html = form_dots(form)
        elo_pos = elo_rank_map[team.name]

        diff = rank - elo_pos
        if diff > 0:
            drift_bg = "#dcfce7"
            drift_fg = "#16a34a"
            drift_text = f"+{diff}"
        elif diff < 0:
            drift_bg = "#fee2e2"
            drift_fg = "#dc2626"
            drift_text = str(diff)
        else:
            drift_bg = "#f1f5f9"
            drift_fg = "#94a3b8"
            drift_text = "±0"
        drift_chip = (
            f'<span class="drift-chip" style="background:{drift_bg}; color:{drift_fg}">'
            f'{drift_text}</span>'
        )

        if team.gd > 0:
            gd_colour = "#22c55e"
            gd_str = f"+{team.gd}"
        elif team.gd < 0:
            gd_colour = "#ef4444"
            gd_str = str(team.gd)
        else:
            gd_colour = "#94a3b8"
            gd_str = "0"

        bar_pct = max(8, (team.elo - elo_floor) / elo_span * 100)
        tc = team_color(team.name)
        badge = team_badge_html(team.name, size=22)
        short = team_short(team.name)
        tr_class = ' class="sep"' if rank == 6 else ""

        rows_html.append(f'''<tr{tr_class}>
            <td style="text-align:center; font-weight:700; color:var(--muted, #94a3b8)">{rank}</td>
            <td class="col-secondary" style="text-align:center; color:var(--muted, #94a3b8); font-size:0.8rem">{drift_chip}</td>
            <td style="font-weight:600; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap">{badge} <span class="name-full" style="vertical-align:middle">{team.name}</span><span class="name-short" style="vertical-align:middle">{short}</span></td>
            <td class="elo-cell" style="text-align:center; font-weight:700; font-size:0.95rem; background:linear-gradient(to right, {tc}20 {bar_pct:.0f}%, transparent {bar_pct:.0f}%)">{team.elo:.0f}</td>
            <td style="text-align:center; white-space:nowrap">{form_html}</td>
            <td class="col-secondary" style="text-align:center; color:var(--muted, #94a3b8); font-size:0.82rem">{team.played}</td>
            <td class="col-secondary" style="text-align:center; color:#22c55e; font-size:0.82rem">{team.wins}</td>
            <td class="col-secondary" style="text-align:center; color:#94a3b8; font-size:0.82rem">{team.draws}</td>
            <td class="col-secondary" style="text-align:center; color:#ef4444; font-size:0.82rem">{team.losses}</td>
            <td class="col-secondary" style="text-align:center; color:var(--muted, #94a3b8); font-size:0.82rem">{team.gf}</td>
            <td class="col-secondary" style="text-align:center; color:var(--muted, #94a3b8); font-size:0.82rem">{team.ga}</td>
            <td class="gd-cell" style="text-align:center; color:{gd_colour}; font-weight:600">{gd_str}</td>
            <td class="pts-cell" style="text-align:center; font-weight:700; font-size:1rem">{team.points}</td>
        </tr>''')

    footer = "</tbody></table>"
    st.html(style + header + "".join(rows_html) + footer)
