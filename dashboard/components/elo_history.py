"""Elo History tab component."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config.constants import BASE_ELO
from config.teams import team_abbr, team_badge_html, team_color, team_short
from engine.elo import GrassrootsEloEngine
from models.team import Team


def _round_num(label: str | None) -> int:
    """Extract numeric round number from a label like 'Round 4'."""
    if not label:
        return 0
    digits = "".join(c for c in label if c.isdigit())
    return int(digits) if digits else 0


def _build_dataframe(engine: GrassrootsEloEngine, history: list[dict]) -> tuple[pd.DataFrame, str, str, str, int, bool]:
    """Build the Elo history DataFrame with a Round 0 baseline.

    Returns (df, x_field, x_title, x_col, max_round, has_rounds).
    """
    match_log = engine.match_log
    has_rounds = any(m.get("round") for m in match_log)

    if has_rounds:
        team_round_elo: dict[str, dict[int, float]] = {}
        for idx, entry in enumerate(match_log):
            rnd = _round_num(entry["round"])
            snapshot = history[idx]
            for team_name in (entry["home"], entry["away"]):
                if team_name not in team_round_elo:
                    team_round_elo[team_name] = {}
                team_round_elo[team_name][rnd] = round(snapshot[team_name], 1)

        initial = getattr(engine, "initial_elos", {})
        rows: list[dict] = []
        for team_name in team_round_elo:
            rows.append({"Round": 0, "Team": team_name, "Elo": round(initial.get(team_name, float(BASE_ELO)), 1)})
            for rnd, elo in sorted(team_round_elo[team_name].items()):
                rows.append({"Round": rnd, "Team": team_name, "Elo": elo})

        df = pd.DataFrame(rows)
        return df, "Round:Q", "Round", "Round", int(df["Round"].max()), True
    else:
        team_match_count: dict[str, int] = {}
        rows = []
        for idx, entry in enumerate(match_log):
            snapshot = history[idx]
            for team_name in (entry["home"], entry["away"]):
                team_match_count[team_name] = team_match_count.get(team_name, 0) + 1
                rows.append({
                    "Match": team_match_count[team_name],
                    "Team": team_name,
                    "Elo": round(snapshot[team_name], 1),
                })

        df = pd.DataFrame(rows)
        return df, "Match:Q", "Match", "Match", int(df["Match"].max()), False


def _compute_movement(
    elo_ranked: list[Team],
    max_round: int,
    has_rounds: bool,
    df: pd.DataFrame,
    x_col: str,
    window: int = 5,
) -> tuple[list[dict], list[dict], list[dict], str]:
    """Compute rising/falling/steady movement data over the last *window* rounds.

    Returns (risers, fallers, steady, round_label).
    """
    lookback = max(0, max_round - window)
    actual_window = max_round - lookback

    if lookback == 0:
        lb_df = df[df[x_col] == 0]
        baseline_elos = dict(zip(lb_df["Team"], lb_df["Elo"])) if not lb_df.empty else {t.name: float(BASE_ELO) for t in elo_ranked}
    else:
        lb_df = df[df[x_col] == lookback]
        baseline_elos = dict(zip(lb_df["Team"], lb_df["Elo"]))

    movement_data = []
    for team in elo_ranked:
        base = baseline_elos.get(team.name, float(BASE_ELO))
        delta = team.elo - base
        movement_data.append({"team": team.name, "current": team.elo, "delta": delta})

    risers = sorted([m for m in movement_data if m["delta"] > 0], key=lambda m: m["delta"], reverse=True)
    fallers = sorted([m for m in movement_data if m["delta"] < 0], key=lambda m: m["delta"])
    steady = [m for m in movement_data if m["delta"] == 0]
    round_label = f"last {actual_window} rounds" if has_rounds else f"last {actual_window} matches"

    return risers, fallers, steady, round_label


def _render_sidebar_panel(
    risers: list[dict],
    fallers: list[dict],
    steady: list[dict],
    round_label: str,
    selected_teams: list[str],
) -> None:
    """Render the Rising / Falling movement panel for the right-hand column."""
    st.markdown('''<style>
    .mv-name-short { display: none; }
    .mv-name-full { display: inline; }
    .elo-spacer { height: 40px; }
    @media (max-width: 640px) {
        .mv-name-full { display: none; }
        .mv-name-short { display: inline; }
        .elo-spacer { height: 0; }
        [data-testid="stHorizontalBlock"]:has(.elo-spacer) { gap: 0 !important; }
    }
    </style>''', unsafe_allow_html=True)

    # Filter to only show teams currently selected on the chart
    selected_set = set(selected_teams)
    vis_risers = [m for m in risers if m["team"] in selected_set]
    vis_fallers = [m for m in fallers if m["team"] in selected_set]
    vis_steady = [m for m in steady if m["team"] in selected_set]

    # Rising section
    lines_html = (
        f'<div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; '
        f'letter-spacing:0.8px; font-weight:700; margin-bottom:4px; '
        f'border-bottom:2px solid #e2e8f0; padding-bottom:4px">Rising '
        f'<span style="text-transform:none; font-weight:400; color:#94a3b8">({round_label})</span></div>'
    )
    for m in vis_risers:
        short = team_short(m["team"])
        badge = team_badge_html(m["team"], size=16)
        lines_html += (
            f'<div style="font-size:0.82rem; padding:3px 0; line-height:1.3; display:flex; align-items:center; gap:6px">'
            f'{badge}'
            f'<span style="color:#16a34a; font-weight:700; font-size:0.85rem; '
            f'min-width:36px; text-align:right">+{m["delta"]:.0f}</span>'
            f'<span class="mv-name-full" style="color:#334155; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{m["team"]}</span>'
            f'<span class="mv-name-short" style="color:#334155">{short}</span></div>'
        )
    if not vis_risers:
        lines_html += '<div style="font-size:0.8rem; color:#94a3b8; padding:3px 0">—</div>'

    # Falling section
    lines_html += (
        f'<div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; '
        f'letter-spacing:0.8px; font-weight:700; margin-top:16px; margin-bottom:4px; '
        f'border-bottom:2px solid #e2e8f0; padding-bottom:4px">Falling '
        f'<span style="text-transform:none; font-weight:400; color:#94a3b8">({round_label})</span></div>'
    )
    for m in vis_fallers:
        short = team_short(m["team"])
        badge = team_badge_html(m["team"], size=16)
        lines_html += (
            f'<div style="font-size:0.82rem; padding:3px 0; line-height:1.3; display:flex; align-items:center; gap:6px">'
            f'{badge}'
            f'<span style="color:#dc2626; font-weight:700; font-size:0.85rem; '
            f'min-width:36px; text-align:right">{m["delta"]:.0f}</span>'
            f'<span class="mv-name-full" style="color:#334155; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{m["team"]}</span>'
            f'<span class="mv-name-short" style="color:#334155">{short}</span></div>'
        )
    if not vis_fallers:
        lines_html += '<div style="font-size:0.8rem; color:#94a3b8; padding:3px 0">—</div>'

    # Steady section (only if any)
    if vis_steady:
        lines_html += (
            f'<div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; '
            f'letter-spacing:0.8px; font-weight:700; margin-top:16px; margin-bottom:4px; '
            f'border-bottom:2px solid #e2e8f0; padding-bottom:4px">Steady</div>'
        )
        for m in vis_steady:
            short = team_short(m["team"])
            badge = team_badge_html(m["team"], size=16)
            lines_html += (
                f'<div style="font-size:0.82rem; padding:3px 0; line-height:1.3; display:flex; align-items:center; gap:6px; color:#94a3b8">'
                f'{badge}'
                f'<span style="font-weight:700; font-size:0.85rem; min-width:36px; text-align:right">0</span>'
                f'<span class="mv-name-full">{m["team"]}</span>'
                f'<span class="mv-name-short">{short}</span></div>'
            )

    st.markdown(lines_html, unsafe_allow_html=True)


def _deconflict_labels(
    last_df: pd.DataFrame,
    y_lo: float,
    y_hi: float,
    chart_height: int = 480,
    min_px: int = 14,
) -> pd.DataFrame:
    """Nudge label y-positions apart to prevent overlap.

    Returns a copy of *last_df* with an extra ``LabelElo`` column.
    """
    if last_df.empty:
        result = last_df.copy()
        result["LabelElo"] = result["Elo"]
        return result

    pts = last_df.sort_values("Elo", ascending=False).reset_index(drop=True)
    elo_span = y_hi - y_lo
    if elo_span <= 0:
        pts["LabelElo"] = pts["Elo"]
        return pts

    min_gap = (min_px / chart_height) * elo_span
    positions = pts["Elo"].tolist()

    # Top-down push: prevent higher labels from overlapping lower ones
    for i in range(1, len(positions)):
        if positions[i - 1] - positions[i] < min_gap:
            positions[i] = positions[i - 1] - min_gap

    # Bottom-up recovery: if lowest label fell below y_lo, shift block up
    if positions and positions[-1] < y_lo:
        shift = y_lo - positions[-1]
        positions = [p + shift for p in positions]

    pts["LabelElo"] = positions
    return pts


def _render_chart(
    df: pd.DataFrame,
    x_field: str,
    x_title: str,
    x_col: str,
    max_round: int,
    selected_teams: list[str],
) -> None:
    """Render the Altair Elo history line chart with multi-layered identification.

    Layer 1 (default): Clean lines, no labels — sidebar panel is the legend.
    Layer 2 (line hover): Spotlight effect — hovered line bold + full-name label,
                          all others fade to background.
    Layer 3 (ambient): Faint abbreviated labels at line endpoints provide
                       quick orientation without permanent clutter.
    """
    filtered = df[df["Team"].isin(selected_teams)]

    elo_min = filtered["Elo"].min()
    elo_max = filtered["Elo"].max()
    y_pad = max(25, (elo_max - elo_min) * 0.1)
    y_lo = max(0, round(elo_min - y_pad, -1))
    y_hi = round(elo_max + y_pad, -1)

    # Build team-color mapping
    color_domain = []
    color_range = []
    for t in selected_teams:
        color_domain.append(t)
        color_range.append(team_color(t))

    color_scale = alt.Scale(domain=color_domain, range=color_range)

    # Two selection params with different empty strategies:
    #   highlight (empty="all")  — lines stay fully visible when nothing is hovered
    #   hover_active (empty="none") — labels react: faint when idle, spotlight on hover
    highlight = alt.selection_point(
        fields=["Team"],
        on="pointerover",
        empty="all",
    )
    hover_active = alt.selection_point(
        fields=["Team"],
        on="pointerover",
        empty="none",
    )

    x_enc = alt.X(
        x_field,
        title=None,
        scale=alt.Scale(domain=[0, max_round]),
        axis=alt.Axis(
            tickMinStep=1,
            grid=False,
        ),
    )
    y_enc = alt.Y(
        "Elo:Q",
        title=None,
        scale=alt.Scale(domain=[y_lo, y_hi]),
        axis=alt.Axis(
            gridColor="#f1f5f9",
            gridDash=[3, 3],
        ),
    )
    color_enc = alt.Color("Team:N", scale=color_scale, legend=None)

    # Invisible wide hit-area for hover detection (carries both selections)
    hit_area = alt.Chart(filtered).mark_line(
        strokeWidth=12, opacity=0,
    ).encode(
        x=x_enc, y=y_enc, color=color_enc,
    ).add_params(highlight, hover_active)

    # Visible lines — all opaque by default, spotlight on hover
    lines = alt.Chart(filtered).mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(size=20, filled=True),
    ).encode(
        x=x_enc, y=y_enc, color=color_enc,
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.12)),
        strokeWidth=alt.condition(highlight, alt.value(3), alt.value(1.5)),
        tooltip=[
            alt.Tooltip(f"{x_col}:Q", title=x_title),
            alt.Tooltip("Team:N"),
            alt.Tooltip("Elo:Q", format=".0f"),
        ],
    )

    # ── Endpoint labels (deconflicted) ─────────────────────────────
    last_points = filtered[filtered[x_col] == max_round].copy()
    last_points["Abbr"] = last_points["Team"].map(team_abbr)
    deconflicted = _deconflict_labels(last_points, y_lo, y_hi)

    label_y = alt.Y(
        "LabelElo:Q",
        scale=alt.Scale(domain=[y_lo, y_hi]),
    )

    # Leader lines connecting data points to nudged label positions
    needs_leader = deconflicted[abs(deconflicted["LabelElo"] - deconflicted["Elo"]) > 1].copy()
    leaders = alt.Chart(needs_leader).mark_rule(
        strokeWidth=0.7,
        strokeDash=[2, 2],
    ).encode(
        x=x_enc,
        y=alt.Y("Elo:Q", scale=alt.Scale(domain=[y_lo, y_hi])),
        y2=alt.Y2("LabelElo:Q"),
        color=color_enc,
        opacity=alt.condition(highlight, alt.value(0.35), alt.value(0.08)),
    )

    # Layer 3: Faint abbreviated labels — visible when idle, hidden for the
    # hovered team (so the full-name label can take its place).
    # hover_active uses empty="none" → when nothing hovered all are FALSE
    # → second branch (0.3) → faint labels shown.
    # When team X hovered → X is TRUE → first branch (0) → hidden.
    abbr_labels = alt.Chart(deconflicted).mark_text(
        align="left",
        dx=8,
        fontSize=9,
        fontWeight="bold",
    ).encode(
        x=x_enc,
        y=label_y,
        text="Abbr:N",
        color=color_enc,
        opacity=alt.condition(hover_active, alt.value(0), alt.value(0.3)),
    )

    # Layer 2: Abbreviation label — hidden when idle, appears bold for
    # the hovered team as a spotlight annotation.
    full_label = alt.Chart(deconflicted).mark_text(
        align="left",
        dx=8,
        fontSize=11,
        fontWeight="bold",
    ).encode(
        x=x_enc,
        y=label_y,
        text="Abbr:N",
        color=color_enc,
        opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
    )

    chart = (
        alt.layer(hit_area, lines, leaders, abbr_labels, full_label)
        .properties(
            height=480,
            padding={"left": 0, "top": 10, "right": 16, "bottom": 10},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#64748b",
            labelFontSize=11,
            tickColor="#e2e8f0",
            domainColor="#e2e8f0",
        )
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def render_elo_history_tab(
    engine: GrassrootsEloEngine,
    history: list[dict],
    team_names: list[str],
    elo_ranked: list[Team],
    league_table: list[Team],
) -> None:
    """Render the full Elo History tab with an annotated two-column layout."""
    if not history:
        st.info("No match history to display. Process some matches first.")
        return

    df, x_field, x_title, x_col, max_round, has_rounds = _build_dataframe(engine, history)

    # Precompute movement data (shared between chart filters and sidebar panel)
    risers, fallers, steady, round_label = _compute_movement(
        elo_ranked, max_round, has_rounds, df, x_col,
    )

    # Filter presets
    top6 = [t.name for t in league_table[:6]]
    bot6 = [t.name for t in league_table[6:]]

    # Biggest movers: top 3 risers + top 3 fallers
    top_risers = risers[:3]
    top_fallers = fallers[:3]
    biggest_movers = [m["team"] for m in top_risers + top_fallers]

    # ── Two-column layout: chart (left) + movement panel (right) ──
    # On mobile Streamlit stacks columns vertically, which is fine.
    col_chart, col_panel = st.columns([7, 3], gap="medium")

    with col_chart:
        # Filter pills — tightly coupled above the chart
        filter_options = ["Biggest movers", "All teams", "Semi-finalists (top 6)", "Outside top 6", "Custom"]
        filter_mode = st.pills(
            "Filter",
            filter_options,
            default="Biggest movers",
            label_visibility="collapsed",
        )
        if filter_mode is None:
            filter_mode = "Biggest movers"

        if filter_mode == "Biggest movers":
            selected_teams = biggest_movers
        elif filter_mode == "All teams":
            selected_teams = team_names
        elif filter_mode == "Semi-finalists (top 6)":
            selected_teams = top6
        elif filter_mode == "Outside top 6":
            selected_teams = bot6
        else:
            selected_teams = st.multiselect(
                "Select teams",
                options=team_names,
                default=team_names,
            )

        if selected_teams:
            _render_chart(df, x_field, x_title, x_col, max_round, selected_teams)
        else:
            st.warning("Select at least one team to display.")

    with col_panel:
        # Spacer aligns with chart top on desktop; collapses on mobile
        st.markdown('<div class="elo-spacer"></div>', unsafe_allow_html=True)
        _render_sidebar_panel(risers, fallers, steady, round_label, selected_teams if selected_teams else [])
