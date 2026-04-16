"""Elo History tab component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

from config.constants import BASE_ELO
from config.teams import team_abbr, team_badge_html, team_color, team_short
from engine.elo import GrassrootsEloEngine
from models.team import Team

XMode = Literal["date", "round", "match"]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _round_num(label: str | None) -> int:
    """Extract a numeric round from labels like 'Round 4'."""
    if not label:
        return 0
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


def _safe_dt(value: object, fallback: pd.Timestamp) -> pd.Timestamp:
    """Parse a timestamp, falling back safely."""
    if not value:
        return fallback
    try:
        return pd.to_datetime(value)
    except Exception:
        return fallback


def _baseline_elo(team_name: str, initial_elos: dict[str, float]) -> float:
    return round(float(initial_elos.get(team_name, float(BASE_ELO))), 1)


def _movement_bucket(movement: list[dict], sign: str) -> list[dict]:
    if sign == "positive":
        return sorted((m for m in movement if m["delta"] > 0), key=lambda m: m["delta"], reverse=True)
    if sign == "negative":
        return sorted((m for m in movement if m["delta"] < 0), key=lambda m: m["delta"])
    return [m for m in movement if m["delta"] == 0]


# -----------------------------------------------------------------------------
# Data shaping
# -----------------------------------------------------------------------------

def _build_dataframe(
    engine: GrassrootsEloEngine,
    history: list[dict],
) -> tuple[pd.DataFrame, str, str, str, object, XMode]:
    """
    Build the Elo history dataframe and metadata.

    Returns:
        df, x_field, x_title, x_col, max_x, x_mode
    """
    match_log = engine.match_log
    has_dates = any(m.get("date") for m in match_log)
    has_rounds = any(m.get("round") for m in match_log)
    initial_elos = getattr(engine, "initial_elos", {})

    if has_dates:
        team_names_seen: set[str] = set()
        rows: list[dict] = []
        prev_elo: dict[str, float] = {}

        parsed_dates = [
            pd.to_datetime(m["date"])
            for m in match_log
            if m.get("date")
            and not pd.isna(pd.to_datetime(m["date"], errors="coerce"))
        ]
        baseline_date = (
            (min(parsed_dates) - pd.Timedelta(days=1))
            if parsed_dates
            else pd.Timestamp.now() - pd.Timedelta(days=180)
        )

        for idx, entry in enumerate(match_log):
            snapshot = history[idx]
            dt = _safe_dt(entry.get("date"), baseline_date + pd.Timedelta(hours=idx))

            home, away = entry["home"], entry["away"]
            hs, as_ = entry["home_score"], entry["away_score"]

            for team_name in (home, away):
                team_names_seen.add(team_name)
                elo_now = round(float(snapshot[team_name]), 1)
                base = prev_elo.get(team_name, initial_elos.get(team_name, float(BASE_ELO)))
                delta = round(elo_now - base, 1)

                if team_name == home:
                    opp = away
                    result = "W" if hs > as_ else ("D" if hs == as_ else "L")
                    score_str = f"{hs}–{as_}"
                else:
                    opp = home
                    result = "W" if as_ > hs else ("D" if hs == as_ else "L")
                    score_str = f"{as_}–{hs}"

                sign = "+" if delta > 0 else ""
                ctx = f"{result} {score_str} vs {team_short(opp)} ({sign}{delta:.0f})"

                rows.append(
                    {
                        "Date": dt,
                        "Team": team_name,
                        "Elo": elo_now,
                        "Context": ctx,
                        "EloDelta": delta,
                    }
                )
                prev_elo[team_name] = elo_now

        for team_name in team_names_seen:
            rows.append(
                {
                    "Date": baseline_date,
                    "Team": team_name,
                    "Elo": _baseline_elo(team_name, initial_elos),
                    "Context": "Season start",
                    "EloDelta": 0.0,
                }
            )

        df = pd.DataFrame(rows).sort_values(["Team", "Date"]).reset_index(drop=True)
        max_date = df["Date"].max()

        # Extend each line to the latest point so step-after holds the final Elo.
        trailing: list[dict] = []
        for team_name in team_names_seen:
            team_df = df[df["Team"] == team_name]
            if team_df.empty:
                continue
            last_date = team_df["Date"].max()
            if last_date < max_date:
                last_row = team_df.loc[team_df["Date"].idxmax()]
                trailing.append(
                    {
                        "Date": max_date,
                        "Team": team_name,
                        "Elo": float(last_row["Elo"]),
                        "Context": "Current",
                        "EloDelta": 0.0,
                    }
                )

        if trailing:
            df = pd.concat([df, pd.DataFrame(trailing)], ignore_index=True).sort_values(
                ["Team", "Date"]
            )

        return df, "Date:T", "Date", "Date", max_date, "date"

    if has_rounds:
        team_round_elo: dict[str, dict[int, float]] = {}
        for idx, entry in enumerate(match_log):
            rnd = _round_num(entry.get("round"))
            snapshot = history[idx]
            for team_name in (entry["home"], entry["away"]):
                team_round_elo.setdefault(team_name, {})[rnd] = round(float(snapshot[team_name]), 1)

        rows: list[dict] = []
        for team_name, round_map in team_round_elo.items():
            rows.append(
                {
                    "Round": 0,
                    "Team": team_name,
                    "Elo": _baseline_elo(team_name, initial_elos),
                }
            )
            for rnd, elo in sorted(round_map.items()):
                rows.append({"Round": rnd, "Team": team_name, "Elo": elo})

        df = pd.DataFrame(rows)
        return df, "Round:Q", "Round", "Round", int(df["Round"].max()), "round"

    team_match_count: dict[str, int] = {}
    rows = []
    for idx, entry in enumerate(match_log):
        snapshot = history[idx]
        for team_name in (entry["home"], entry["away"]):
            team_match_count[team_name] = team_match_count.get(team_name, 0) + 1
            rows.append(
                {
                    "Match": team_match_count[team_name],
                    "Team": team_name,
                    "Elo": round(float(snapshot[team_name]), 1),
                }
            )

    df = pd.DataFrame(rows)
    return df, "Match:Q", "Match", "Match", int(df["Match"].max()), "match"


def _compute_movement(
    elo_ranked: list[Team],
    max_x: object,
    x_mode: XMode,
    df: pd.DataFrame,
    x_col: str,
    window: int = 5,
) -> tuple[list[dict], list[dict], list[dict], str]:
    """
    Compute movement buckets over the last window of rounds/matches.
    """
    movement: list[dict] = []

    if x_mode == "date":
        for team in elo_ranked:
            team_df = df[df["Team"] == team.name]
            start_row = team_df[team_df["Context"] == "Season start"]
            season_start_elo = (
                float(start_row.iloc[0]["Elo"]) if not start_row.empty
                else float(BASE_ELO)
            )
            match_df = team_df[~team_df["Context"].isin(("Season start", "Current"))]
            match_df = match_df.sort_values("Date")
            if match_df.empty:
                movement.append({"team": team.name, "current": team.elo, "delta": 0.0})
                continue

            n = len(match_df)
            if n <= window:
                # Window reaches back to season start — use pre-season Elo
                base_elo = season_start_elo
            else:
                # Elo after the match just before the window
                base_elo = float(match_df.iloc[n - window - 1]["Elo"])
            movement.append(
                {
                    "team": team.name,
                    "current": team.elo,
                    "delta": team.elo - base_elo,
                }
            )

        return (
            _movement_bucket(movement, "positive"),
            _movement_bucket(movement, "negative"),
            _movement_bucket(movement, "zero"),
            f"last {window} matches",
        )

    max_x_int = int(max_x)
    lookback = max(0, max_x_int - window)
    actual_window = max_x_int - lookback

    if lookback == 0:
        lb_df = df[df[x_col] == 0]
        baseline_elos = (
            dict(zip(lb_df["Team"], lb_df["Elo"]))
            if not lb_df.empty
            else {t.name: float(BASE_ELO) for t in elo_ranked}
        )
    else:
        lb_df = df[df[x_col] == lookback]
        baseline_elos = dict(zip(lb_df["Team"], lb_df["Elo"])) if not lb_df.empty else {}

    for team in elo_ranked:
        base = float(baseline_elos.get(team.name, BASE_ELO))
        movement.append({"team": team.name, "current": team.elo, "delta": team.elo - base})

    return (
        _movement_bucket(movement, "positive"),
        _movement_bucket(movement, "negative"),
        _movement_bucket(movement, "zero"),
        f"last {actual_window} rounds" if x_mode == "round" else f"last {actual_window} matches",
    )


# -----------------------------------------------------------------------------
# Sidebar movement panel
# -----------------------------------------------------------------------------

def _movement_row_html(team_name: str, delta: float, muted: bool = False) -> str:
    short = team_short(team_name)
    badge = team_badge_html(team_name, size=16)
    delta_text = "0" if delta == 0 else f"{delta:+.0f}"
    delta_color = "#94a3b8" if delta == 0 else ("#16a34a" if delta > 0 else "#dc2626")
    text_color = "#94a3b8" if muted else "#334155"

    return (
        f'<div style="font-size:0.82rem; padding:3px 0; line-height:1.3; display:flex; align-items:center; gap:6px">'
        f"{badge}"
        f'<span style="color:{delta_color}; font-weight:700; font-size:0.85rem; min-width:36px; text-align:right">{delta_text}</span>'
        f'<span class="mv-name-full" style="color:{text_color}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{team_name}</span>'
        f'<span class="mv-name-short" style="color:{text_color}">{short}</span>'
        f"</div>"
    )


def _render_sidebar_panel(
    risers: list[dict],
    fallers: list[dict],
    steady: list[dict],
    round_label: str,
    selected_teams: list[str],
) -> None:
    """Render the rising/falling movement panel."""
    st.markdown(
        """
        <style>
        .mv-name-short { display: none; }
        .mv-name-full { display: inline; }
        .elo-spacer { height: 40px; }
        @media (max-width: 640px) {
            .mv-name-full { display: none; }
            .mv-name-short { display: inline; }
            .elo-spacer { height: 0; }
            [data-testid="stHorizontalBlock"]:has(.elo-spacer) { gap: 0 !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    selected_set = set(selected_teams)
    vis_risers = [m for m in risers if m["team"] in selected_set]
    vis_fallers = [m for m in fallers if m["team"] in selected_set]
    vis_steady = [m for m in steady if m["team"] in selected_set]

    def section_header(title: str, subtitle: str | None = None) -> str:
        sub = (
            f'<span style="text-transform:none; font-weight:400; color:#94a3b8">({subtitle})</span>'
            if subtitle
            else ""
        )
        return (
            f'<div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; '
            f'letter-spacing:0.8px; font-weight:700; margin-bottom:4px; '
            f'border-bottom:2px solid #e2e8f0; padding-bottom:4px">{title} {sub}</div>'
        )

    html = section_header("Rising", round_label)
    html += "".join(_movement_row_html(m["team"], float(m["delta"])) for m in vis_risers)
    if not vis_risers:
        html += '<div style="font-size:0.8rem; color:#94a3b8; padding:3px 0">—</div>'

    html += '<div style="margin-top:16px"></div>'
    html += section_header("Falling", round_label)
    html += "".join(_movement_row_html(m["team"], float(m["delta"])) for m in vis_fallers)
    if not vis_fallers:
        html += '<div style="font-size:0.8rem; color:#94a3b8; padding:3px 0">—</div>'

    if vis_steady:
        html += section_header("Steady")
        html += "".join(_movement_row_html(m["team"], 0.0, muted=True) for m in vis_steady)

    st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Label deconfliction
# -----------------------------------------------------------------------------

def _deconflict_labels(
    last_df: pd.DataFrame,
    y_lo: float,
    y_hi: float,
    chart_height: int = 480,
    min_px: int = 14,
) -> pd.DataFrame:
    """
    Nudge label positions apart to reduce overlap.
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

    for i in range(1, len(positions)):
        if positions[i - 1] - positions[i] < min_gap:
            positions[i] = positions[i - 1] - min_gap

    if positions and positions[-1] < y_lo:
        shift = y_lo - positions[-1]
        positions = [p + shift for p in positions]

    pts["LabelElo"] = positions
    return pts


# -----------------------------------------------------------------------------
# Chart rendering
# -----------------------------------------------------------------------------

def _render_chart(
    df: pd.DataFrame,
    x_field: str,
    x_title: str,
    x_col: str,
    max_x: object,
    x_mode: XMode,
    selected_teams: list[str],
) -> None:
    """
    Render the Altair Elo history line chart.

    The chart uses:
    - step-after lines
    - hover spotlighting
    - nearest-point tooltips
    - deconflicted endpoint labels
    """
    filtered = df[df["Team"].isin(selected_teams)].copy()
    if filtered.empty:
        st.warning("No data to display for the current selection.")
        return

    elo_min = float(filtered["Elo"].min())
    elo_max = float(filtered["Elo"].max())
    y_pad = max(30, (elo_max - elo_min) * 0.12)
    y_lo = max(0, round(elo_min - y_pad, -1))
    y_hi = round(elo_max + y_pad, -1)

    color_scale = alt.Scale(
        domain=selected_teams,
        range=[team_color(t) for t in selected_teams],
    )

    highlight = alt.selection_point(
        fields=["Team"],
        on="pointerover",
        clear="pointerout",
        empty=True,
    )
    hover_active = alt.selection_point(
        fields=["Team"],
        on="pointerover",
        clear="pointerout",
        empty=False,
    )
    nearest = alt.selection_point(
        nearest=True,
        on="pointerover",
        clear="pointerout",
        fields=[x_col, "Team"],
        empty=False,
    )

    x_enc = alt.X(
        x_field,
        title=None,
        **({} if x_mode == "date" else {"scale": alt.Scale(domain=[0, max_x])}),
        axis=alt.Axis(
            **({"format": "%d %b"} if x_mode == "date" else {"tickMinStep": 1}),
            grid=False,
            labelPadding=6,
        ),
    )
    y_enc = alt.Y(
        "Elo:Q",
        title=None,
        scale=alt.Scale(domain=[y_lo, y_hi]),
        axis=alt.Axis(
            gridColor="#e8ecf1",
            gridDash=[3, 3],
            gridOpacity=0.6,
            labelPadding=6,
        ),
    )
    color_enc = alt.Color("Team:N", scale=color_scale, legend=None)

    base_rule = (
        alt.Chart(pd.DataFrame({"y": [float(BASE_ELO)]}))
        .mark_rule(strokeDash=[6, 4], strokeWidth=0.8, color="#cbd5e1")
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[y_lo, y_hi])))
    )

    hit_area = (
        alt.Chart(filtered)
        .mark_line(strokeWidth=6, opacity=0, interpolate="step-after")
        .encode(x=x_enc, y=y_enc, color=color_enc)
        .add_params(highlight, hover_active)
    )

    lines = (
        alt.Chart(filtered)
        .mark_line(strokeWidth=1.8, interpolate="step-after")
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=alt.condition(highlight, alt.value(0.9), alt.value(0.08)),
            strokeWidth=alt.condition(highlight, alt.value(2.5), alt.value(1)),
        )
    )

    tooltip = (
        [
            alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
            alt.Tooltip("Team:N"),
            alt.Tooltip("Elo:Q", format=".0f"),
            alt.Tooltip("Context:N", title="Match"),
        ]
        if x_mode == "date" and "Context" in filtered.columns
        else [
            alt.Tooltip(
                f"{x_col}:{'T' if x_mode == 'date' else 'Q'}",
                title=x_title,
                **({"format": "%d %b %Y"} if x_mode == "date" else {}),
            ),
            alt.Tooltip("Team:N"),
            alt.Tooltip("Elo:Q", format=".0f"),
        ]
    )

    point_hit = (
        alt.Chart(filtered)
        .mark_circle(size=150)
        .encode(x=x_enc, y=y_enc, color=color_enc, opacity=alt.value(0), tooltip=tooltip)
        .add_params(nearest)
    )

    points_visible = (
        alt.Chart(filtered)
        .mark_circle()
        .encode(
            x=x_enc,
            y=y_enc,
            color=color_enc,
            opacity=alt.condition(hover_active, alt.value(0.85), alt.value(0)),
            size=alt.condition(nearest, alt.value(70), alt.value(0)),
            tooltip=tooltip,
        )
    )

    if x_mode == "date":
        last_idx = filtered.groupby("Team")[x_col].idxmax()
        last_points = filtered.loc[last_idx].copy()
    else:
        last_points = filtered[filtered[x_col] == max_x].copy()

    last_points["Abbr"] = last_points["Team"].map(team_abbr)
    deconflicted = _deconflict_labels(last_points, y_lo, y_hi)

    label_y = alt.Y("LabelElo:Q", scale=alt.Scale(domain=[y_lo, y_hi]))

    needs_leader = deconflicted[abs(deconflicted["LabelElo"] - deconflicted["Elo"]) > 1].copy()
    leaders = (
        alt.Chart(needs_leader)
        .mark_rule(strokeWidth=0.7, strokeDash=[2, 2])
        .encode(
            x=x_enc,
            y=alt.Y("Elo:Q", scale=alt.Scale(domain=[y_lo, y_hi])),
            y2=alt.Y2("LabelElo:Q"),
            color=color_enc,
            opacity=alt.condition(highlight, alt.value(0.35), alt.value(0.08)),
        )
    )

    abbr_labels = (
        alt.Chart(deconflicted)
        .mark_text(align="left", dx=8, fontSize=9, fontWeight="bold")
        .encode(
            x=x_enc,
            y=label_y,
            text="Abbr:N",
            color=color_enc,
            opacity=alt.condition(hover_active, alt.value(0), alt.value(0.3)),
        )
    )

    full_label = (
        alt.Chart(deconflicted)
        .mark_text(align="left", dx=8, fontSize=11, fontWeight="bold")
        .encode(
            x=x_enc,
            y=label_y,
            text="Abbr:N",
            color=color_enc,
            opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
        )
    )

    chart = (
        alt.layer(
            base_rule,
            hit_area,
            lines,
            points_visible,
            point_hit,
            leaders,
            abbr_labels,
            full_label,
        )
        .properties(height=480, padding={"left": 4, "top": 14, "right": 20, "bottom": 14})
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#94a3b8",
            labelFontSize=11,
            labelFont='"Source Sans Pro", "Segoe UI", sans-serif',
            tickColor="#e2e8f0",
            domainColor="#e2e8f0",
            titleColor="#94a3b8",
        )
    )

    st.altair_chart(chart, width="stretch", theme=None)


# -----------------------------------------------------------------------------
# Public render entrypoint
# -----------------------------------------------------------------------------

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

    df, x_field, x_title, x_col, max_x, x_mode = _build_dataframe(engine, history)
    risers, fallers, steady, round_label = _compute_movement(
        elo_ranked, max_x, x_mode, df, x_col
    )

    top6 = [t.name for t in league_table[:6]]
    bot6 = [t.name for t in league_table[6:]]
    biggest_movers = [m["team"] for m in (risers[:3] + fallers[:3])]

    col_chart, col_panel = st.columns([7, 3], gap="medium")

    with col_chart:
        filter_mode = st.pills(
            "Filter",
            ["Biggest movers", "All teams", "Semi-finalists (top 6)", "Outside top 6", "Custom"],
            default="Biggest movers",
            label_visibility="collapsed",
        ) or "Biggest movers"

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
            _render_chart(df, x_field, x_title, x_col, max_x, x_mode, selected_teams)
        else:
            st.warning("Select at least one team to display.")

    with col_panel:
        st.markdown('<div class="elo-spacer"></div>', unsafe_allow_html=True)
        _render_sidebar_panel(
            risers=risers,
            fallers=fallers,
            steady=steady,
            round_label=round_label,
            selected_teams=selected_teams if selected_teams else [],
        )
