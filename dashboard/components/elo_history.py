"""Elo History tab component — All-Time Snake Chart (v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config.constants import BASE_ELO
from config.teams import team_abbr, team_color, team_short
from dashboard.data import GRADE_MAP, build_full_history
from engine.elo import GrassrootsEloEngine
from models.team import Team

XMode = Literal["date", "round", "match"]


# -----------------------------------------------------------------------------
# Small Utilities
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


def _chip(name: str, value_html: str, bg: str, border: str) -> str:
    """Render a single styled team chip as an HTML snippet."""
    c = team_color(name)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'background:{bg};border:1px solid {border};border-radius:6px;'
        f'padding:2px 8px;margin:1px 3px;white-space:nowrap">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{c};flex-shrink:0"></span>'
        f'<span style="font-weight:600;font-size:0.78rem;color:#1e293b">{team_short(name)}</span>'
        f'<span style="font-size:0.75rem;font-weight:700;{value_html}</span>'
        f'</span>'
    )


# -----------------------------------------------------------------------------
# Data Transformation & Stats (Vectorized)
# -----------------------------------------------------------------------------

def _get_elo_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate net Elo changes and volatility for all teams efficiently."""
    df_valid = df.dropna(subset=["Elo"])
    if df_valid.empty:
        return pd.DataFrame()

    grouped = df_valid.groupby("Team")["Elo"]
    stats = pd.DataFrame({
        "start": grouped.first(),
        "end": grouped.last(),
        "stdev": grouped.std().fillna(0)
    })
    stats["delta"] = stats["end"] - stats["start"]
    return stats


def _compute_biggest_movers(df: pd.DataFrame, n: int = 3) -> list[str]:
    """Return top *n* risers + top *n* fallers by net Elo change."""
    stats = _get_elo_deltas(df)
    if stats.empty:
        return[]

    stats = stats.sort_values("delta", ascending=False)
    risers = stats.head(n).index.tolist()
    fallers = stats[stats["delta"] < 0].tail(n).index.tolist()

    # Deduplicate while preserving order
    return list(dict.fromkeys(risers + fallers))


def _compute_rise_and_fall(df: pd.DataFrame) -> list[str]:
    """Return top 2 risers + top 2 fallers + 2 most volatile teams."""
    stats = _get_elo_deltas(df)
    if stats.empty:
        return[]

    by_delta = stats.sort_values("delta", ascending=False)
    risers = by_delta.head(2).index.tolist()
    fallers = by_delta[by_delta["delta"] < 0].tail(2).index.tolist()

    already_selected = set(risers + fallers)
    by_vol = stats.sort_values("stdev", ascending=False)
    volatile =[t for t in by_vol.index if t not in already_selected][:2]

    return risers + fallers + volatile


def _build_subtitle(df: pd.DataFrame, view_mode: str) -> str:
    """Build a one-line narrative subtitle for the chart."""
    stats = _get_elo_deltas(df)
    if stats.empty:
        return ""

    stats = stats.sort_values("delta", ascending=False)
    best_team = stats.index[0]
    worst_team = stats.index[-1]
    best_delta = stats.iloc[0]["delta"]
    worst_delta = stats.iloc[-1]["delta"]

    current_year = datetime.now().year
    is_current = view_mode == f"'{current_year % 100}"

    if is_current:
        period = "this season"
        rise_verb, fall_verb = "has risen", "has fallen"
    elif view_mode == "All":
        period = "all-time"
        rise_verb, fall_verb = "rose", "fell"
    else:
        period = f"the 20{view_mode.replace("'", '')} season"
        rise_verb, fall_verb = "rose", "fell"

    parts: list[str] =[]

    if best_delta > 5:
        parts.append(f"<b>{team_short(best_team)}</b> {rise_verb} <b>+{best_delta:.0f}</b>")
    if worst_delta < -5:
        parts.append(f"<b>{team_short(worst_team)}</b> {fall_verb} <b>{worst_delta:.0f}</b>")

    if not parts:
        return ""
    prefix = "So far " if is_current else "Across "
    return f"{prefix}{period}, " + " while ".join(parts) + " Elo"


def _extend_lines_to_max(df: pd.DataFrame, label: str = "Season end") -> pd.DataFrame:
    """Extend each team's line to the max date for label alignment."""
    df_valid = df.dropna(subset=["Elo"])
    if df_valid.empty:
        return df

    max_date = df_valid["Date"].max()
    last_idx = df_valid.groupby("Team")["Date"].idxmax()
    last_rows = df_valid.loc[last_idx].copy()

    needs_extension = last_rows[last_rows["Date"] < max_date].copy()
    if not needs_extension.empty:
        needs_extension["Date"] = max_date
        needs_extension["Context"] = label
        needs_extension["EloDelta"] = 0.0
        df = pd.concat([df, needs_extension], ignore_index=True)
        df = df.sort_values(["Team", "Date"]).reset_index(drop=True)

    return df


def _inject_gap_years(df: pd.DataFrame) -> pd.DataFrame:
    """Inject np.nan values for teams that miss an entire season to break chart lines."""
    if "Season" not in df.columns:
        return df

    seasons = sorted(df["Season"].dropna().unique())
    if len(seasons) <= 1:
        return df

    # Find the median date of each season to act as the injection anchor
    season_anchors = {s: df[df["Season"] == s]["Date"].median() for s in seasons}
    teams = df["Team"].unique()
    gap_rows = []

    for team in teams:
        team_df = df[df["Team"] == team]
        team_seasons = set(team_df["Season"].dropna().unique())

        if not team_seasons:
            continue

        min_idx = seasons.index(min(team_seasons))
        max_idx = seasons.index(max(team_seasons))

        for i in range(min_idx + 1, max_idx):
            s = seasons[i]
            if s not in team_seasons:
                gap_rows.append({
                    "Date": season_anchors[s],
                    "Team": team,
                    "Elo": np.nan,
                    "Context": "Inactive Season",
                    "EloDelta": 0.0,
                    "Season": s
                })

    if gap_rows:
        df = pd.concat([df, pd.DataFrame(gap_rows)], ignore_index=True)
        df = df.sort_values(["Team", "Date"]).reset_index(drop=True)

    return df


def _compress_offseasons(
    df: pd.DataFrame,
    gap_days: int = 14,
) -> tuple[pd.DataFrame, list[dict], list[float]]:
    """Remap dates so offseason gaps shrink to *gap_days* of visual space."""
    df = df.copy()

    # Move pre-season rows close to first match per season to prevent flatlines
    for season in df["Season"].dropna().unique():
        mask_season = df["Season"] == season
        is_match = mask_season & ~df["Context"].str.contains("Season start|Offseason", na=False) & df["Elo"].notna()

        match_dates = df.loc[is_match, "Date"]
        if match_dates.empty:
            continue

        first_match = match_dates.min()
        anchor = first_match - pd.Timedelta(days=1)
        is_pre = mask_season & (df["Context"].str.contains("Season start|Offseason", na=False) | df["Elo"].isna())
        df.loc[is_pre, "Date"] = anchor

    seasons = sorted(df["Season"].dropna().unique())
    ranges: list[dict] =[]

    for season in seasons:
        s_df = df[(df["Season"] == season) & (df["Elo"].notna())]
        if s_df.empty:
            continue
        ranges.append({
            "season": season,
            "start": s_df["Date"].min(),
            "end": s_df["Date"].max(),
        })

    if not ranges:
        df["PlotX"] = 0.0
        return df, [], []

    pieces: list[dict] =[]
    cursor = 0.0
    for i, sr in enumerate(ranges):
        dur = (sr["end"] - sr["start"]).total_seconds() / 86400
        pieces.append({
            "real_start": sr["start"],
            "real_end": sr["end"],
            "plot_start": cursor,
            "plot_end": cursor + dur,
            "season": sr["season"],
        })
        cursor += dur
        if i < len(ranges) - 1:
            cursor += gap_days

    # Vectorised date → PlotX mapping via season-keyed interval lookup
    piece_df = pd.DataFrame(pieces)
    season_map = df[["Season"]].merge(
        piece_df.rename(columns={"season": "Season"}),
        on="Season",
        how="left",
    )
    d = df["Date"].values
    rs = season_map["real_start"].values
    re = season_map["real_end"].values
    ps = season_map["plot_start"].values
    pe = season_map["plot_end"].values
    span = np.maximum((re - rs) / np.timedelta64(1, "s"), 1.0)
    frac = np.clip((d - rs) / np.timedelta64(1, "s") / span, 0.0, 1.0)
    plot_x = ps + frac * (pe - ps)
    # Rows with no matching season fall back to the first piece
    df["PlotX"] = np.where(season_map["plot_start"].isna(), pieces[0]["plot_start"] if pieces else 0.0, plot_x)

    labels = [{"label": p["season"], "center": (p["plot_start"] + p["plot_end"]) / 2} for p in pieces]
    boundaries = [(pieces[i]["plot_end"] + pieces[i + 1]["plot_start"]) / 2 for i in range(len(pieces) - 1)]

    return df, labels, boundaries


# -----------------------------------------------------------------------------
# Data Shaping Core
# -----------------------------------------------------------------------------

def _build_dataframe(
    engine: GrassrootsEloEngine,
    history: list[dict],
) -> tuple[pd.DataFrame, str, str, str, object, XMode]:
    """Build the Elo history dataframe from engine logs."""
    match_log = engine.match_log
    has_dates = any(m.get("date") for m in match_log)
    initial_elos = getattr(engine, "initial_elos", {})

    if has_dates:
        team_names_seen: set[str] = set()
        rows: list[dict] =[]
        prev_elo: dict[str, float] = {}

        parsed_dates =[
            pd.to_datetime(m["date"]) for m in match_log
            if m.get("date") and not pd.isna(pd.to_datetime(m["date"], errors="coerce"))
        ]
        baseline_date = (min(parsed_dates) - pd.Timedelta(days=1)) if parsed_dates else pd.Timestamp.now() - pd.Timedelta(days=180)

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

                is_home = (team_name == home)
                opp = away if is_home else home
                score_str = f"{hs}\u2013{as_}" if is_home else f"{as_}\u2013{hs}"

                # Result resolution
                if hs == as_: result = "D"
                elif (is_home and hs > as_) or (not is_home and as_ > hs): result = "W"
                else: result = "L"

                sign = "+" if delta > 0 else ""
                ctx = f"{result} {score_str} vs {team_short(opp)} ({sign}{delta:.0f})"

                rows.append({
                    "Date": dt, "Team": team_name, "Elo": elo_now, "Context": ctx, "EloDelta": delta,
                })
                prev_elo[team_name] = elo_now

        # Initialise baselines
        for team_name in team_names_seen:
            rows.append({
                "Date": baseline_date, "Team": team_name, "Elo": _baseline_elo(team_name, initial_elos),
                "Context": "Season start", "EloDelta": 0.0,
            })

        df = pd.DataFrame(rows).sort_values(["Team", "Date"]).reset_index(drop=True)
        df = _extend_lines_to_max(df, label="Current")
        return df, "Date:T", "Date", "Date", df["Date"].max(), "date"

    # Fallback to pure match count mode if dates aren't available
    team_match_count: dict[str, int] = {}
    rows =[]
    for idx, entry in enumerate(match_log):
        snapshot = history[idx]
        for team_name in (entry["home"], entry["away"]):
            team_match_count[team_name] = team_match_count.get(team_name, 0) + 1
            rows.append({
                "Match": team_match_count[team_name],
                "Team": team_name,
                "Elo": round(float(snapshot[team_name]), 1),
            })

    df = pd.DataFrame(rows)
    return df, "Match:Q", "Match", "Match", int(df["Match"].max()), "match"


# -----------------------------------------------------------------------------
# Chart Layout & Rendering
# -----------------------------------------------------------------------------

def _deconflict_labels(
    last_df: pd.DataFrame,
    y_lo: float,
    y_hi: float,
    chart_height: int = 480,
    min_px: int = 14,
) -> pd.DataFrame:
    """Nudge label positions vertically to reduce overlap."""
    if last_df.empty:
        return last_df.assign(LabelElo=last_df.get("Elo", 0.0))

    pts = last_df.sort_values("Elo", ascending=False).reset_index(drop=True)
    elo_span = y_hi - y_lo
    if elo_span <= 0:
        return pts.assign(LabelElo=pts["Elo"])

    min_gap = (min_px / chart_height) * elo_span
    positions = pts["Elo"].tolist()

    for i in range(1, len(positions)):
        if positions[i - 1] - positions[i] < min_gap:
            positions[i] = positions[i - 1] - min_gap

    # Clamp: shift all labels up if the lowest overflows below y_lo
    if positions and positions[-1] < y_lo:
        shift = y_lo - positions[-1]
        positions = [p + shift for p in positions]

    # Clamp: shift all labels down if the highest overflows above y_hi
    if positions and positions[0] > y_hi:
        shift = positions[0] - y_hi
        positions = [p - shift for p in positions]

    pts["LabelElo"] = positions
    return pts


def _render_chart(
    df: pd.DataFrame,
    x_field: str,
    x_title: str,
    x_col: str,
    max_x: object,
    x_mode: XMode,
    selected_teams: list[str],
    season_labels: list[dict] | None = None,
    season_boundaries: list | None = None,
    compressed: bool = False,
) -> None:
    """Render the responsive Altair Elo history line chart."""
    filtered = df[df["Team"].isin(selected_teams)].copy()
    if filtered.empty:
        st.warning("No data to display for the current selection.")
        return

    elo_min = float(filtered["Elo"].min(skipna=True))
    elo_max = float(filtered["Elo"].max(skipna=True))
    y_pad = max(30, (elo_max - elo_min) * 0.12)
    y_lo = max(0, round(min(elo_min - y_pad, 1450), -1))
    y_hi = round(max(elo_max + y_pad, 1550), -1)

    color_scale = alt.Scale(domain=selected_teams, range=[team_color(t) for t in selected_teams])

    highlight = alt.selection_point(fields=["Team"], on="pointerover", clear="pointerout", empty=True)
    hover_active = alt.selection_point(fields=["Team"], on="pointerover", clear="pointerout", empty=False)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", clear="pointerout",
        fields=["PlotX" if compressed else x_col, "Team"], empty=False
    )

    # Dynamic Axis Configuration
    if compressed:
        axis_config = alt.Axis(labels=False, ticks=False, domain=False, grid=False)
    else:
        axis_kwargs = {"grid": False, "labelPadding": 6}
        if x_mode == "date":
            axis_kwargs["format"] = "%d %b"
        else:
            axis_kwargs["tickMinStep"] = 1

        axis_config = alt.Axis(**axis_kwargs)

    x_enc_kwargs = {
        "title": None,
        "axis": axis_config
    }

    if compressed:
        # Always use the FULL dataset's PlotX range so every team appears
        # in correct position within the complete timeline, even when filtered
        # to a single team that only spans a subset of seasons.
        full_plot_min = float(df["PlotX"].min())
        full_plot_max = float(df["PlotX"].max())
        # Pad the right edge so a young current season has room to breathe
        full_plot_range = full_plot_max - full_plot_min
        if full_plot_range > 0:
            all_seasons = df["Season"].dropna().unique()
            last_season_pts = df.loc[df["Season"] == str(datetime.now().year)]
            if not last_season_pts.empty:
                season_span = float(last_season_pts["PlotX"].max()) - float(last_season_pts["PlotX"].min())
                avg_season = full_plot_range / max(len(all_seasons), 1)
                # If the current season spans less than 35% of an average season's
                # width, pad the right edge by 25% of an average season so the
                # chart doesn't look cramped at the start of a new year.
                if season_span < avg_season * 0.35:
                    full_plot_max += avg_season * 0.25
        x_enc_kwargs["scale"] = alt.Scale(domain=[full_plot_min, full_plot_max])
        x_enc = alt.X("PlotX:Q", **x_enc_kwargs)
    else:
        if x_mode == "date":
            # Use full dataset range so single-team filters stay in position
            date_min = pd.Timestamp(df["Date"].min())
            date_max = pd.Timestamp(df["Date"].max())
            # Pad forward if the season is young (< ~3 months of data)
            span_days = (date_max - date_min).days
            if 0 < span_days < 90:
                date_max = date_max + pd.Timedelta(days=max(45 - span_days, 14))
            x_enc_kwargs["scale"] = alt.Scale(domain=[date_min.isoformat(), date_max.isoformat()])
        else:
            x_enc_kwargs["scale"] = alt.Scale(domain=[0, max_x])
        x_enc = alt.X(x_field, **x_enc_kwargs)

    y_enc = alt.Y("Elo:Q", title=None, scale=alt.Scale(domain=[y_lo, y_hi]), axis=alt.Axis(gridColor="#e8ecf1", gridDash=[3, 3], gridOpacity=0.6, labelPadding=6))
    color_enc = alt.Color("Team:N", scale=color_scale, legend=None)

    layers =[]

    # Baseline 1500 line
    layers.append(
        alt.Chart(pd.DataFrame({"y": [float(BASE_ELO)]}))
        .mark_rule(strokeDash=[6, 4], strokeWidth=0.8, color="#cbd5e1")
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[y_lo, y_hi])))
    )

    # Season boundary markers
    if season_boundaries:
        b_df = pd.DataFrame({"bx": season_boundaries})
        bound_x = alt.X("bx:Q", scale=x_enc_kwargs["scale"]) if compressed else alt.X("bx:T")
        layers.append(
            alt.Chart(b_df).mark_rule(strokeDash=[4, 4], strokeWidth=0.8, color="#94a3b8").encode(x=bound_x)
        )

    # Core Lines & Hit Areas
    layers.append(
        alt.Chart(filtered)
        .mark_line(strokeWidth=3, opacity=0, interpolate="step-after")
        .encode(x=x_enc, y=y_enc, color=color_enc)
        .add_params(highlight, hover_active)
    )

    layers.append(
        alt.Chart(filtered)
        .mark_line(strokeWidth=1.8, interpolate="step-after")
        .encode(
            x=x_enc, y=y_enc, color=color_enc,
            opacity=alt.condition(highlight, alt.value(0.9), alt.value(0.08)),
            strokeWidth=alt.condition(highlight, alt.value(2.5), alt.value(1)),
        )
    )

    # Tooltips and hover dots
    tooltip_data =[
        alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
        alt.Tooltip("Team:N"),
        alt.Tooltip("Elo:Q", format=".0f"),
        alt.Tooltip("Context:N", title="Match")
    ] if (x_mode == "date" or compressed) and "Context" in filtered.columns else[
        alt.Tooltip(f"{x_col}:{'T' if x_mode == 'date' else 'Q'}", title=x_title),
        alt.Tooltip("Team:N"), alt.Tooltip("Elo:Q", format=".0f")
    ]

    layers.append(
        alt.Chart(filtered).mark_circle(size=40)
        .encode(x=x_enc, y=y_enc, color=color_enc, opacity=alt.value(0), tooltip=tooltip_data)
        .add_params(nearest)
    )

    layers.append(
        alt.Chart(filtered).mark_circle()
        .encode(
            x=x_enc, y=y_enc, color=color_enc,
            opacity=alt.condition(hover_active, alt.value(0.85), alt.value(0)),
            size=alt.condition(nearest, alt.value(55), alt.value(0)),
            tooltip=tooltip_data,
        )
    )

    # Label Deconfliction Setup
    last_idx_col = "PlotX" if compressed else x_col
    if x_mode == "date" or compressed:
        last_points = filtered.loc[filtered.dropna(subset=["Elo"]).groupby("Team")[last_idx_col].idxmax()].copy()
    else:
        last_points = filtered[filtered[x_col] == max_x].copy()

    last_points["Abbr"] = last_points["Team"].map(team_abbr)
    deconflicted = _deconflict_labels(last_points, y_lo, y_hi)
    label_y = alt.Y("LabelElo:Q", scale=alt.Scale(domain=[y_lo, y_hi]))

    # Label Lines & Text
    needs_leader = deconflicted[abs(deconflicted["LabelElo"] - deconflicted["Elo"]) > 1].copy()
    layers.append(
        alt.Chart(needs_leader).mark_rule(strokeWidth=0.7, strokeDash=[2, 2])
        .encode(
            x=x_enc, y=alt.Y("Elo:Q", scale=alt.Scale(domain=[y_lo, y_hi])), y2=alt.Y2("LabelElo:Q"),
            color=color_enc, opacity=alt.condition(highlight, alt.value(0.35), alt.value(0.08)),
        )
    )

    # Determine which teams are inactive (their last real data point is before the overall max)
    _overall_max = filtered["Date"].max() if "Date" in filtered.columns else None
    _active_teams = set()
    if _overall_max is not None:
        # ~120 days: treat teams whose last match is older than roughly one off-season as inactive
        _cutoff = _overall_max - pd.Timedelta(days=120)
        for _t, _grp in filtered.dropna(subset=["Elo"]).groupby("Team"):
            if _grp["Date"].max() >= _cutoff:
                _active_teams.add(_t)
    else:
        _active_teams = set(selected_teams)

    active_labels = deconflicted[deconflicted["Team"].isin(_active_teams)].copy()
    inactive_labels = deconflicted[~deconflicted["Team"].isin(_active_teams)].copy()

    # Active team labels (normal weight)
    layers.append(
        alt.Chart(active_labels).mark_text(align="left", dx=8, fontSize=9, fontWeight="bold")
        .encode(
            x=x_enc, y=label_y, text="Abbr:N", color=color_enc,
            opacity=alt.condition(hover_active, alt.value(0), alt.value(0.3)),
        )
    )
    layers.append(
        alt.Chart(active_labels).mark_text(align="left", dx=8, fontSize=11, fontWeight="bold")
        .encode(
            x=x_enc, y=label_y, text="Abbr:N", color=color_enc,
            opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
        )
    )

    # Inactive team labels (dimmed + italic)
    if not inactive_labels.empty:
        layers.append(
            alt.Chart(inactive_labels).mark_text(align="left", dx=8, fontSize=9, fontWeight="normal", fontStyle="italic")
            .encode(
                x=x_enc, y=label_y, text="Abbr:N", color=color_enc,
                opacity=alt.condition(hover_active, alt.value(0), alt.value(0.15)),
            )
        )
        layers.append(
            alt.Chart(inactive_labels).mark_text(align="left", dx=8, fontSize=11, fontWeight="normal", fontStyle="italic")
            .encode(
                x=x_enc, y=label_y, text="Abbr:N", color=color_enc,
                opacity=alt.condition(hover_active, alt.value(0.6), alt.value(0)),
            )
        )

    # Top Season Labels
    if season_labels:
        lbl_df = pd.DataFrame(season_labels)
        lbl_x = alt.X("center:Q", scale=x_enc_kwargs["scale"]) if compressed else alt.X("center:T")
        layers.append(
            alt.Chart(lbl_df).mark_text(fontSize=11, fontWeight="bold", color="#94a3b8", align="center")
            .encode(x=lbl_x, y=alt.value(10), text="label:N")
        )

    chart = (
        alt.layer(*layers)
        .resolve_scale(y="shared")
        .properties(height=480, padding={"left": 4, "top": 14, "right": 20, "bottom": 14})
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#94a3b8", labelFontSize=11, labelFont='"Source Sans Pro", "Segoe UI", sans-serif',
            tickColor="#e2e8f0", domainColor="#e2e8f0", titleColor="#94a3b8",
        )
    )

    st.altair_chart(chart, use_container_width=True, theme=None)


# -----------------------------------------------------------------------------
# Main View Rendering Controller
# -----------------------------------------------------------------------------

def render_elo_history_tab(
    engine: GrassrootsEloEngine,
    history: list[dict],
    league_table: list[Team],
    league_key: str,
) -> None:
    """Render the full-width Elo History tab."""
    if not history:
        st.info("No match history to display. Process some matches first.")
        return

    current_year = datetime.now().year
    grade = GRADE_MAP.get(league_key, "first_grade")
    hist_df = build_full_history(grade)

    has_history = not hist_df.empty
    hist_seasons = sorted(hist_df["Season"].dropna().unique()) if has_history else []

    # View Controls
    view_options = ([f"'{s[-2:]}" for s in hist_seasons] +[f"'{current_year % 100}", "All"]) if has_history else[f"'{current_year % 100}"]
    default_view = f"'{current_year % 100}"
    view_mode = st.pills("View", view_options, default=default_view, label_visibility="collapsed") or default_view

    season_labels, season_boundaries, compressed = None, None, False

    # Data Prep routing
    if view_mode == f"'{current_year % 100}":
        df, x_field, x_title, x_col, max_x, x_mode = _build_dataframe(engine, history)

    elif view_mode == "All":
        live_df, _, _, _, _, _ = _build_dataframe(engine, history)
        live_df["Season"] = str(current_year)

        regression_rows: list[dict] =[]
        initial_elos = getattr(engine, "initial_elos", {})

        if hist_seasons:
            last_hist_season = hist_seasons[-1]
            for t_name, start_elo in initial_elos.items():
                team_hist = hist_df[(hist_df["Team"] == t_name) & (hist_df["Season"] == last_hist_season) & (hist_df["Elo"].notna())]
                if team_hist.empty: continue

                last_elo = float(team_hist.sort_values("Date").iloc[-1]["Elo"])
                delta = start_elo - last_elo
                regression_rows.append({
                    "Date": pd.Timestamp(f"{current_year}-01-01"), "Team": t_name, "Elo": round(start_elo, 1),
                    "Context": f"Offseason regression ({delta:+.0f} Elo)", "EloDelta": round(delta, 1), "Season": str(current_year),
                })

        live_for_all = live_df[live_df["Context"] != "Season start"].copy()

        parts = [hist_df]
        if regression_rows: parts.append(pd.DataFrame(regression_rows))
        parts.append(live_for_all)

        df = pd.concat(parts, ignore_index=True).sort_values(["Team", "Date"]).reset_index(drop=True)
        df = _inject_gap_years(df)

        x_field, x_title, x_col, max_x, x_mode = "Date:T", "Date", "Date", df["Date"].max(), "date"
        df, season_labels, season_boundaries = _compress_offseasons(df)
        compressed = True

    else:
        year_str = "20" + view_mode.replace("'", "")
        df = hist_df[hist_df["Season"] == year_str].copy()

        if df.empty:
            st.warning(f"No historical data for {year_str}.")
            return

        match_rows = df[~df["Context"].str.contains("Season start|Offseason", na=False) & df["Elo"].notna()]
        if not match_rows.empty:
            df.loc[df["Context"].str.contains("Season start|Offseason", na=False) | df["Elo"].isna(), "Date"] = match_rows["Date"].min() - pd.Timedelta(days=1)

        df = _extend_lines_to_max(df)
        x_field, x_title, x_col, max_x, x_mode = "Date:T", "Date", "Date", df["Date"].max(), "date"

    # Team Filters
    is_all_time = (view_mode == "All")
    is_current_season = (view_mode == f"'{current_year % 100}")
    all_view_teams = sorted(df["Team"].dropna().unique().tolist())

    top6 = [t.name for t in league_table[:6]]
    biggest_movers = _compute_rise_and_fall(df) if is_all_time else _compute_biggest_movers(df)

    if is_current_season:
        filter_options = ["Biggest movers", "All teams", "Semi-finalists (top 6)", "Outside top 6", "Custom"]
    elif is_all_time:
        filter_options = ["Biggest movers", "Current top 6", "All teams", "Custom"]
    else:
        filter_options = ["Biggest movers", "All teams", "Custom"]
    filter_mode = st.pills("Filter", filter_options, default="Biggest movers", label_visibility="collapsed") or "Biggest movers"

    if filter_mode == "Biggest movers":
        selected_teams = biggest_movers
        # Build styled chip legend grouped by Risers / Fallers / Volatile
        _stats = _get_elo_deltas(df)
        if not _stats.empty and biggest_movers:
            _risers = []
            _fallers = []
            _wild = []
            _selected_set = set(biggest_movers)
            _by_delta = _stats.loc[_stats.index.isin(_selected_set)].sort_values("delta", ascending=False)
            _rise_names = set(_by_delta.head(3 if not is_all_time else 2).index)
            _fall_names = set(_by_delta[_by_delta["delta"] < 0].tail(3 if not is_all_time else 2).index)

            for t in biggest_movers:
                if t not in _stats.index:
                    continue
                _d = _stats.loc[t, "delta"]
                _sign = "+" if _d > 0 else ""
                if t in _rise_names:
                    _risers.append(_chip(t, f'color:#16a34a">{_sign}{_d:.0f}', "#f0fdf4", "#bbf7d0"))
                elif t in _fall_names:
                    _fallers.append(_chip(t, f'color:#dc2626">{_sign}{_d:.0f}', "#fef2f2", "#fecaca"))
                else:
                    _s = _stats.loc[t, "stdev"]
                    _wild.append(_chip(t, f'color:#d97706">σ\u2009{_s:.0f}', "#fffbeb", "#fde68a"))

            _sections = []
            if _risers:
                _sections.append(
                    f'<div style="display:flex;align-items:center;gap:2px;flex-wrap:wrap">'
                    f'<span style="color:#16a34a;font-weight:700;font-size:0.68rem;letter-spacing:0.05em;'
                    f'text-transform:uppercase;margin-right:2px">▲ Risers</span>{"".join(_risers)}</div>'
                )
            if _fallers:
                _sections.append(
                    f'<div style="display:flex;align-items:center;gap:2px;flex-wrap:wrap">'
                    f'<span style="color:#dc2626;font-weight:700;font-size:0.68rem;letter-spacing:0.05em;'
                    f'text-transform:uppercase;margin-right:2px">▼ Fallers</span>{"".join(_fallers)}</div>'
                )
            if _wild:
                _sections.append(
                    f'<div style="display:flex;align-items:center;gap:2px;flex-wrap:wrap">'
                    f'<span style="color:#d97706;font-weight:700;font-size:0.68rem;letter-spacing:0.05em;'
                    f'text-transform:uppercase;margin-right:2px">◆ Volatile</span>{"".join(_wild)}</div>'
                )
            if _sections:
                _html = (
                    '<div style="display:flex;flex-wrap:wrap;gap:6px 18px;padding:4px 0 6px 0;'
                    'align-items:center">' + ''.join(_sections) + '</div>'
                )
                st.markdown(_html, unsafe_allow_html=True)
    elif filter_mode == "All teams":
        selected_teams = all_view_teams
    elif filter_mode in ("Semi-finalists (top 6)", "Current top 6"):
        selected_teams =[t for t in top6 if t in set(all_view_teams)]
    elif filter_mode == "Outside top 6":
        bot6 = [t.name for t in league_table[6:]]
        selected_teams = [t for t in bot6 if t in set(all_view_teams)]
    else:
        _ms_key = f"custom_ms_{view_mode}"
        if _ms_key not in st.session_state:
            st.session_state[_ms_key] = list(all_view_teams)

        selected_teams = st.multiselect(
            "Select teams",
            options=all_view_teams,
            default=st.session_state[_ms_key],
            format_func=lambda t: f"{team_abbr(t)}  \u2014  {team_short(t)}",
            key=_ms_key,
            placeholder="Search or pick teams\u2026",
        )

    # --- Final Chart Rendering ---
    if selected_teams:
        subtitle = _build_subtitle(df, view_mode)
        if subtitle:
            st.markdown(
                f'<div style="background:#eff6ff;border-left:3px solid #3b82f6;'
                f'padding:0.55em 0.9em;border-radius:0 6px 6px 0;'
                f'color:#1e3a5f;font-size:0.95rem;margin:0.3em 0 0 0">{subtitle}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:0.8em 0 0.4em 0'>", unsafe_allow_html=True)

        _render_chart(
            df, x_field, x_title, x_col, max_x, x_mode, selected_teams,
            season_labels=season_labels,
            season_boundaries=season_boundaries,
            compressed=compressed,
        )
    else:
        st.warning("Select at least one team to display.")
