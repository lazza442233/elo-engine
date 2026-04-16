# PRD: Elo History v2 — All-Time Snake Chart

**Status:** Approved
**Author:** Co-authored with Copilot
**Date:** 2026-04-16

---

## 1. Problem

The current Elo History tab only shows the **current season**. It's a nice chart,
but it misses the most interesting story the data can tell: how a team's strength
has evolved across multiple years of competition (2022–present).

Grassroots fans don't think in single-season windows. They think in narratives:
"We've been climbing for two years", "Cherrybrook have always been on top",
"Eastwood came out of nowhere in 2025". The current view can't tell those stories.

### What we're removing

- **Rising / Falling / Steady sidebar panel** — the movement chips will be
  rethought separately (possibly as part of the Rankings tab or header). They
  add clutter to the Elo History tab and aren't essential to the chart experience.

---

## 2. Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Let users see the full multi-season Elo journey for any team | All-time view loads correctly with 2022–current data |
| G2 | Default to current season (most relevant), with easy access to other views | First load shows 2026 data; switching views takes one tap |
| G3 | Season boundaries are visually obvious | Vertical markers + regression drops are clearly rendered |
| G4 | Works well on mobile | Chart readable at 375px; no horizontal scroll |
| G5 | No new data infrastructure | Uses existing `all_seasons.csv` + live API data |

---

## 3. User Experience

### 3.1 View Modes

A **segmented control** (Streamlit `st.pills`) at the top of the tab with options:

| Option | What it shows | X-axis |
|--------|--------------|--------|
| **2026** (default) | Current season only — identical to today's chart but without the sidebar panel | Date |
| **2025** | Full 2025 season (from `all_seasons.csv`) | Date |
| **2024** | Full 2024 season | Date |
| **2023** | Full 2023 season | Date |
| **2022** | Full 2022 season | Date |
| **All-Time** | 2022 through current, including inter-season regression | Date |

> **Why pills, not a dropdown?** The number of options is small (6) and fixed.
> Pills give a visible, one-tap selection that works well on desktop and mobile.
> Each year label is short (4 chars), so they fit comfortably.

### 3.2 Team Filtering

Retain the existing filter pills row below the view mode selector:

- **Biggest movers** (default) — top 3 risers + top 3 fallers within the selected view window
- **All teams**
- **Semi-finalists (top 6)**
- **Outside top 6**
- **Custom** — multi-select dropdown

When in All-Time mode, "Biggest movers" should be computed over the entire
multi-season span (comparing start of 2022 to current). Teams that only appear
in some seasons should still be plottable.

> **Important:** "Biggest movers" must use **net change** (`Elo at end of window
> − Elo at start of window`), not sum of absolute movements. A team that gains
> 300 in 2023 and loses 300 in 2024 is flat, not a "big mover".

### 3.3 All-Time View — Season Boundaries

When the All-Time view is selected, the chart should clearly communicate
inter-season transitions:

- **Vertical dashed lines** at each season boundary (between last match of
  season N and first match of season N+1)
- **Season labels** ("2022", "2023", etc.) at the top of the chart, centred
  within each season's x-range
- **Regression drops** are rendered as explicit offseason data points (see 3.4
  for the date-anchoring rule) so the step-down happens cleanly in the gap
  between seasons, not on Match Day 1

### 3.4 Chart Behaviour (all views)

Carried over from v1 (keep these):
- Step-after line interpolation
- Hover to spotlight a single team (others fade to ~8% opacity)
- Nearest-point tooltip showing: date, team, Elo, match context (result, score, opponent, Δ)
- Deconflicted endpoint labels (team abbreviations)
- 1500 baseline rule (dashed)
- Team colours from `config.teams`

New:
- **Season boundary markers** (All-Time only): vertical dashed lines + top labels
- **Offseason regression anchoring**: When inserting the regressed Elo point
  between seasons, anchor it to a fixed offseason date (Jan 1 of the new year).
  Because `step-after` interpolation draws a horizontal line from the last data
  point until the next one, attaching the regression to the first match date
  would visually imply the team regressed on Match Day 1. Placing it in the
  offseason gap keeps the drop cleanly between the vertical boundary markers.
  Tooltip for this point: `"Offseason regression (−XX Elo)"`.
- **Gap-year line breaks**: If a team is absent for an entire season (e.g.
  plays in 2022, absent in 2023, returns in 2024), insert a `NaN` Elo row for
  that team in the gap. Altair naturally breaks lines at null values, preventing
  a misleading straight line drawn through a season the team didn't play.
- **Responsive height**: 480px desktop, 320px on mobile (≤640px)

### 3.5 Layout

**Single column** — the full width of the content area. No sidebar panel.
The movement panel is being removed from this tab.

```
┌──────────────────────────────────────────────────┐
│  ['22] ['23] ['24] ['25] ['26] [All]             │  ← view mode pills
│  [Biggest movers] [All] [Top 6] [...] [Custom]   │  ← team filter pills
├──────────────────────────────────────────────────┤
│                                                  │
│              Elo snake chart                      │
│              (full width)                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 3.6 Mobile Behaviour

- Pills wrap naturally (Streamlit handles this)
- Chart height reduces to 320px
- Endpoint labels use shorter abbreviations (already handled by `team_abbr`)
- Tooltip works via tap (Altair default on touch devices)

---

## 4. Data Architecture

### 4.1 Single Cached History Function

A single cached function builds the **complete** historical DataFrame once:

```python
@st.cache_data(ttl=3600)
def _build_full_history(grade: str) -> pd.DataFrame:
    """
    Walk-forward sim over all_seasons.csv for the given grade.
    Returns a DataFrame with columns:
        Date, Team, Elo, Context, EloDelta, Season
    Includes:
        - Season-start baseline rows (all teams at their opening Elo)
        - Offseason regression rows (anchored to Jan 1 of the new year)
        - NaN rows for gap-year teams
    """
```

Keyed on `grade` only. Historical data is static, so TTL can be long (3600s).
The DataFrame is ~1,500 rows — trivially small.

### 4.2 View Slicing (No Redundant Sims)

When the user selects a specific year pill (e.g. `'24`), **do not re-run the
simulation**. Instead, apply a Pandas `.loc` date filter to the cached All-Time
DataFrame:

```python
if view_mode == "All":
    df = full_history_df
else:
    year = int(view_mode)  # e.g. 2024
    df = full_history_df.loc[full_history_df["Season"] == str(year)]
```

Slicing a ~1,500-row DataFrame takes microseconds, so pill-switching feels
instant with zero additional compute or cache entries.

### 4.3 Current Season (2026)

Uses the live engine's `elo_history` and `match_log` as today. When the user
selects `'26`, this is the sole data source (API-fresh).

When `All` is selected, the 2026 live DataFrame is appended to the cached
historical DataFrame. An offseason regression row is inserted at Jan 1 2026
between the historical and live data.

### 4.4 Offseason Regression Rows

At each season boundary, insert one row per team:
- **Date:** Jan 1 of the new season year (e.g. `2023-01-01` between 2022→2023)
- **Elo:** `BASE_ELO + (end_of_season_elo - BASE_ELO) * (1 - REGRESSION_FACTOR)`
- **Context:** `"Offseason regression (−XX Elo)"`
- **Season:** The new season year

This ensures the step-down renders cleanly in the offseason gap.

### 4.5 Gap-Year NaN Rows

For any team present in season N but absent in season N+1, insert a single row
with `Elo = NaN` dated to the midpoint of the gap. Altair breaks lines at null
values, preventing a misleading connection through a season the team didn't play.

### 4.4 Grade Filtering

The dashboard already has a league selector (sidebar). The Elo History tab
should respect it:
- `prem-men` → `first_grade` matches from `all_seasons.csv`
- `prem-res` → `reserve_grade` matches from `all_seasons.csv`

---

## 5. Implementation Plan

### Phase 1: Data Layer

| # | Task | Detail |
|---|------|--------|
| 1.1 | Create `_build_full_history(grade)` | Single cached function. Walk-forward sim on `all_seasons.csv`, capturing per-match snapshots. Returns complete DataFrame with `Date, Team, Elo, Context, EloDelta, Season` columns. |
| 1.2 | Insert offseason regression rows | At each season boundary, add regressed-Elo rows dated Jan 1 of the new year. Tooltip: `"Offseason regression (−XX Elo)"`. |
| 1.3 | Insert gap-year NaN rows | For teams absent in a season, insert `NaN` Elo row to break the Altair line. |
| 1.4 | Map league key to grade | `prem-men` → `first_grade`, `prem-res` → `reserve_grade`. |
| 1.5 | View slicing | Year pills filter the cached DataFrame via `df.loc[df["Season"] == year]`. `All` returns the full DataFrame. `'26` uses the live engine data. |

### Phase 2: Chart Updates

| # | Task | Detail |
|---|------|--------|
| 2.1 | Add season boundary vertical rules | Altair `mark_rule` layer, positioned at boundary dates, styled as dashed lines. Only rendered in All-Time mode. |
| 2.2 | Add season labels | Altair `mark_text` layer at top of chart, one per season. |
| 2.3 | Responsive chart height | Use Streamlit's column width detection or a simple media query approach. |

### Phase 3: UI Changes

| # | Task | Detail |
|---|------|--------|
| 3.1 | Add view mode pills | `st.pills` with `["'22", "'23", "'24", "'25", "'26", "All"]`, default `"'26"`. |
| 3.2 | Remove sidebar movement panel | Delete `_render_sidebar_panel()` call and the `col_chart, col_panel = st.columns(...)` split. Chart goes full-width. |
| 3.3 | Rewire `render_elo_history_tab()` | Based on selected view mode, call `_build_dataframe()` for `'26`, or slice `_build_full_history()` for historical years / All. Pass result to `_render_chart()`. |
| 3.4 | Update team filter "Biggest movers" | Use **net change** (end − start of selected window). For All-Time, compare start of 2022 to current. |

### Phase 4: Cleanup

| # | Task | Detail |
|---|------|--------|
| 4.1 | Remove `_compute_movement()` | No longer needed once sidebar panel is removed. |
| 4.2 | Remove `_render_sidebar_panel()` | And associated `_movement_row_html()`. |
| 4.3 | Remove `_movement_bucket()` helper | Dead code after panel removal. |

---

## 6. Edge Cases

| Case | Handling |
|------|----------|
| Team only exists in some seasons | Line starts/ends at their first/last match. NaN row inserted at gap boundaries to break the line (see 4.5). |
| Team changes name between seasons | Not applicable — names are consistent in `all_seasons.csv`. |
| No historical data file | Gracefully fall back to current-season-only view. Disable historical pills. |
| Very early in 2026 season (few matches) | Chart still renders; endpoint labels still work. |
| All-Time with 18 teams on mobile | "Biggest movers" default keeps it manageable (6 lines). Full "All teams" is available but user's choice. |

---

## 7. Out of Scope

- **Movement chips redesign** — will be addressed separately, potentially on the Rankings tab
- **Season-over-season comparison view** (e.g. overlaying two seasons) — future enhancement
- **Reserve grade cross-referencing** — each grade is independent
- **Exporting/sharing** — not in this iteration

---

## 8. Resolved Questions

| # | Question | Decision | Rationale |
|---|----------|----------|----------|
| Q1 | Default view: `'26` or `All`? | **`'26`** | Grassroots fans check dashboards weekly for "what happened on the weekend". All-Time is a secondary exploratory feature. Don't punish weekly active users with a zoomed-out chart. |
| Q2 | Show regression factor on chart? | **No** | Too technical. Offseason regression points get a descriptive tooltip instead: `"Offseason regression (−XX Elo)"`. |
| Q3 | Include finals in historical views? | **Yes** | Finals are the narrative climax and carry high Elo stakes. Omitting them would make prior standings inaccurate. |
| Q4 | Full year labels or short? | **Short: `'22`, `'23`, `'24`, `'25`, `'26`, `All`** | Screen real estate at 375px is premium. Short labels guarantee a single-row pill bar. |
