"""Predictions tab component."""

from __future__ import annotations

import streamlit as st

from dashboard.helpers import closeness_score, confidence_label, parse_fixture_dt
from dashboard.data import load_fixtures
from config.teams import team_badge_html, team_color, team_abbr, team_short
from engine.elo import GrassrootsEloEngine

# ── Presentation-layer xG override for extreme mismatches ───────────
# The core Skellam model intentionally caps xG to preserve global
# calibration.  For extreme mismatches (prob > 98%), the capped xG
# looks unrealistically low.  This cosmetic function inflates the
# *displayed* xG without touching the model's actual prediction.
_MISMATCH_PROB_THRESHOLD = 0.95   # favourite win prob to trigger
_MISMATCH_XG_KNEE = 5.0          # xG above this gets inflated


def _display_xg(xg_home: float, xg_away: float, prob_home: float, prob_away: float) -> tuple[float, float, bool]:
    """Return (display_home_xg, display_away_xg, is_overridden).

    Applies a probability-scaled inflation to the favoured side's xG
    when the match is an extreme mismatch, for display purposes only.
    The more extreme the win probability, the larger the inflation.
    """
    fav_prob = max(prob_home, prob_away)
    if fav_prob < _MISMATCH_PROB_THRESHOLD:
        return xg_home, xg_away, False

    # How extreme is this mismatch? 0 at threshold, 1 at 100%
    intensity = (fav_prob - _MISMATCH_PROB_THRESHOLD) / (1.0 - _MISMATCH_PROB_THRESHOLD)

    if prob_home >= prob_away and xg_home > _MISMATCH_XG_KNEE:
        excess = xg_home - _MISMATCH_XG_KNEE
        inflated = xg_home + excess * (1.0 + intensity * 4.0)
        return inflated, xg_away, True
    elif prob_away > prob_home and xg_away > _MISMATCH_XG_KNEE:
        excess = xg_away - _MISMATCH_XG_KNEE
        inflated = xg_away + excess * (1.0 + intensity * 4.0)
        return xg_home, inflated, True

    return xg_home, xg_away, False


def render_predictions_tab(
    engine: GrassrootsEloEngine,
    league_key: str,
    detected_round: int,
) -> None:
    """Render prediction cards for upcoming fixtures."""

    # ── Round selector ──────────────────────────────────────────────
    r_left, r_right = st.columns([1, 3])
    with r_left:
        round_number = st.number_input(
            "Round",
            min_value=1,
            max_value=30,
            value=detected_round,
            key="predictions_round",
        )
    with r_right:
        st.markdown("")  # vertical spacer to align

    try:
        raw_fixtures, _ = load_fixtures(league_key, round_number)
    except Exception:
        st.error("Could not fetch fixtures for this round.")
        return

    if not raw_fixtures:
        st.info("No upcoming fixtures found.")
        return

    # Mobile tweaks for prediction cards
    st.markdown('''<style>
    .pred-name-short { display: none; }
    .pred-name-full { display: inline; }
    .pred-ko { display: inline; }
    @media (max-width: 640px) {
        .pred-card { gap: 0 6px !important; }
        .pred-name-full { display: none; }
        .pred-name-short { display: inline; }
        .pred-delta { font-size: 0.58rem !important; padding: 1px 6px !important; }
        .pred-conf { font-size: 0.60rem !important; }
        .pred-ko { display: none; }
    }
    </style>''', unsafe_allow_html=True)

    sort_mode = st.pills(
        "Sort by",
        ["Closest matchup", "Most lopsided", "Kick-off time"],
        default="Closest matchup",
        label_visibility="collapsed",
    )
    if sort_mode is None:
        sort_mode = "Closest matchup"

    cards: list[tuple] = []
    for fix in raw_fixtures:
        attrs = fix["attributes"]
        if attrs.get("bye_flag"):
            continue
        home = GrassrootsEloEngine._shorten_name(attrs["home_team_name"])
        away = GrassrootsEloEngine._shorten_name(attrs["away_team_name"])
        result = engine.predict_match(home, away)
        home_team = engine.teams.get(home)
        away_team = engine.teams.get(away)
        cards.append((fix, home, away, result, home_team, away_team))

    if sort_mode == "Closest matchup":
        cards.sort(key=lambda c: closeness_score(c[3]["home_win"], c[3]["away_win"]))
    elif sort_mode == "Most lopsided":
        cards.sort(key=lambda c: closeness_score(c[3]["home_win"], c[3]["away_win"]), reverse=True)

    for fix, home, away, result, home_team, away_team in cards:
        attrs = fix["attributes"]
        ground = attrs.get("ground_name", "")

        hw = result["home_win"]
        dr = result["draw"]
        aw = result["away_win"]
        xg_h = result["xg_home"]
        xg_a = result["xg_away"]
        exp_gd = result["expected_gd"]

        # Cosmetic xG override for extreme mismatches (display only)
        disp_xg_h, disp_xg_a, xg_overridden = _display_xg(xg_h, xg_a, hw, aw)

        home_elo_val = home_team.elo if home_team else 0
        away_elo_val = away_team.elo if away_team else 0
        home_elo = f"{home_elo_val:.0f}" if home_team else "\u2014"
        away_elo = f"{away_elo_val:.0f}" if away_team else "\u2014"
        elo_delta = abs(int(home_elo_val - away_elo_val))

        home_bar = team_color(home)
        away_bar = team_color(away)
        if hw > aw:
            fav_pct = hw
            home_weight, away_weight = "700", "400"
        elif aw > hw:
            fav_pct = aw
            home_weight, away_weight = "400", "700"
        else:
            fav_pct = dr
            home_weight = away_weight = "600"

        conf_label, conf_colour = confidence_label(fav_pct)
        ko_time = parse_fixture_dt(attrs)

        hw_pct = max(hw * 100, 1)
        dr_pct = max(dr * 100, 1)
        aw_pct = max(aw * 100, 1)
        bar_total = hw_pct + dr_pct + aw_pct
        hw_pct = hw_pct / bar_total * 100
        dr_pct = dr_pct / bar_total * 100
        aw_pct = aw_pct / bar_total * 100

        meta_parts = []
        if ground:
            meta_parts.append(f'<span>{ground}</span>')
        if ko_time:
            meta_parts.append(f'<span class="pred-ko"> &middot; {ko_time}</span>')
        meta_str = "".join(meta_parts)

        with st.container(border=True):
            home_badge = team_badge_html(home, size=22)
            away_badge = team_badge_html(away, size=22)
            home_abbr = team_abbr(home)
            away_abbr = team_abbr(away)
            home_short = team_short(home)
            away_short = team_short(away)

            st.markdown(
                f'''<div class="pred-card" style="display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:0 12px; margin-bottom:8px">
                <div style="display:flex; align-items:center; justify-content:flex-end; gap:6px">
                    {home_badge}
                    <span class="pred-name-full" style="font-size:1.05rem; font-weight:{home_weight}">{home_short}</span>
                    <span class="pred-name-short" style="font-size:1.05rem; font-weight:{home_weight}">{home_abbr}</span>
                    <span style="font-size:0.75rem; color:#94a3b8">{home_elo}</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; gap:2px">
                    <span class="pred-delta" style="font-size:0.65rem; color:#94a3b8; background:#f1f5f9; padding:1px 8px; border-radius:8px; font-weight:500">\u0394 {elo_delta}</span>
                    <span class="pred-conf" style="font-size:0.68rem; font-weight:600; color:{conf_colour}; text-transform:uppercase; letter-spacing:0.5px">{conf_label}</span>
                </div>
                <div style="display:flex; align-items:center; justify-content:flex-start; gap:6px">
                    <span style="font-size:0.75rem; color:#94a3b8">{away_elo}</span>
                    <span class="pred-name-full" style="font-size:1.05rem; font-weight:{away_weight}">{away_short}</span>
                    <span class="pred-name-short" style="font-size:1.05rem; font-weight:{away_weight}">{away_abbr}</span>
                    {away_badge}
                </div>
            </div>''', unsafe_allow_html=True)

            st.markdown(
                f'''<div style="display:flex; width:100%; height:14px; border-radius:3px; overflow:hidden; margin-bottom:8px">
                <div title="{home} {hw*100:.0f}%" style="width:{hw_pct:.1f}%; background:{home_bar}"></div>
                <div title="Draw {dr*100:.0f}%" style="width:{dr_pct:.1f}%; background:#f1f5f9"></div>
                <div title="{away} {aw*100:.0f}%" style="width:{aw_pct:.1f}%; background:{away_bar}"></div>
            </div>''', unsafe_allow_html=True)

            xg_note = (' <span title="Adjusted for display — extreme mismatch. '
                       'Model win probability is the primary predictor." '
                       'style="cursor:help; opacity:0.6">ⓘ</span>') if xg_overridden else ""

            st.markdown(
                f'''<div class="pred-meta" style="display:grid; grid-template-columns:1fr auto 1fr; align-items:center; font-size:0.72rem; color:#94a3b8">
                <span>{hw*100:.0f}% &ndash; {dr*100:.0f}% &ndash; {aw*100:.0f}%</span>
                <span style="text-align:center">xG {disp_xg_h:.1f} &#8211; {disp_xg_a:.1f}{xg_note}</span>
                <span style="text-align:right">{meta_str}</span>
            </div>''', unsafe_allow_html=True)
