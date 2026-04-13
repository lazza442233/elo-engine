"""Sidebar configuration component."""

from __future__ import annotations

import streamlit as st

from config.constants import BASE_ELO, DEFAULT_LEAGUE, LEAGUES

_PRIORS_PATHS = {
    "prem-men": "data/end_of_season_elos_first_grade.json",
    "prem-res": "data/end_of_season_elos_reserve_grade.json",
}


def render_sidebar() -> dict:
    """Render the sidebar and return user selections.

    Returns a dict with keys: league_key, manual_round, use_priors.
    """
    st.sidebar.markdown(
        '<div style="padding:8px 0 4px">'
        '<span style="font-size:1.5rem; font-weight:800; letter-spacing:-0.5px">'
        'Elo Engine</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<p style="font-size:0.75rem; color:#94a3b8; margin:-4px 0 12px; line-height:1.4">'
        'Elo ratings &amp; match predictions for grassroots football</p>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p style="font-size:0.72rem; color:#64748b; text-transform:uppercase; '
        'letter-spacing:0.6px; font-weight:600; margin-bottom:4px">League</p>',
        unsafe_allow_html=True,
    )
    league_key = st.sidebar.selectbox(
        "League",
        options=list(LEAGUES.keys()),
        format_func=lambda k: LEAGUES[k]["name"],
        index=list(LEAGUES.keys()).index(DEFAULT_LEAGUE),
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.sidebar.markdown(
        '<p style="font-size:0.72rem; color:#64748b; text-transform:uppercase; '
        'letter-spacing:0.6px; font-weight:600; margin-bottom:4px">Round</p>',
        unsafe_allow_html=True,
    )
    round_mode = st.sidebar.radio(
        "Round selection", ["Auto-detect", "Manual"],
        horizontal=True, label_visibility="collapsed",
    )
    manual_round = None
    if round_mode == "Manual":
        if "manual_round_input" not in st.session_state:
            st.session_state["manual_round_input"] = st.session_state.get("detected_round", 1)
        manual_round = st.sidebar.number_input(
            "Round number", min_value=1, max_value=30, key="manual_round_input",
            label_visibility="collapsed",
        )

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    from pathlib import Path
    priors_path = _PRIORS_PATHS.get(league_key, _PRIORS_PATHS["prem-men"])
    priors_available = Path(priors_path).exists()

    st.sidebar.markdown(
        '<p style="font-size:0.72rem; color:#64748b; text-transform:uppercase; '
        'letter-spacing:0.6px; font-weight:600; margin-bottom:4px">Season Priors</p>',
        unsafe_allow_html=True,
    )
    use_priors = st.sidebar.toggle(
        "Carry forward ratings",
        value=priors_available,
        disabled=not priors_available,
        help="Seed team ratings from last season's end-of-season Elo (regressed 20% toward 1500). "
             "Improves early-season accuracy." if priors_available
             else "No priors file found. Run `python main.py --export-ratings` at end of season.",
    )

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    with st.sidebar.expander("How it works"):
        st.markdown(
            "Ratings update after every match. The size of the rating "
            "change is determined by the match result, the margin of "
            "victory, and the pre-match odds.\n\n"
            "All parameters have been **empirically optimized** "
            "against four seasons of historical data (2022–2025) using "
            "a walk-forward grid search with asymmetric MoV dampening "
            "and opponent-adjusted rates. The model achieves a Brier "
            "score of **0.496** on the 2025 holdout season (random "
            "guessing = 0.67).\n\n"
            "- **Early-Season Calibration**: For a team's first 10 "
            "games, their rating is intentionally more volatile "
            "(K-factor 40 → 30). This allows the model to quickly "
            "find their true strength. Afterwards, ratings stabilise.\n"
            "- **Home-Field Advantage**: +50 Elo points. Analysis of "
            "four historical seasons shows a strong home advantage in "
            "this league — likely due to familiar grounds and local "
            "supporter presence.\n"
            "- **Margin of Victory**: A dominant 5-0 victory will "
            "cause a significantly larger rating shift than a narrow "
            "1-0 win, with diminishing returns for extreme blowouts. "
            "Blowouts between mismatched teams are dampened so "
            "expected results don't over-inflate ratings.\n"
            "- **Opponent-Adjusted Rates**: Each team's attack and "
            "defence rates are adjusted for the quality of opponents "
            "faced, so scoring against a strong defence counts more.\n"
            "- **Predictive Model**: Match odds are calculated via a "
            "Skellam distribution using per-team expected goals (xG), "
            "blended from opponent-adjusted rates and Elo-derived "
            "expectations. For extreme mismatches, the displayed xG "
            "is adjusted for realism (marked with ⓘ) while the "
            "underlying win probability remains the primary predictor.\n"
            "- **Adaptive League Average**: The league-wide goals "
            "per game is computed dynamically from processed matches "
            "rather than using a fixed constant, allowing the model "
            "to self-correct for changes in league scoring trends.\n"
            "- **Carry-Forward Ratings**: When enabled, teams retain "
            "80% of their Elo delta from the previous season, "
            "accounting for roster turnover while preserving "
            "established team strength. New entrants to the "
            "competition are seeded based on their expected level.\n"
            f"- **Baseline Rating**: Without carry-forward, all "
            f"teams start at {BASE_ELO}.\n\n"
            "*Ratings are provisional for the first ~10 rounds while "
            "the system calibrates.*\n\n"
            "---\n\n"
            "<small style='color:#94a3b8'>Skellam-Elo Hybrid Model · Data via Dribl API</small>",
            unsafe_allow_html=True,
        )

    return {
        "league_key": league_key,
        "manual_round": manual_round,
        "use_priors": use_priors,
    }
