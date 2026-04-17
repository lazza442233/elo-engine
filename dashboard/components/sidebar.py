"""Sidebar configuration component."""

from __future__ import annotations

import streamlit as st

from config.constants import DEFAULT_LEAGUE, LEAGUES


def render_sidebar() -> dict:
    """Render the sidebar and return user selections.

    Returns a dict with keys: league_key.
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

    _PILL_LABELS = {k: cfg["name"].replace("Premier League ", "") for k, cfg in LEAGUES.items()}
    _LABEL_TO_KEY = {v: k for k, v in _PILL_LABELS.items()}
    _default = _PILL_LABELS[DEFAULT_LEAGUE]

    def _enforce_selection():
        if st.session_state["league_pills"] is None:
            st.session_state["league_pills"] = _default

    league_label = st.sidebar.pills(
        "League",
        options=list(_PILL_LABELS.values()),
        default=_default,
        key="league_pills",
        on_change=_enforce_selection,
    )
    if league_label is None:
        league_label = _default
    league_key = _LABEL_TO_KEY[league_label]

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    with st.sidebar.expander("How it works"):
        st.markdown(
            "Every team has an **Elo rating** — a single number that "
            "represents how strong they are right now. When a team "
            "wins, their rating goes up and the loser's goes down. "
            "A bigger win means a bigger change.\n\n"
            "Beating a strong team is worth more than beating a weak "
            "one, and the home side gets a small boost because home "
            "advantage is real in this league.\n\n"
            "At the start of a new season, ratings carry over from "
            "last year (with a moderate pull back toward average to "
            "account for player movement and roster turnover).\n\n"
            "**Predictions** use each team's rating — and their "
            "attacking and defensive record — to estimate the most "
            "likely scoreline and win probability for every upcoming "
            "match.",
        )

    with st.sidebar.expander("Technical details"):
        st.markdown(
            "All parameters have been **empirically optimized** "
            "against four seasons of historical data (2022–2025) using "
            "a 2,000-sample Latin Hypercube search followed by a "
            "refined grid, with 3-fold expanding-window walk-forward "
            "validation. The model achieves a Brier score of **0.479** "
            "on the 2025 holdout season (random guessing ≈ 0.67).\n\n"
            "- **K-factor**: Linearly ramps from 30 → 35 over "
            "a team's first 10 games (averaged across both teams). "
            "Game counts carry across seasons, so only genuine "
            "newcomers use the lower initial K.\n"
            "- **Home-Field Advantage**: +30 Elo points, derived from "
            "four seasons of observed home win rates (~47%).\n"
            "- **Margin of Victory**: Asymmetric log dampening "
            "(C₁=0.001, C₂=2.0) — large wins shift ratings more, "
            "but with diminishing returns. Draws carry a mild "
            "total-goals signal so high-scoring draws are "
            "distinguished from goalless ones.\n"
            "- **Opponent-Adjusted Rates**: Per-team attack and "
            "defence rates are adjusted for opponent quality. Scoring "
            "against a strong defence counts more.\n"
            "- **Skellam Predictions**: Match odds via a Skellam "
            "distribution using per-team xG blended 50/50 from "
            "opponent-adjusted rates and Elo-derived expectations "
            "(75 Elo pts ≈ 1 goal difference). Display xG is inflated "
            "for extreme mismatches (marked with ⓘ); win probability "
            "is the primary predictor.\n"
            "- **Carry-Forward**: 60% Elo delta retention across "
            "seasons (40% regression toward 1500). Newcomers seeded "
            "at an anchor based on their expected competitive level. "
            "League average is conserved at 1500.\n\n"
            "---\n\n"
            "<small style='color:#94a3b8'>Skellam-Elo Hybrid · "
            "374 matches · 4 seasons · Data via Dribl API</small>",
            unsafe_allow_html=True,
        )

    return {
        "league_key": league_key,
    }
