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

    use_priors = st.sidebar.toggle(
        "Carry-forward ratings",
        value=True,
        help="Use regressed end-of-season ratings as starting priors for active teams.",
    )

    st.sidebar.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    with st.sidebar.expander("How it works"):
        st.markdown(
            "📈 **Elo Rating**\n"
            "Every team has a single rating number. Beating a strong team increases your rating more than beating a weak one.\n\n"
            "🏟️ **Home Advantage**\n"
            "The home side gets a small Elo boost based on real historical data.\n\n"
            "🔄 **Season Carry-Over**\n"
            "Ratings carry over from last year, but are pulled 40% toward average to account for player turnover.\n\n"
            "🔮 **Predictions**\n"
            "The model estimates win probabilities and expected goals (xG) using a Skellam-Elo hybrid approach."
        )

    with st.sidebar.expander("Technical details"):
        st.markdown(
            "Empirically optimized against 2022–2025 historical data.\n\n"
            "**Key Parameters:**\n"
            "- **K-factor**: Ramps 30 → 35 over a team's first 10 games.\n"
            "- **Home Advantage**: +30 Elo points.\n"
            "- **Margin of Victory**: Asymmetric log dampening. Large wins yield diminishing returns.\n"
            "- **Opponent-Adjusted Rates**: Attack/defence adjusted for opponent quality.\n"
            "- **Skellam Predictions**: Win probability via Skellam distributions using blended xG.\n"
            "- **Carry-Forward**: 60% Elo retention across seasons.\n\n"
            "---\n"
            "<small style='color:#94a3b8'>Skellam-Elo Hybrid · 4 seasons · Data via Dribl API</small>",
            unsafe_allow_html=True,
        )

    return {
        "league_key": league_key,
        "use_priors": use_priors,
    }


def render_sidebar_stats(engine) -> None:
    """Render summary statistics in the sidebar below the league selector."""
    played_matches = engine.processed_matches

    st.sidebar.markdown(
        f"""
        <div style="background:var(--secondary-background-color, #f8fafc); border:1px solid var(--border-color, #e2e8f0); border-radius:8px; padding:12px; margin-bottom: 20px;">
            <div style="font-size:0.65rem; color:var(--text-color, #475569); opacity:0.8; text-transform:uppercase; font-weight:600; letter-spacing:0.5px;">League Overview</div>
            <div style="margin-top:6px;">
                <span style="display:block; font-size:0.88rem; font-weight:600; color:var(--text-color, #0f172a);">Matches Tracked</span>
                <span style="font-size:0.72rem; color:var(--text-color, #64748b); opacity:0.8;">{played_matches} completed matches</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
