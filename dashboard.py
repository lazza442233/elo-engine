"""
Streamlit Web Dashboard for the Grassroots Elo Engine.

Run with:  streamlit run dashboard.py
"""

import streamlit as st

from config.constants import LEAGUES
from dashboard.components.elo_history import render_elo_history_tab
from dashboard.components.header import render_header
from dashboard.components.predictions import render_predictions_tab
from dashboard.components.rankings import render_rankings_tab
from dashboard.components.sidebar import render_sidebar
from dashboard.data import build_engine, compute_league_state, load_fixtures, load_matches

st.set_page_config(page_title="Elo Engine", page_icon="E", layout="wide")

st.markdown("""<style>
[data-testid="stPills"] button[aria-checked="true"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
}
/* Hide Streamlit deploy button & decoration, keep sidebar toggle */
[data-testid="stToolbar"] [data-testid="stToolbarActions"] > div:has([data-testid="stBaseButton-header"]) { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
/* Tighten default block-container padding */
.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; }
/* Mobile: tighten main content padding */
@media (max-width: 640px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    h1 { font-size: 1.6rem !important; }
}
</style>""", unsafe_allow_html=True)

# Sidebar
sidebar_cfg = render_sidebar()
league_key = sidebar_cfg["league_key"]
league_cfg = LEAGUES[league_key]

# Data loading
try:
    raw_matches = load_matches(league_key)
    raw_fixtures, detected_round = load_fixtures(league_key, round_number=None)
    st.session_state["detected_round"] = detected_round
except Exception as exc:
    st.error(f"Could not fetch data from Dribl API: {exc}")
    st.stop()

# Engine
history, team_names, engine = build_engine(
    league_key, raw_matches, use_priors=True,
)
state = compute_league_state(engine)

# Header
render_header(
    engine=engine,
    league_table=state["league_table"],
    raw_fixtures=raw_fixtures,
    detected_round=detected_round,
    league_name=league_cfg["name"],
)

# Tabs
tab_rank, tab_pred, tab_hist = st.tabs(["Rankings", "Predictions", "Elo History"])

with tab_rank:
    render_rankings_tab(
        league_table=state["league_table"],
        elo_rank_map=state["elo_rank_map"],
        match_log=engine.match_log,
    )

with tab_pred:
    render_predictions_tab(
        engine=engine,
        league_key=league_key,
        detected_round=detected_round,
    )

with tab_hist:
    render_elo_history_tab(
        engine=engine,
        history=history,
        team_names=team_names,
        elo_ranked=state["elo_ranked"],
        league_table=state["league_table"],
        league_key=league_key,
    )
