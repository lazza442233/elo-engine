"""Data loading and engine construction with Streamlit caching."""

from __future__ import annotations

import streamlit as st

from api.client import detect_next_round, fetch_dribl_data
from config.constants import _build_api_url, fixtures_url
from engine.elo import GrassrootsEloEngine

PRIORS_PATHS = {
    "prem-men": "data/end_of_season_elos_first_grade.json",
    "prem-res": "data/end_of_season_elos_reserve_grade.json",
}


@st.cache_data(ttl=300, show_spinner="Fetching match data...")
def load_matches(league_key: str):
    """Fetch completed match results from the Dribl API."""
    api_url = _build_api_url(league_key)
    return fetch_dribl_data(api_url)


@st.cache_data(ttl=300, show_spinner="Fetching fixtures...")
def load_fixtures(league_key: str, round_number: int | None):
    """Fetch fixtures for a specific round, or auto-detect the next round."""
    if round_number is not None:
        raw_fixtures = fetch_dribl_data(fixtures_url(round_number, league_key))
        return raw_fixtures, round_number
    else:
        detected_round, raw_fixtures = detect_next_round(league_key=league_key)
        return raw_fixtures, detected_round

@st.cache_data(ttl=300, show_spinner="Processing matches...")
def build_engine(league_key: str, _raw_matches: list, use_priors: bool = False):
    """Process matches and return engine plus derived data."""
    engine = GrassrootsEloEngine()

    if use_priors:
        priors_path = PRIORS_PATHS.get(league_key, PRIORS_PATHS["prem-men"])
        try:
            priors = GrassrootsEloEngine.load_priors_from_file(priors_path)
            # Only inject priors for teams in the current season
            active = _active_teams(_raw_matches)
            priors = {k: v for k, v in priors.items() if k in active}
            engine.inject_priors(priors, quiet=True)
        except FileNotFoundError:
            pass

    engine.process_matches(_raw_matches, quiet=True)

    history = engine.elo_history
    team_names = sorted(engine.teams.keys())

    return history, team_names, engine


def _active_teams(raw_matches: list) -> set[str]:
    """Extract the set of team names present in the current season's data."""
    names = set()
    for match in raw_matches:
        attrs = match["attributes"]
        names.add(GrassrootsEloEngine._shorten_name(attrs["home_team_name"]))
        names.add(GrassrootsEloEngine._shorten_name(attrs["away_team_name"]))
    return names


def compute_league_state(engine: GrassrootsEloEngine) -> dict:
    """Derive league table, Elo rankings, and rank map from the engine.

    Returns a dict with keys: elo_ranked, elo_rank_map, league_table.
    """
    elo_ranked = engine.standings()
    elo_rank_map = {t.name: i for i, t in enumerate(elo_ranked, 1)}
    league_table = sorted(
        elo_ranked,
        key=lambda t: (t.points, t.gd, t.gf),
        reverse=True,
    )
    return {
        "elo_ranked": elo_ranked,
        "elo_rank_map": elo_rank_map,
        "league_table": league_table,
    }
