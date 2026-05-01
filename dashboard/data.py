"""Data loading and engine construction with Streamlit caching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from api.client import detect_next_round, fetch_dribl_data
from config.constants import BASE_ELO, PRIOR_REGRESSION_FACTOR, _build_api_url, fixtures_url
from config.teams import team_short
from engine.elo import GrassrootsEloEngine
from engine.match_record import normalize_match_records

PRIORS_PATHS = {
    "prem-men": "data/end_of_season_elos_first_grade.json",
    "prem-res": "data/end_of_season_elos_reserve_grade.json",
}

GRADE_MAP = {"prem-men": "first_grade", "prem-res": "reserve_grade"}


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
    match_records, _ = normalize_match_records(_raw_matches)

    if use_priors:
        priors_path = PRIORS_PATHS.get(league_key, PRIORS_PATHS["prem-men"])
        try:
            priors = GrassrootsEloEngine.load_priors_from_file(priors_path)
            # Only inject priors for teams in the current season
            active = {m.home_team for m in match_records} | {m.away_team for m in match_records}
            priors = {k: v for k, v in priors.items() if k in active}
            engine.inject_priors(priors, quiet=True)
        except FileNotFoundError:
            pass

    engine.process_matches(match_records, quiet=True)

    history = engine.elo_history
    team_names = sorted(engine.teams.keys())

    return history, team_names, engine

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


def _append_history_rows(
    rows: list[dict],
    match_log: list[dict],
    snapshots: list[dict[str, float]],
    pre_season_snapshot: dict[str, float],
    season: str,
):
    """Convert replayed match logs plus Elo snapshots into chart rows."""
    previous_snapshot = dict(pre_season_snapshot)

    for match_entry, snapshot in zip(match_log, snapshots):
        dt = pd.to_datetime(match_entry["date"])
        home = match_entry["home"]
        away = match_entry["away"]
        home_score = match_entry["home_score"]
        away_score = match_entry["away_score"]

        h_before = previous_snapshot.get(home, float(BASE_ELO))
        a_before = previous_snapshot.get(away, float(BASE_ELO))
        h_after = snapshot.get(home, float(BASE_ELO))
        a_after = snapshot.get(away, float(BASE_ELO))
        h_delta = h_after - h_before
        a_delta = a_after - a_before

        h_res = "W" if home_score > away_score else ("D" if home_score == away_score else "L")
        a_res = "W" if away_score > home_score else ("D" if home_score == away_score else "L")
        h_sign = "+" if h_delta > 0 else ""
        a_sign = "+" if a_delta > 0 else ""

        rows.append({
            "Date": dt,
            "Team": home,
            "Elo": round(h_after, 1),
            "Context": f"{h_res} {home_score}\u2013{away_score} vs {team_short(away)} ({h_sign}{h_delta:.0f})",
            "EloDelta": round(h_delta, 1),
            "Season": season,
        })
        rows.append({
            "Date": dt,
            "Team": away,
            "Elo": round(a_after, 1),
            "Context": f"{a_res} {away_score}\u2013{home_score} vs {team_short(home)} ({a_sign}{a_delta:.0f})",
            "EloDelta": round(a_delta, 1),
            "Season": season,
        })

        previous_snapshot = snapshot


# ---------------------------------------------------------------------------
# Historical full-history builder
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Building historical ratings...")
def build_full_history(grade: str) -> pd.DataFrame:
    """Walk-forward sim over all_seasons.csv for the given grade.

    Returns a DataFrame with columns:
        Date, Team, Elo, Context, EloDelta, Season
    Includes offseason regression rows (anchored to Jan 1) and gap-year
    NaN rows for absent teams.
    """
    csv_path = Path("data/processed/all_seasons.csv")
    if not csv_path.exists():
        return pd.DataFrame(columns=["Date", "Team", "Elo", "Context", "EloDelta", "Season"])

    all_matches = pd.read_csv(csv_path)
    grade_matches = all_matches[all_matches["grade"] == grade].copy()
    grade_matches = grade_matches[grade_matches["status"] == "complete"]
    grade_matches = grade_matches.dropna(subset=["home_goals", "away_goals"])
    grade_matches["match_date"] = pd.to_datetime(grade_matches["match_date"])
    grade_matches = grade_matches.sort_values("match_date")

    seasons = sorted(grade_matches["season"].astype(str).unique())
    engine = GrassrootsEloEngine()
    rows: list[dict] = []
    prev_season_teams: set[str] = set()

    for si, season in enumerate(seasons):
        season_matches = grade_matches[grade_matches["season"].astype(str) == season]
        season_teams = set(season_matches["home_team_id"]) | set(season_matches["away_team_id"])

        if si == 0:
            # First season: baseline rows
            baseline_date = season_matches["match_date"].min() - pd.Timedelta(days=1)
            for t in season_teams:
                engine._get_or_create(t)
                rows.append({
                    "Date": baseline_date,
                    "Team": t,
                    "Elo": float(BASE_ELO),
                    "Context": "Season start",
                    "EloDelta": 0.0,
                    "Season": season,
                })
        else:
            regression_date = pd.Timestamp(f"{season}-01-01")

            # Snapshot pre-regression Elos then apply regression to ALL teams
            pre = {t: team.elo for t, team in engine.teams.items()}
            for t_name, team_obj in engine.teams.items():
                team_obj.elo = BASE_ELO + (pre[t_name] - BASE_ELO) * (1 - PRIOR_REGRESSION_FACTOR)
            engine.reset_played()

            # Regression rows only for teams active this season
            for t_name in season_teams & set(engine.teams.keys()):
                delta = engine.teams[t_name].elo - pre[t_name]
                rows.append({
                    "Date": regression_date,
                    "Team": t_name,
                    "Elo": round(engine.teams[t_name].elo, 1),
                    "Context": f"Offseason regression ({delta:+.0f} Elo)",
                    "EloDelta": round(delta, 1),
                    "Season": season,
                })

            # New teams joining this season
            for t in season_teams - set(engine.teams.keys()):
                engine._get_or_create(t)
                rows.append({
                    "Date": regression_date,
                    "Team": t,
                    "Elo": float(BASE_ELO),
                    "Context": "Season start",
                    "EloDelta": 0.0,
                    "Season": season,
                })

            # Gap-year NaN rows for teams absent this season
            for t in prev_season_teams - season_teams:
                rows.append({
                    "Date": regression_date + pd.Timedelta(days=15),
                    "Team": t,
                    "Elo": float("nan"),
                    "Context": "",
                    "EloDelta": 0.0,
                    "Season": season,
                })

        season_records, _ = normalize_match_records(season_matches.to_dict("records"))
        pre_season_snapshot = {name: team.elo for name, team in engine.teams.items()}
        history_start = len(engine.elo_history)
        log_start = len(engine.match_log)
        engine.replay_matches(season_records, quiet=True)
        _append_history_rows(
            rows,
            engine.match_log[log_start:],
            engine.elo_history[history_start:],
            pre_season_snapshot,
            season,
        )

        prev_season_teams = season_teams

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["Team", "Date"]).reset_index(drop=True)
    return result
