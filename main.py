"""
Grassroots Football Elo Engine & Match Predictor
=================================================
Skellam-Elo Hybrid Model for local/amateur football leagues.

Entry point — run with: python main.py
                        python main.py --priors          (use manual prior ratings)
                        python main.py --league prem-res (different league/grade)
                        python main.py --output json     (machine-readable JSON output)
                        python main.py --export-ratings  (save regressed Elos for next season)
                        python main.py --export-history  (save per-match Elo trajectory CSV)
                        python main.py --calibrate       (show Brier score from logged predictions)
"""

import argparse

from api.client import detect_next_round, fetch_dribl_data
from config.constants import (
    DEFAULT_LEAGUE,
    DEFAULT_ROUND,
    LEAGUES,
    _build_api_url,
    fixtures_url,
)
from display.output import (
    output_json,
    print_prediction,
    print_rankings,
    print_round_predictions,
)
from engine.calibration import compute_brier_score, log_prediction
from engine.elo import GrassrootsEloEngine
from persistence.db import (
    init_db,
    load_matches,
    save_matches,
    save_team_ratings,
)


def main():
    league_choices = list(LEAGUES.keys())
    parser = argparse.ArgumentParser(description="Grassroots Football Elo Engine")
    parser.add_argument(
        "--priors",
        action="store_true",
        help="Use manual prior ratings from league config instead of starting all teams at BASE_ELO (1500)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help=f"Fixture round number (default: auto-detect, fallback: {DEFAULT_ROUND})",
    )
    parser.add_argument(
        "--league",
        choices=league_choices,
        default=DEFAULT_LEAGUE,
        help=f"League to analyse (default: {DEFAULT_LEAGUE}). Available: {', '.join(league_choices)}",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or json",
    )
    parser.add_argument(
        "--export-ratings",
        action="store_true",
        help="Export regressed end-of-season Elo ratings to JSON for next-season priors",
    )
    parser.add_argument(
        "--export-history",
        action="store_true",
        help="Export per-match Elo history to CSV for plotting trajectories",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Show Brier score from previously logged predictions",
    )
    parser.add_argument(
        "--log-results",
        action="store_true",
        help="Log predictions for completed matches to the calibration CSV",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached match data from SQLite instead of fetching from the API",
    )
    args = parser.parse_args()

    quiet = args.output == "json"
    league_key = args.league
    league_cfg = LEAGUES[league_key]
    api_url = _build_api_url(league_key)

    # ------------------------------------------------------------------
    # Calibration-only mode
    # ------------------------------------------------------------------
    if args.calibrate:
        result = compute_brier_score()
        if result is None:
            print("  No calibration data found. Run with --log-results first.")
        else:
            print()
            print(f"  Brier Score: {result['brier_score']:.4f}  "
                  f"({result['n_predictions']} predictions logged)")
            print("  (lower is better — random = 0.667, perfect = 0.000)")
            print()
        return

    if not quiet:
        print()
        print("┌─────────────────────────────────────────────────────────────────┐")
        print("│  Grassroots Football Elo Engine  •  Skellam-Elo Hybrid Model    │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print(f"  League: {league_cfg['name']}")
        print()

    # ------------------------------------------------------------------
    # 1. Fetch results + upcoming fixtures
    # ------------------------------------------------------------------
    init_db()
    raw_matches = []
    raw_fixtures = []
    offline = args.offline

    if offline:
        if not quiet:
            print("[1/5] Loading cached match data from SQLite...")
        cached = load_matches(league_key)
        if not cached:
            if not quiet:
                print("      ERROR: No cached data. Run online first to populate.")
            return
        # Convert DB rows into the API-like dicts the engine expects
        raw_matches = _db_rows_to_api_format(cached)
        if not quiet:
            print(f"      Loaded {len(raw_matches)} cached result(s).")
        if not quiet:
            print("[2/5] Skipping fixture fetch (offline mode).")
    else:
        if not quiet:
            print("[1/5] Fetching match results from Dribl API...")
        try:
            raw_matches = fetch_dribl_data(api_url)
            if not quiet:
                print(f"      Retrieved {len(raw_matches)} result(s).")
        except Exception as exc:
            # Fall back to cached data if available
            cached = load_matches(league_key)
            if cached:
                raw_matches = _db_rows_to_api_format(cached)
                offline = True
                if not quiet:
                    print(f"      WARNING: API failed ({exc}). "
                          f"Falling back to {len(raw_matches)} cached result(s).")
            else:
                if not quiet:
                    print(f"      ERROR: Could not fetch results — {exc}")
                return

        # Auto-detect round or use explicit --round
        if not offline:
            if args.round is not None:
                if not quiet:
                    print(f"[2/5] Fetching upcoming fixtures (round {args.round}) from Dribl API...")
                try:
                    raw_fixtures = fetch_dribl_data(fixtures_url(args.round, league_key))
                    if not quiet:
                        print(f"      Retrieved {len(raw_fixtures)} upcoming fixture(s).")
                except Exception as exc:
                    if not quiet:
                        print(f"      WARNING: Could not fetch fixtures — {exc}")
            else:
                if not quiet:
                    print("[2/5] Auto-detecting next round...")
                try:
                    detected_round, raw_fixtures = detect_next_round(league_key=league_key)
                    if not quiet:
                        print(f"      Detected round {detected_round} with {len(raw_fixtures)} fixture(s).")
                except Exception as exc:
                    if not quiet:
                        print(f"      WARNING: Could not auto-detect round — {exc}")
        elif not quiet:
            print("[2/5] Skipping fixture fetch (API fallback mode).")

    # ------------------------------------------------------------------
    # 3. Build engine
    # ------------------------------------------------------------------
    if not quiet:
        print("[3/5] Initialising Elo engine...")
    engine = GrassrootsEloEngine()

    # ------------------------------------------------------------------
    # PRIORS — Load carry-forward ratings from file, or manual overrides
    # ------------------------------------------------------------------
    _PRIORS_PATHS = {
        "prem-men": "data/end_of_season_elos_first_grade.json",
        "prem-res": "data/end_of_season_elos_reserve_grade.json",
    }
    if args.priors:
        # Try grade-specific file first, fall back to league config overrides
        priors_path = _PRIORS_PATHS.get(league_key, _PRIORS_PATHS["prem-men"])
        try:
            priors = GrassrootsEloEngine.load_priors_from_file(priors_path)
            # Only inject priors for teams in the current season
            active = {
                GrassrootsEloEngine._shorten_name(m["attributes"]["home_team_name"])
                for m in raw_matches
            } | {
                GrassrootsEloEngine._shorten_name(m["attributes"]["away_team_name"])
                for m in raw_matches
            }
            priors = {k: v for k, v in priors.items() if k in active}
            engine.inject_priors(priors, quiet=quiet)
        except FileNotFoundError:
            priors = league_cfg["priors"]
            if priors:
                engine.inject_priors(priors, quiet=quiet)
            elif not quiet:
                print(f"      No priors file at {priors_path} or league config. "
                      "Run with --export-ratings at end of season to generate one.")
    elif not quiet:
        print("      All teams start at BASE_ELO (1500).")

    # ------------------------------------------------------------------
    # 4. Process matches
    # ------------------------------------------------------------------
    if not quiet:
        print("[4/5] Processing match history...")
    engine.process_matches(raw_matches, quiet=quiet)
    if not quiet:
        print(f"      Processed {engine.processed_matches} match(es) across "
              f"{len(engine.teams)} team(s).")

    # Persist to SQLite (skip in offline mode — data already there)
    if not offline:
        _save_to_db(league_key, raw_matches, engine, quiet)

    # ------------------------------------------------------------------
    # 4b. Log predictions for completed matches (calibration)
    # ------------------------------------------------------------------
    if args.log_results:
        if not quiet:
            print("      Logging predictions for completed matches...")
        logged = _log_completed_matches(engine, raw_matches)
        if not quiet:
            print(f"      Logged {logged} prediction(s) to calibration CSV.")

    # ------------------------------------------------------------------
    # 5. Output
    # ------------------------------------------------------------------
    if quiet:
        output_json(engine, raw_fixtures)
    else:
        print("[5/5] Generating output...\n")
        print_rankings(engine)
        if raw_fixtures:
            print_round_predictions(engine, raw_fixtures)
        else:
            top_two = engine.standings()[:2]
            if len(top_two) == 2:
                print()
                print("  Sample prediction: top-2 matchup")
                print_prediction(engine, top_two[0].name, top_two[1].name)

    # ------------------------------------------------------------------
    # Export ratings for next season
    # ------------------------------------------------------------------
    if args.export_ratings:
        engine.export_ratings()

    # ------------------------------------------------------------------
    # Export Elo history
    # ------------------------------------------------------------------
    if args.export_history:
        engine.export_elo_history()


def _save_to_db(league_key: str, raw_matches: list[dict],
                engine: GrassrootsEloEngine, quiet: bool):
    """Persist processed matches and team ratings to SQLite."""
    SKIP_STATUSES = {"forfeit", "abandoned", "postponed", "upcoming", "bye"}
    rows = []
    for match in raw_matches:
        attrs = match["attributes"]
        if attrs.get("bye_flag", 0):
            continue
        status = attrs.get("status", "").lower()
        if status in SKIP_STATUSES:
            continue
        home_score = attrs.get("home_score")
        away_score = attrs.get("away_score")
        if home_score is None or away_score is None:
            continue
        rows.append({
            "date": attrs["date"],
            "home_team": GrassrootsEloEngine._shorten_name(attrs["home_team_name"]),
            "away_team": GrassrootsEloEngine._shorten_name(attrs["away_team_name"]),
            "home_score": int(home_score),
            "away_score": int(away_score),
            "status": status,
        })
    inserted = save_matches(league_key, rows)
    save_team_ratings(league_key, engine.teams)
    if not quiet and inserted:
        print(f"      Saved {inserted} new match(es) to database.")


def _log_completed_matches(engine: GrassrootsEloEngine, raw_matches: list[dict]) -> int:
    """Generate predictions and log them against actual results for calibration."""
    SKIP_STATUSES = {"forfeit", "abandoned", "postponed", "upcoming", "bye"}
    logged = 0

    for match in raw_matches:
        attrs = match["attributes"]
        if attrs.get("bye_flag", 0):
            continue
        status = attrs.get("status", "").lower()
        if status in SKIP_STATUSES:
            continue
        home_score = attrs.get("home_score")
        away_score = attrs.get("away_score")
        if home_score is None or away_score is None:
            continue

        home = GrassrootsEloEngine._shorten_name(attrs["home_team_name"])
        away = GrassrootsEloEngine._shorten_name(attrs["away_team_name"])

        prediction = engine.predict_match(home, away)
        log_prediction(prediction, home, away, int(home_score), int(away_score))
        logged += 1

    return logged


def _db_rows_to_api_format(rows: list[dict]) -> list[dict]:
    """Convert SQLite match rows into the Dribl API-like dict format
    that GrassrootsEloEngine.process_matches() expects."""
    result = []
    for r in rows:
        result.append({
            "attributes": {
                "date": r["match_date"],
                "home_team_name": r["home_team"],
                "away_team_name": r["away_team"],
                "home_score": r["home_score"],
                "away_score": r["away_score"],
                "status": r.get("status", "played"),
                "bye_flag": 0,
            }
        })
    return result


if __name__ == "__main__":
    main()
