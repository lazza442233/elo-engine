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
from engine.calibration import compute_brier_score
from engine.elo import GrassrootsEloEngine
from engine.match_record import MatchRecord, normalize_match_records
from persistence.db import (
    init_db,
    load_match_records,
    save_matches,
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
    match_records = []
    raw_fixtures = []
    offline = args.offline

    if offline:
        if not quiet:
            print("[1/5] Loading cached match data from SQLite...")
        match_records = load_match_records(league_key)
        if not match_records:
            if not quiet:
                print("      ERROR: No cached data. Run online first to populate.")
            return
        if not quiet:
            print(f"      Loaded {len(match_records)} cached result(s).")
        if not quiet:
            print("[2/5] Skipping fixture fetch (offline mode).")
    else:
        if not quiet:
            print("[1/5] Fetching match results from Dribl API...")
        try:
            raw_matches = fetch_dribl_data(api_url)
            match_records, _ = normalize_match_records(raw_matches)
            if not quiet:
                print(f"      Retrieved {len(raw_matches)} result(s).")
        except Exception as exc:
            # Fall back to cached data if available
            match_records = load_match_records(league_key)
            if match_records:
                offline = True
                if not quiet:
                    print(f"      WARNING: API failed ({exc}). "
                          f"Falling back to {len(match_records)} cached result(s).")
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
            active = {m.home_team for m in match_records} | {m.away_team for m in match_records}
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
    engine.process_matches(match_records, quiet=quiet, log_calibration=args.log_results)
    if not quiet:
        print(f"      Processed {engine.processed_matches} match(es) across "
              f"{len(engine.teams)} team(s).")

    # Persist to SQLite (skip in offline mode — data already there)
    if not offline:
        _save_to_db(league_key, match_records, quiet)

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


def _save_to_db(league_key: str, match_records: list[MatchRecord], quiet: bool):
    """Persist processed matches to SQLite for offline replay."""
    rows = []
    for record in match_records:
        if record.match_date is None:
            continue
        rows.append({
            "match_date": record.match_date,
            "home_team": record.home_team,
            "away_team": record.away_team,
            "home_score": record.home_score,
            "away_score": record.away_score,
            "status": record.status,
            "round_label": record.round_label,
        })
    inserted = save_matches(league_key, rows)
    if not quiet and inserted:
        print(f"      Saved {inserted} new match(es) to database.")


if __name__ == "__main__":
    main()
