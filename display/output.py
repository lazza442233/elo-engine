"""
Display helpers for rendering rankings and predictions to the terminal.
"""

import json
from zoneinfo import ZoneInfo

from dashboard.helpers import parse_fixture_datetime
from engine.elo import GrassrootsEloEngine

SYDNEY_TZ = ZoneInfo("Australia/Sydney")


def print_rankings(engine: GrassrootsEloEngine):
    table = engine.standings()
    print()
    print("=" * 72)
    print("  CURRENT ELO RANKINGS")
    print("=" * 72)
    header = f"  {'Rank':<5} {'Team':<38} {'Elo':>6}  {'MP':>3}  {'W':>2}  {'D':>2}  {'L':>2}  {'GD':>4}"
    print(header)
    print("-" * 72)
    for rank, team in enumerate(table, start=1):
        print(
            f"  {rank:<5} {team.name:<38} {team.elo:>6.1f}  "
            f"{team.played:>3}  {team.wins:>2}  {team.draws:>2}  "
            f"{team.losses:>2}  {team.gd:>+4}"
        )
    print("=" * 72)


def _prediction_bars(result: dict, home: str, away: str):
    """Render Win/Draw/Loss probability bars."""
    bar_len = 36
    hw = result["home_win"]
    dr = result["draw"]
    aw = result["away_win"]
    hw_bar = round(hw * bar_len)
    dr_bar = round(dr * bar_len)
    aw_bar = bar_len - hw_bar - dr_bar

    print(f"    xG  {result['xg_home']:.2f} — {result['xg_away']:.2f}   "
          f"Expected GD: {result['expected_gd']:+.2f}")
    print(f"    {home:<28} Win  [{('█' * hw_bar):<{bar_len}}] {hw * 100:5.1f}%")
    print(f"    {'Draw':<28}      [{('█' * dr_bar):<{bar_len}}] {dr * 100:5.1f}%")
    print(f"    {away:<28} Win  [{('█' * aw_bar):<{bar_len}}] {aw * 100:5.1f}%")


def print_prediction(engine: GrassrootsEloEngine, home: str, away: str):
    result = engine.predict_match(home, away)
    print()
    print("=" * 72)
    print("  MATCH PREDICTION  (Skellam-Elo Hybrid)")
    print("=" * 72)
    print(f"  {home}  vs  {away}")
    print()
    _prediction_bars(result, home, away)
    print("=" * 72)
    print()


def print_round_predictions(engine: GrassrootsEloEngine, raw_fixtures: list[dict]):
    """
    Print a Skellam prediction for every upcoming fixture in the fetched round.
    Fixtures are displayed in kick-off time order.
    """

    fixtures = sorted(
        raw_fixtures,
        key=lambda m: parse_fixture_datetime(m["attributes"]["date"]),
    )

    # Derive round label from first fixture
    round_label = fixtures[0]["attributes"].get("full_round", "Next Round") if fixtures else "Next Round"

    print()
    print("=" * 72)
    print(f"  {round_label.upper()} PREDICTIONS  (Skellam-Elo Hybrid)")
    print("=" * 72)

    shown = 0
    for fix in fixtures:
        attrs = fix["attributes"]
        if attrs.get("bye_flag"):
            continue

        home = GrassrootsEloEngine._shorten_name(attrs["home_team_name"])
        away = GrassrootsEloEngine._shorten_name(attrs["away_team_name"])

        # Parse kick-off time for display (convert UTC → Sydney)
        utc_dt = parse_fixture_datetime(attrs["date"])
        if utc_dt.year > 1:  # not datetime.min
            ko = utc_dt.astimezone(SYDNEY_TZ).strftime("%a %d %b  %H:%M AEST")
        else:
            ko = attrs["date"]

        print(f"\n  {home}  vs  {away}")
        print(f"  {attrs.get('ground_name', '')}  •  {ko}")
        print()

        result = engine.predict_match(home, away)
        _prediction_bars(result, home, away)
        print()
        shown += 1

    if shown == 0:
        print("  No upcoming fixtures found.")

    print("=" * 72)
    print()


# ------------------------------------------------------------------ #
# JSON output                                                          #
# ------------------------------------------------------------------ #

def output_json(engine: GrassrootsEloEngine, raw_fixtures: list[dict]):
    """Print rankings and predictions as a single JSON object to stdout."""
    table = engine.standings()

    rankings = []
    for rank, team in enumerate(table, start=1):
        rankings.append({
            "rank": rank,
            "team": team.name,
            "elo": round(team.elo, 1),
            "played": team.played,
            "wins": team.wins,
            "draws": team.draws,
            "losses": team.losses,
            "gd": team.gd,
        })

    predictions = []
    for fix in raw_fixtures:
        attrs = fix["attributes"]
        if attrs.get("bye_flag"):
            continue
        home = GrassrootsEloEngine._shorten_name(attrs["home_team_name"])
        away = GrassrootsEloEngine._shorten_name(attrs["away_team_name"])
        result = engine.predict_match(home, away)
        predictions.append({
            "home": home,
            "away": away,
            "ground": attrs.get("ground_name", ""),
            "date": attrs.get("date", ""),
            "home_win": round(result["home_win"], 4),
            "draw": round(result["draw"], 4),
            "away_win": round(result["away_win"], 4),
            "xg_home": round(result["xg_home"], 2),
            "xg_away": round(result["xg_away"], 2),
            "expected_gd": round(result["expected_gd"], 2),
        })

    print(json.dumps({"rankings": rankings, "predictions": predictions}, indent=2))
