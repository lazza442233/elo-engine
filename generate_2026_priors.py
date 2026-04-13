"""
Generate 2026 Season Priors — Methodologically Sound
=====================================================
Produces production-ready priors files for First Grade and Reserve Grade
by running a walk-forward simulation on all historical data, then applying
a validated inter-season regression.

Every prior has a documented origin:
  - RETURNING:      Active in 2025. Raw end-of-2025 Elo → regressed.
  - HIATUS:         Not in 2025 but present in prior seasons. Their Elo has
                    already been regressed at each season boundary they were
                    absent for. One final regression applied for 2025→2026.
  - TRANSFERRED:    New to the district (Castle Hill, Kellyville). Anchored
                    via historical 1st/2nd place averages, per grade, then
                    regressed. Assumes they were competitive in their prior
                    district; the anchor represents "strong newcomer, not
                    dominant".
  - PROMOTED:       Promoted from a lower division (Gladesville). Starts at
                    BASE_ELO (1500.0) — no information advantage.

Regression factor
-----------------
The D1 grid search (backtest_logs/d1_grid_results.csv) validated a retention
rate of 0.8 (REGRESSION_FACTOR column), which corresponds to:
    PRIOR_REGRESSION_FACTOR = 0.2  →  retention = 1 - 0.2 = 0.8 (80%)

This matches config/constants.py. To override, change REGRESSION_FACTOR below.

Usage:  python generate_2026_priors.py
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from config.constants import BASE_ELO
from engine.elo import GrassrootsEloEngine

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Validated regression factor (retain 80% of delta). Change here to override.
REGRESSION_FACTOR = 0.2   # 1 - REGRESSION_FACTOR = retention rate

DATA_PATH = Path("data/processed/all_seasons.csv")
OUTPUT_DIR = Path("data")

# New teams entering the 2026 competition from outside the district.
# Map: team_name → {"origin": str, "anchor": "1st"|"2nd"|None}
NEW_TEAMS = {
    "Castle Hill United Football Club": {
        "origin": "TRANSFERRED",
        "anchor": "1st",  # strong newcomer, anchored at historical 2nd place
    },
    "Kellyville Kolts Soccer Club": {
        "origin": "TRANSFERRED",
        "anchor": "1st",  # strong newcomer, anchored at historical 2nd place
    },
    "Gladesville Ravens SC": {
        "origin": "PROMOTED",
        "anchor": None,   # starts at BASE_ELO
    },
}


# ──────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────

def load_matches(grade: str) -> list[dict]:
    with open(DATA_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["grade"] == grade and r["status"] == "complete"]
    rows.sort(key=lambda r: r["match_date"])
    return rows


def group_by_season(matches: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for m in matches:
        groups[m["season"]].append(m)
    return dict(sorted(groups.items()))


# ──────────────────────────────────────────────────────────────────────
# 2. Walk-forward simulation (captures raw end-of-season Elos)
# ──────────────────────────────────────────────────────────────────────

def run_walk_forward(by_season: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """
    Run a walk-forward simulation with inter-season regression.

    Returns:
      - raw_elos: dict of {team_name: raw_end_of_final_season_elo}
                  (regression applied at every boundary EXCEPT after the last season)
      - season_top2: list of dicts with 1st/2nd place raw Elos per season
    """
    engine = GrassrootsEloEngine()
    seasons = list(by_season.keys())
    season_top2: list[dict] = []

    for i, season in enumerate(seasons):
        for m in by_season[season]:
            engine.process_match(
                m["home_team_id"], m["away_team_id"],
                int(m["home_goals"]), int(m["away_goals"]),
                round_label=m["full_round"],
            )

        # Capture standings at end of this season (raw, pre-regression)
        standings = engine.standings()
        season_top2.append({
            "season": season,
            "1st_name": standings[0].name,
            "1st_elo": standings[0].elo,
            "2nd_name": standings[1].name,
            "2nd_elo": standings[1].elo,
        })

        # Apply inter-season regression — but NOT after the last season
        # (we want the raw end-of-last-season Elos to regress ourselves)
        if i < len(seasons) - 1:
            for team in engine.teams.values():
                team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - REGRESSION_FACTOR)

    # Capture raw end-of-final-season Elos
    raw_elos = {t.name: t.elo for t in engine.teams.values()}
    return raw_elos, season_top2


# ──────────────────────────────────────────────────────────────────────
# 3. Compute historical anchors for transferred teams
# ──────────────────────────────────────────────────────────────────────

def compute_anchors(season_top2: list[dict]) -> dict[str, float]:
    """Average historical end-of-season Elo for 1st and 2nd place."""
    first_elos = [s["1st_elo"] for s in season_top2]
    second_elos = [s["2nd_elo"] for s in season_top2]
    return {
        "1st": statistics.mean(first_elos),
        "2nd": statistics.mean(second_elos),
    }


# ──────────────────────────────────────────────────────────────────────
# 4. Apply regression and assign priors
# ──────────────────────────────────────────────────────────────────────

def regress(elo: float) -> float:
    """Apply one round of regression toward BASE_ELO."""
    return BASE_ELO + (elo - BASE_ELO) * (1 - REGRESSION_FACTOR)


def classify_team(name: str, raw_elos: dict[str, float], teams_in_2025: set[str]) -> str:
    """Classify a team's origin for documentation."""
    if name in NEW_TEAMS:
        return NEW_TEAMS[name]["origin"]
    if name in teams_in_2025:
        return "RETURNING"
    if name in raw_elos:
        return "HIATUS"
    return "UNKNOWN"


def generate_priors(
    grade: str,
    raw_elos: dict[str, float],
    season_top2: list[dict],
    teams_in_2025: set[str],
) -> dict[str, float]:
    """Generate the final priors dict for a grade."""
    anchors = compute_anchors(season_top2)
    priors: dict[str, float] = {}
    audit_log: list[str] = []

    # 1. Process all returning/hiatus teams from the simulation
    for name, raw_elo in sorted(raw_elos.items(), key=lambda x: -x[1]):
        origin = classify_team(name, raw_elos, teams_in_2025)
        prior = regress(raw_elo)
        priors[name] = round(prior, 1)
        audit_log.append(
            f"  {name:<45s} raw={raw_elo:7.1f} → prior={prior:7.1f}  "
            f"(Δ={prior - BASE_ELO:+6.1f})  [{origin}]"
        )

    # 2. Process new teams
    for name, cfg in NEW_TEAMS.items():
        if name in priors:
            continue  # already handled (shouldn't happen)

        if cfg["origin"] == "PROMOTED":
            prior = float(BASE_ELO)
            anchor_label = "BASE_ELO"
            raw_val = BASE_ELO
        elif cfg["origin"] == "TRANSFERRED":
            anchor_key = cfg["anchor"]  # "1st" or "2nd"
            ghost_rating = anchors[anchor_key]
            prior = regress(ghost_rating)
            anchor_label = f"anchor={ghost_rating:.1f} (hist avg {anchor_key} place)"
            raw_val = ghost_rating
        else:
            prior = float(BASE_ELO)
            anchor_label = "FALLBACK"
            raw_val = BASE_ELO

        priors[name] = round(prior, 1)
        audit_log.append(
            f"  {name:<45s} raw={raw_val:7.1f} → prior={prior:7.1f}  "
            f"(Δ={prior - BASE_ELO:+6.1f})  [{cfg['origin']}] {anchor_label}"
        )

    return priors, audit_log, anchors


# ──────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    retention = 1 - REGRESSION_FACTOR
    print("=" * 78)
    print("  2026 SEASON PRIORS GENERATOR")
    print("=" * 78)
    print()
    print(f"  Regression factor:  {REGRESSION_FACTOR}  (retention: {retention:.0%})")
    print(f"  Base Elo:           {BASE_ELO}")
    print(f"  Data source:        {DATA_PATH}")
    print()

    for grade, grade_label, output_file in [
        ("first_grade", "FIRST GRADE", "end_of_season_elos_first_grade.json"),
        ("reserve_grade", "RESERVE GRADE", "end_of_season_elos_reserve_grade.json"),
    ]:
        print("─" * 78)
        print(f"  {grade_label}")
        print("─" * 78)
        print()

        # Load and simulate
        all_matches = load_matches(grade)
        by_season = group_by_season(all_matches)
        seasons = list(by_season.keys())

        # Identify teams active in 2025
        teams_in_2025 = set()
        for m in by_season.get("2025", []):
            teams_in_2025.add(m["home_team_id"])
            teams_in_2025.add(m["away_team_id"])

        print(f"  Loaded {len(all_matches)} matches across {len(seasons)} seasons")
        print(f"  Teams active in 2025: {len(teams_in_2025)}")
        print()

        # Walk-forward simulation
        raw_elos, season_top2 = run_walk_forward(by_season)

        # Display historical top-2 per season
        print("  Historical end-of-season top-2 (raw Elo):")
        for s in season_top2:
            print(f"    {s['season']}: 1st {s['1st_name']:<35s} {s['1st_elo']:7.1f}")
            print(f"          2nd {s['2nd_name']:<35s} {s['2nd_elo']:7.1f}")
        print()

        # Compute anchors
        anchors = compute_anchors(season_top2)
        print(f"  Historical anchor — avg 1st place Elo: {anchors['1st']:.1f}")
        print(f"  Historical anchor — avg 2nd place Elo: {anchors['2nd']:.1f}")
        print()

        # Generate priors
        priors, audit_log, _ = generate_priors(grade, raw_elos, season_top2, teams_in_2025)

        # Display audit log
        print(f"  2026 Priors (regression={REGRESSION_FACTOR}, retention={retention:.0%}):")
        print()
        for line in audit_log:
            print(line)
        print()

        # Elo conservation check
        avg_prior = statistics.mean(priors.values())
        print(f"  Average prior: {avg_prior:.1f} (ideal: {BASE_ELO})")
        print(f"  Teams in priors file: {len(priors)}")
        print()

        # Sort by prior Elo (descending) for the output file
        sorted_priors = dict(sorted(priors.items(), key=lambda x: -x[1]))

        # Write output
        out_path = OUTPUT_DIR / output_file
        out_path.write_text(json.dumps(sorted_priors, indent=2))
        print(f"  ✓ Written to {out_path}")
        print()

    # ── Summary ────────────────────────────────────────────────────
    print("=" * 78)
    print("  PRIORS GENERATION COMPLETE")
    print("=" * 78)
    print()
    print("  Methodology applied:")
    print(f"    • Walk-forward simulation across 2022–2025 with {REGRESSION_FACTOR} regression at each boundary")
    print(f"    • Final 2025→2026 regression: retain {retention:.0%} of delta from {BASE_ELO}")
    print(f"    • Transferred teams: anchored at historical avg 2nd-place Elo per grade, then regressed")
    print(f"    • Promoted teams: start at {BASE_ELO:.0f} (no information advantage)")
    print(f"    • Hiatus teams: carry accumulated (already-regressed) Elo, plus one more regression")
    print()
    print("  ⚠  Regression factor note:")
    print(f"    Production code (config/constants.py) uses PRIOR_REGRESSION_FACTOR = 0.2 (80% retention).")
    print(f"    D1 grid search validated retention=0.8 as optimal (REGRESSION_FACTOR column).")
    print(f"    This script used REGRESSION_FACTOR = {REGRESSION_FACTOR} ({retention:.0%} retention).")
    if REGRESSION_FACTOR != 0.2:
        print(f"    ⚠ This differs from the validated value. Review before committing.")
    print()


if __name__ == "__main__":
    main()
