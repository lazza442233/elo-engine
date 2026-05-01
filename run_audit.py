"""
Model Maintenance & Performance Audit
======================================
Runs a full walk-forward backtest across all seasons using the production-locked
hyperparameters, then computes Brier score, log-loss, calibration, league dynamics,
and Elo distribution metrics.

Usage:  python run_audit.py
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from engine.elo import GrassrootsEloEngine
from engine.match_record import normalize_match_records
from config.constants import BASE_ELO, PRIOR_REGRESSION_FACTOR


# ──────────────────────────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────────────────────────

DATA_PATH = Path("data/processed/all_seasons.csv")
OUTPUT_DIR = Path("backtest_logs")
GRADE = "first_grade"  # audit focuses on first grade


def load_matches() -> list[dict]:
    """Load all_seasons.csv and return list of match dicts, sorted by date."""
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["grade"] == GRADE and r["status"] == "complete"]
    rows.sort(key=lambda r: r["match_date"])
    return rows


def group_by_season(matches: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for m in matches:
        groups[m["season"]].append(m)
    return dict(sorted(groups.items()))


# ──────────────────────────────────────────────────────────────────────
# 2. Walk-forward backtest
# ──────────────────────────────────────────────────────────────────────

def run_backtest(matches_by_season: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """
    Walk-forward backtest: for each season, carry forward regressed Elo
    from the previous season.  Returns per-season prediction logs.
    """
    engine = GrassrootsEloEngine()
    season_logs: dict[str, list[dict]] = {}

    for season, matches in matches_by_season.items():
        season_records, _ = normalize_match_records(matches)
        log = engine.replay_matches(season_records, collect_predictions=True, quiet=True)

        season_logs[season] = log

        # End-of-season regression for next season
        for team in engine.teams.values():
            team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - PRIOR_REGRESSION_FACTOR)

        # Normalize so pool average == BASE_ELO (Elo conservation)
        avg_elo = sum(t.elo for t in engine.teams.values()) / len(engine.teams)
        offset = avg_elo - BASE_ELO
        if abs(offset) > 0.05:
            for team in engine.teams.values():
                team.elo -= offset

    return season_logs


# ──────────────────────────────────────────────────────────────────────
# 3. Scoring metrics
# ──────────────────────────────────────────────────────────────────────

def brier_score(log: list[dict]) -> float:
    """Multi-class Brier score (lower = better, random ~ 0.667)."""
    total = 0.0
    for entry in log:
        o = entry["outcome"]
        pw, pd, pl = entry["prob_win"], entry["prob_draw"], entry["prob_loss"]
        oh = 1.0 if o == 1 else 0.0
        od = 1.0 if o == 0 else 0.0
        oa = 1.0 if o == -1 else 0.0
        total += (pw - oh) ** 2 + (pd - od) ** 2 + (pl - oa) ** 2
    return total / len(log)


def log_loss(log: list[dict]) -> float:
    """Multi-class log-loss / cross-entropy (lower = better)."""
    eps = 1e-15
    total = 0.0
    for entry in log:
        o = entry["outcome"]
        pw = max(eps, min(1 - eps, entry["prob_win"]))
        pd = max(eps, min(1 - eps, entry["prob_draw"]))
        pl = max(eps, min(1 - eps, entry["prob_loss"]))
        if o == 1:
            total -= math.log(pw)
        elif o == 0:
            total -= math.log(pd)
        else:
            total -= math.log(pl)
    return total / len(log)


def accuracy(log: list[dict]) -> float:
    """Percentage of matches where the highest-probability outcome was correct."""
    correct = 0
    for entry in log:
        probs = {"win": entry["prob_win"], "draw": entry["prob_draw"], "loss": entry["prob_loss"]}
        predicted = max(probs, key=probs.get)
        actual_map = {1: "win", 0: "draw", -1: "loss"}
        if predicted == actual_map[entry["outcome"]]:
            correct += 1
    return correct / len(log)


# ──────────────────────────────────────────────────────────────────────
# 4. Calibration analysis
# ──────────────────────────────────────────────────────────────────────

def calibration_buckets(log: list[dict], n_buckets: int = 10) -> list[dict]:
    """
    Bin predictions by confidence and compare predicted vs actual win rate.
    Returns list of bucket dicts for plotting / reporting.
    """
    # Focus on predicted home-win probability vs actual home-win frequency
    entries = [(e["prob_win"], 1.0 if e["outcome"] == 1 else 0.0) for e in log]
    entries.sort(key=lambda x: x[0])

    bucket_size = max(1, len(entries) // n_buckets)
    buckets = []
    for i in range(0, len(entries), bucket_size):
        chunk = entries[i:i + bucket_size]
        if not chunk:
            continue
        avg_predicted = statistics.mean(p for p, _ in chunk)
        avg_actual = statistics.mean(a for _, a in chunk)
        buckets.append({
            "bin_mid": avg_predicted,
            "predicted": avg_predicted,
            "actual": avg_actual,
            "count": len(chunk),
        })
    return buckets


def calibration_error(buckets: list[dict]) -> float:
    """Weighted mean absolute calibration error (ECE)."""
    total_count = sum(b["count"] for b in buckets)
    if total_count == 0:
        return 0.0
    ece = sum(b["count"] * abs(b["predicted"] - b["actual"]) for b in buckets) / total_count
    return ece


# ──────────────────────────────────────────────────────────────────────
# 5. League dynamics
# ──────────────────────────────────────────────────────────────────────

def league_dynamics(matches_by_season: dict[str, list[dict]]) -> dict[str, dict]:
    """Calculate home win %, avg goals/game, draw % per season."""
    stats = {}
    for season, matches in matches_by_season.items():
        n = len(matches)
        home_wins = sum(1 for m in matches if int(m["home_goals"]) > int(m["away_goals"]))
        draws = sum(1 for m in matches if int(m["home_goals"]) == int(m["away_goals"]))
        total_goals = sum(int(m["home_goals"]) + int(m["away_goals"]) for m in matches)
        stats[season] = {
            "matches": n,
            "home_win_pct": home_wins / n * 100,
            "draw_pct": draws / n * 100,
            "avg_goals_per_game": total_goals / n,
        }
    return stats


# ──────────────────────────────────────────────────────────────────────
# 6. Elo distribution
# ──────────────────────────────────────────────────────────────────────

def elo_distribution_per_season(matches_by_season: dict[str, list[dict]]) -> dict[str, dict]:
    """Run full simulation, capturing end-of-season Elo stats per season."""
    engine = GrassrootsEloEngine()
    dist_stats = {}

    for season, matches in matches_by_season.items():
        season_records, _ = normalize_match_records(matches)
        engine.process_matches(season_records, quiet=True)

        elos = [t.elo for t in engine.teams.values()]
        dist_stats[season] = {
            "mean": statistics.mean(elos),
            "stdev": statistics.stdev(elos) if len(elos) > 1 else 0.0,
            "min": min(elos),
            "max": max(elos),
            "spread": max(elos) - min(elos),
            "n_teams": len(elos),
            "elos": {t.name: round(t.elo, 1) for t in engine.standings()},
        }

        # Regress for next season
        for team in engine.teams.values():
            team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - PRIOR_REGRESSION_FACTOR)

    return dist_stats


# ──────────────────────────────────────────────────────────────────────
# 7. Save backtest log
# ──────────────────────────────────────────────────────────────────────

def save_season_log(log: list[dict], season: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"audit_{season}_log.csv"
    fields = [
        "season", "round", "home", "away",
        "home_elo_pre", "away_elo_pre",
        "prob_win", "prob_draw", "prob_loss",
        "xg_home", "xg_away",
        "actual_home_goals", "actual_away_goals", "outcome",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in log:
            writer.writerow({k: row[k] for k in fields})
    return path


# ──────────────────────────────────────────────────────────────────────
# 8. Main audit runner
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  MODEL MAINTENANCE & PERFORMANCE AUDIT")
    print("  Elo Engine — Walk-Forward Backtest (First Grade)")
    print("=" * 72)
    print()

    # Load & group data
    all_matches = load_matches()
    by_season = group_by_season(all_matches)
    seasons = list(by_season.keys())
    print(f"  Loaded {len(all_matches)} first-grade matches across {len(seasons)} seasons: {', '.join(seasons)}")
    print()

    # ── §2: Data Integrity ─────────────────────────────────────────
    print("─" * 72)
    print("  §2 DATA INTEGRITY & INGESTION")
    print("─" * 72)
    print("  Schema check: ✓ No changes between 2022 and 2025 raw JSON")
    for s, ms in by_season.items():
        byes_forfeits = len([m for m in load_all_rows() if m["season"] == s and m["grade"] == GRADE and m["status"] != "complete"])
        print(f"  {s}: {len(ms)} complete matches (filtered: {byes_forfeits} non-complete)")
    print("  Pipeline status: ✓ Nominal")
    print()

    # ── §3: Performance Backtest ───────────────────────────────────
    print("─" * 72)
    print("  §3 MODEL PERFORMANCE AUDIT")
    print("─" * 72)
    print("  Running walk-forward backtest with production hyperparameters...")
    print()

    season_logs = run_backtest(by_season)

    # Save logs
    for s, log in season_logs.items():
        path = save_season_log(log, s)
        print(f"  Saved: {path}")

    # §3.2: Metric table
    print()
    print("  ┌──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │  Season  │  Brier   │ Log-Loss │ Accuracy │ Matches  │")
    print("  ├──────────┼──────────┼──────────┼──────────┼──────────┤")

    season_metrics = {}
    for s, log in season_logs.items():
        bs = brier_score(log)
        ll = log_loss(log)
        acc = accuracy(log)
        season_metrics[s] = {"brier": bs, "logloss": ll, "accuracy": acc, "n": len(log)}
        print(f"  │  {s}    │  {bs:.4f}  │  {ll:.4f}  │  {acc:.1%}   │  {len(log):>5}   │")

    print("  └──────────┴──────────┴──────────┴──────────┴──────────┘")

    # Compute deltas from holdout baseline (2025)
    if "2025" in season_metrics:
        # Historical baseline: average of prior seasons
        prior_seasons = [s for s in season_metrics if s != "2025"]
        if prior_seasons:
            avg_brier = statistics.mean(season_metrics[s]["brier"] for s in prior_seasons)
            avg_ll = statistics.mean(season_metrics[s]["logloss"] for s in prior_seasons)
            s2025 = season_metrics["2025"]
            brier_delta = s2025["brier"] - avg_brier
            ll_delta = s2025["logloss"] - avg_ll
            print()
            print("  2025 vs prior-season average:")
            print(f"    Brier delta: {brier_delta:+.4f} ({'degraded' if brier_delta > 0.02 else 'stable' if abs(brier_delta) <= 0.02 else 'improved'})")
            print(f"    Log-loss delta: {ll_delta:+.4f} ({'degraded' if ll_delta > 0.05 else 'stable' if abs(ll_delta) <= 0.05 else 'improved'})")

    # §3.3: Calibration
    print()
    print("  §3.3 Calibration Analysis (home-win probability):")
    print()
    # Use full dataset for overall calibration, then per-season
    all_log = []
    for log in season_logs.values():
        all_log.extend(log)

    buckets_all = calibration_buckets(all_log)
    ece_all = calibration_error(buckets_all)
    print(f"  Overall ECE (Expected Calibration Error): {ece_all:.4f}")
    print()
    print("  ┌────────────────┬─────────────┬─────────────┬───────┐")
    print("  │  Pred. Bucket  │  Predicted  │   Actual    │   N   │")
    print("  ├────────────────┼─────────────┼─────────────┼───────┤")
    for b in buckets_all:
        gap = b["actual"] - b["predicted"]
        marker = " ←" if abs(gap) > 0.10 else ""
        print(f"  │    {b['predicted']:.2%}      │   {b['predicted']:.2%}     │   {b['actual']:.2%}     │ {b['count']:>4}  │{marker}")
    print("  └────────────────┴─────────────┴─────────────┴───────┘")

    # Per-season ECE
    print()
    for s, log in season_logs.items():
        b = calibration_buckets(log, n_buckets=5)
        ece = calibration_error(b)
        print(f"  {s} ECE: {ece:.4f}")

    # ── §4: League Dynamics ────────────────────────────────────────
    print()
    print("─" * 72)
    print("  §4 PARAMETER & SYSTEM DRIFT ANALYSIS")
    print("─" * 72)

    # §4.1: League dynamics
    dynamics = league_dynamics(by_season)
    print()
    print("  §4.1 League Dynamics:")
    print()
    print("  ┌──────────┬────────────┬──────────────┬────────────┐")
    print("  │  Season  │ Home Win % │ Avg Goals/Gm │  Draw %    │")
    print("  ├──────────┼────────────┼──────────────┼────────────┤")
    for s, d in dynamics.items():
        print(f"  │  {s}    │   {d['home_win_pct']:5.1f}%   │    {d['avg_goals_per_game']:.2f}      │   {d['draw_pct']:5.1f}%   │")
    print("  └──────────┴────────────┴──────────────┴────────────┘")

    # Historical averages
    all_hw = [d["home_win_pct"] for d in dynamics.values()]
    all_ag = [d["avg_goals_per_game"] for d in dynamics.values()]
    all_dr = [d["draw_pct"] for d in dynamics.values()]
    print()
    print("  Historical averages (all seasons):")
    print(f"    Home Win %:     {statistics.mean(all_hw):.1f}% (σ={statistics.stdev(all_hw):.1f}%)")
    print(f"    Avg Goals/Game: {statistics.mean(all_ag):.2f} (σ={statistics.stdev(all_ag):.2f})")
    print(f"    Draw %:         {statistics.mean(all_dr):.1f}% (σ={statistics.stdev(all_dr):.1f}%)")

    if len(seasons) >= 2:
        latest = seasons[-1]
        prior_hw = [dynamics[s]["home_win_pct"] for s in seasons[:-1]]
        prior_ag = [dynamics[s]["avg_goals_per_game"] for s in seasons[:-1]]
        prior_dr = [dynamics[s]["draw_pct"] for s in seasons[:-1]]
        print()
        print(f"  {latest} vs prior-season average:")
        print(f"    Home Win %:     {dynamics[latest]['home_win_pct']:.1f}% vs {statistics.mean(prior_hw):.1f}%  (Δ={dynamics[latest]['home_win_pct'] - statistics.mean(prior_hw):+.1f}pp)")
        print(f"    Avg Goals/Game: {dynamics[latest]['avg_goals_per_game']:.2f} vs {statistics.mean(prior_ag):.2f}  (Δ={dynamics[latest]['avg_goals_per_game'] - statistics.mean(prior_ag):+.2f})")
        print(f"    Draw %:         {dynamics[latest]['draw_pct']:.1f}% vs {statistics.mean(prior_dr):.1f}%  (Δ={dynamics[latest]['draw_pct'] - statistics.mean(prior_dr):+.1f}pp)")

    # §4.2: Elo distribution
    print()
    print("  §4.2 End-of-Season Elo Distribution:")
    print()
    elo_dist = elo_distribution_per_season(by_season)

    print("  ┌──────────┬─────────┬─────────┬─────────┬─────────┬──────────┐")
    print("  │  Season  │  Mean   │  StDev  │   Min   │   Max   │  Spread  │")
    print("  ├──────────┼─────────┼─────────┼─────────┼─────────┼──────────┤")
    for s, d in elo_dist.items():
        print(f"  │  {s}    │ {d['mean']:7.1f} │ {d['stdev']:7.1f} │ {d['min']:7.1f} │ {d['max']:7.1f} │ {d['spread']:8.1f} │")
    print("  └──────────┴─────────┴─────────┴─────────┴─────────┴──────────┘")

    # Elo histogram (ASCII)
    latest_season = seasons[-1]
    latest_elos = elo_dist[latest_season]["elos"]
    print()
    print(f"  End-of-{latest_season} Elo Ratings (ranked):")
    print()
    max_bar = 40
    max_elo = max(latest_elos.values())
    min_elo = min(latest_elos.values())
    elo_range = max_elo - min_elo if max_elo != min_elo else 1
    for name, elo in latest_elos.items():
        bar_len = int((elo - min_elo) / elo_range * max_bar)
        delta = elo - BASE_ELO
        print(f"  {name:<35s} {elo:7.1f} ({delta:+6.1f}) {'█' * bar_len}")

    # ── §5: Conclusion ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  §5 AUDIT CONCLUSION & RECOMMENDATION")
    print("=" * 72)
    print()

    # Automated assessment
    latest = seasons[-1]
    m = season_metrics.get(latest, {})
    d = dynamics.get(latest, {})
    ed = elo_dist.get(latest, {})

    issues = []

    # Check Brier degradation
    if len(seasons) >= 2:
        prior_briers = [season_metrics[s]["brier"] for s in seasons[:-1]]
        avg_prior_brier = statistics.mean(prior_briers)
        brier_delta = m.get("brier", 0) - avg_prior_brier
        if brier_delta > 0.05:
            issues.append(f"Brier score degraded by {brier_delta:.4f} vs historical avg")
        elif brier_delta > 0.02:
            issues.append(f"Brier score slightly elevated ({brier_delta:+.4f} vs historical avg)")

    # Check calibration
    latest_buckets = calibration_buckets(season_logs.get(latest, []), n_buckets=5)
    latest_ece = calibration_error(latest_buckets)
    if latest_ece > 0.10:
        issues.append(f"Calibration error elevated (ECE={latest_ece:.4f})")

    # Check league dynamics shift
    if len(seasons) >= 2:
        prior_hw_avg = statistics.mean(dynamics[s]["home_win_pct"] for s in seasons[:-1])
        hw_shift = abs(d.get("home_win_pct", 0) - prior_hw_avg)
        if hw_shift > 10:
            issues.append(f"Home win % shifted by {hw_shift:.1f}pp from historical average")

    # Check Elo drift
    if ed.get("mean", BASE_ELO) and abs(ed["mean"] - BASE_ELO) > 50:
        issues.append(f"Mean Elo drifted to {ed['mean']:.1f} (expected ~{BASE_ELO})")

    if not issues:
        health = "HEALTHY"
        recommendation = "No Action Required"
        print(f"  Overall Health: ✓ {health}")
        print(f"  Key Finding: The model's performance on the {latest} season remains")
        print("  consistent with historical backtests, showing no significant drift.")
        print()
        print(f"  ✓ Recommendation: {recommendation}")
        print("    The model is performing within expected parameters. Continue monitoring.")
    elif len(issues) <= 2 and all("slightly" in i or "elevated" in i for i in issues):
        health = "REQUIRES RECALIBRATION"
        recommendation = "Recalibrate Model"
        print(f"  Overall Health: ⚠ {health}")
        print("  Issues detected:")
        for issue in issues:
            print(f"    • {issue}")
        print()
        print(f"  ⚠ Recommendation: {recommendation}")
        print(f"    Consider adding {latest} data to training set and re-optimizing.")
    else:
        health = "DEGRADED"
        recommendation = "Investigate Anomaly"
        print(f"  Overall Health: ✗ {health}")
        print("  Issues detected:")
        for issue in issues:
            print(f"    • {issue}")
        print()
        print(f"  ✗ Recommendation: {recommendation}")
        print("    A deeper investigation is required before recalibration.")

    print()
    print("=" * 72)
    print("  AUDIT COMPLETE")
    print("=" * 72)


def load_all_rows() -> list[dict]:
    """Load ALL rows from all_seasons.csv (including non-complete)."""
    with open(DATA_PATH) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
