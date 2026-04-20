"""
Parameter Optimisation v2 — 16 April 2026
==========================================
Three-fold expanding-window walk-forward optimisation.

Usage:
    python optimise_v2.py --stage 1          # LHS coarse search
    python optimise_v2.py --stage 2          # Refined grid (reads Stage 1 results)
    python optimise_v2.py --stage 3          # Statistical validation
    python optimise_v2.py --stage 1 --grade reserve_grade
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data loading (same as run_audit.py)
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/processed/all_seasons.csv")
OUTPUT_DIR = Path("backtest_logs")

FOLDS = [
    {"train": ["2022"], "val": "2023", "name": "F1"},
    {"train": ["2022", "2023"], "val": "2024", "name": "F2"},
    {"train": ["2022", "2023", "2024"], "val": "2025", "name": "F3"},
]


def load_all_matches(grade: str) -> dict[str, list[dict]]:
    """Load all_seasons.csv grouped by season for a given grade."""
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["grade"] == grade and r["status"] == "complete"]
    rows.sort(key=lambda r: r["match_date"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["season"]].append(r)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Evaluation: run one parameter config through 3-fold walk-forward
# ---------------------------------------------------------------------------

def evaluate_config(params: dict, matches_by_season: dict[str, list[dict]]) -> dict:
    """Run 3-fold expanding-window backtest for a single parameter config.

    Returns dict with per-fold Brier/log-loss and means.
    """
    # Import inside worker to avoid pickling issues
    import engine.elo as elo_mod
    from config.constants import BASE_ELO
    from engine.elo import GrassrootsEloEngine

    # Monkey-patch constants in the engine module's namespace
    elo_mod.K_FACTOR_INITIAL = params["K_FACTOR_INITIAL"]
    elo_mod.K_FACTOR_SETTLED = params["K_FACTOR_SETTLED"]
    elo_mod.K_TRANSITION_GAMES = params["K_TRANSITION_GAMES"]
    elo_mod.HOME_FIELD_ADVANTAGE = params["HOME_FIELD_ADVANTAGE"]
    elo_mod.MOV_C1 = params["MOV_C1"]
    elo_mod.MOV_C2 = params["MOV_C2"]
    elo_mod.ELO_TO_GOAL_RATIO = params["ELO_TO_GOAL_RATIO"]
    elo_mod.XG_ASYMMETRY_FACTOR = params["XG_ASYMMETRY_FACTOR"]
    elo_mod.XG_BLEND_WEIGHT = params["XG_BLEND_WEIGHT"]
    elo_mod.MIN_XG = params["MIN_XG"]
    elo_mod.PRIOR_REGRESSION_FACTOR = params["PRIOR_REGRESSION_FACTOR"]

    reset_played = params["RESET_PLAYED_PER_SEASON"]
    regression = params["PRIOR_REGRESSION_FACTOR"]

    fold_results = {}
    for fold in FOLDS:
        engine = GrassrootsEloEngine()
        train_seasons = fold["train"]
        val_season = fold["val"]

        # Train: process all training seasons with between-season regression
        for i, season in enumerate(train_seasons):
            if season not in matches_by_season:
                continue
            for m in matches_by_season[season]:
                engine.process_match(
                    m["home_team_id"], m["away_team_id"],
                    int(m["home_goals"]), int(m["away_goals"]),
                    round_label=m.get("full_round"),
                )
            # Between-season regression (not after the last training season — that's before val)
            if i < len(train_seasons) - 1:
                for team in engine.teams.values():
                    team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - regression)
                # Elo conservation: re-center pool at BASE_ELO (matches production)
                if engine.teams:
                    avg_elo = sum(t.elo for t in engine.teams.values()) / len(engine.teams)
                    offset = avg_elo - BASE_ELO
                    if abs(offset) > 0.05:
                        for team in engine.teams.values():
                            team.elo -= offset
                if reset_played:
                    engine.reset_played()

        # Regress before validation season
        for team in engine.teams.values():
            team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - regression)
        # Elo conservation: re-center pool at BASE_ELO (matches production)
        if engine.teams:
            avg_elo = sum(t.elo for t in engine.teams.values()) / len(engine.teams)
            offset = avg_elo - BASE_ELO
            if abs(offset) > 0.05:
                for team in engine.teams.values():
                    team.elo -= offset
        if reset_played:
            engine.reset_played()

        # Validate: predict-then-update on validation season
        if val_season not in matches_by_season:
            continue

        preds = []
        for m in matches_by_season[val_season]:
            home = m["home_team_id"]
            away = m["away_team_id"]
            h_goals = int(m["home_goals"])
            a_goals = int(m["away_goals"])

            pred = engine.predict_match(home, away)

            if h_goals > a_goals:
                outcome = 1
            elif h_goals < a_goals:
                outcome = -1
            else:
                outcome = 0

            preds.append({
                "prob_win": pred["home_win"],
                "prob_draw": pred["draw"],
                "prob_loss": pred["away_win"],
                "outcome": outcome,
            })

            # Update after prediction (walk-forward)
            engine.process_match(home, away, h_goals, a_goals,
                                 round_label=m.get("full_round"))

        # Score
        brier = _brier(preds)
        ll = _logloss(preds)
        acc = _accuracy(preds)

        fold_results[fold["name"]] = {
            "brier": brier, "logloss": ll, "accuracy": acc, "n": len(preds)
        }

    # Compute means — F1 is diagnostic only (93 training matches, too few for
    # 11 parameters).  Optimisation objective uses F2+F3 only.
    # F1 values are preserved in per-fold columns for inspection.
    opt_folds = [fn for fn in fold_results if fn != "F1"]
    briers = [fold_results[fn]["brier"] for fn in opt_folds]
    losses = [fold_results[fn]["logloss"] for fn in opt_folds]
    accs = [fold_results[fn]["accuracy"] for fn in opt_folds]

    result = {**params}
    for fname, fdata in fold_results.items():
        result[f"brier_{fname}"] = fdata["brier"]
        result[f"logloss_{fname}"] = fdata["logloss"]
        result[f"accuracy_{fname}"] = fdata["accuracy"]
        result[f"n_{fname}"] = fdata["n"]
    result["brier_mean"] = statistics.mean(briers) if briers else 999
    result["logloss_mean"] = statistics.mean(losses) if losses else 999
    result["accuracy_mean"] = statistics.mean(accs) if accs else 0

    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _brier(preds: list[dict]) -> float:
    total = 0.0
    for p in preds:
        o = p["outcome"]
        oh = 1.0 if o == 1 else 0.0
        od = 1.0 if o == 0 else 0.0
        oa = 1.0 if o == -1 else 0.0
        total += (p["prob_win"] - oh) ** 2 + (p["prob_draw"] - od) ** 2 + (p["prob_loss"] - oa) ** 2
    return total / max(len(preds), 1)


def _logloss(preds: list[dict]) -> float:
    eps = 1e-15
    total = 0.0
    for p in preds:
        o = p["outcome"]
        pw = max(eps, min(1 - eps, p["prob_win"]))
        pd = max(eps, min(1 - eps, p["prob_draw"]))
        pl = max(eps, min(1 - eps, p["prob_loss"]))
        if o == 1:
            total -= math.log(pw)
        elif o == 0:
            total -= math.log(pd)
        else:
            total -= math.log(pl)
    return total / max(len(preds), 1)


def _accuracy(preds: list[dict]) -> float:
    correct = 0
    for p in preds:
        probs = {"win": p["prob_win"], "draw": p["prob_draw"], "loss": p["prob_loss"]}
        predicted = max(probs, key=probs.get)
        actual_map = {1: "win", 0: "draw", -1: "loss"}
        if predicted == actual_map[p["outcome"]]:
            correct += 1
    return correct / max(len(preds), 1)


# ---------------------------------------------------------------------------
# Stage 1: Latin Hypercube Sampling
# ---------------------------------------------------------------------------

PARAM_SPACE = {
    "K_FACTOR_INITIAL":        [30, 40, 50, 60],
    "K_FACTOR_SETTLED":        [20, 25, 30, 35],
    "K_TRANSITION_GAMES":      [5, 8, 10, 15],
    "HOME_FIELD_ADVANTAGE":    [30, 40, 50, 60],
    "MOV_C1":                  [0.0005, 0.001, 0.002],
    "MOV_C2":                  [2.0, 3.0, 4.0],
    "PRIOR_REGRESSION_FACTOR": [0.1, 0.2, 0.3, 0.4],
    "RESET_PLAYED_PER_SEASON": [True, False],
    "ELO_TO_GOAL_RATIO":       [50, 75, 100, 125],
    "XG_ASYMMETRY_FACTOR":     [0.5, 0.65, 0.75, 0.85, 1.0],
    "XG_BLEND_WEIGHT":         [0.2, 0.35, 0.5, 0.65, 0.8],
    "MIN_XG":                  [0.1, 0.2, 0.3],
}


def generate_lhs_samples(n: int, seed: int = 42) -> list[dict]:
    """Generate n Latin Hypercube samples from the parameter space."""
    rng = np.random.default_rng(seed)
    param_names = list(PARAM_SPACE.keys())
    n_params = len(param_names)

    # LHS: for each parameter, divide into n strata and sample one from each
    samples = []
    intervals = np.zeros((n_params, n))
    for i in range(n_params):
        perm = rng.permutation(n)
        for j in range(n):
            intervals[i, j] = (perm[j] + rng.random()) / n

    for j in range(n):
        config = {}
        for i, name in enumerate(param_names):
            values = PARAM_SPACE[name]
            idx = int(intervals[i, j] * len(values))
            idx = min(idx, len(values) - 1)
            config[name] = values[idx]
        samples.append(config)

    return samples


def run_stage1(grade: str, n_samples: int = 2000, workers: int | None = None):
    """Stage 1: LHS coarse search."""
    print(f"[Stage 1] Loading {grade} data...")
    matches = load_all_matches(grade)
    print(f"[Stage 1] Seasons: {list(matches.keys())}, "
          f"total matches: {sum(len(v) for v in matches.values())}")

    samples = generate_lhs_samples(n_samples)
    print(f"[Stage 1] Generated {len(samples)} LHS samples")

    # Also include the current production config
    production = {
        "K_FACTOR_INITIAL": 30, "K_FACTOR_SETTLED": 35,
        "K_TRANSITION_GAMES": 10, "HOME_FIELD_ADVANTAGE": 30,
        "MOV_C1": 0.001, "MOV_C2": 2.0,
        "PRIOR_REGRESSION_FACTOR": 0.4, "RESET_PLAYED_PER_SEASON": False,
        "ELO_TO_GOAL_RATIO": 75, "XG_ASYMMETRY_FACTOR": 0.85,
        "XG_BLEND_WEIGHT": 0.5, "MIN_XG": 0.2,
    }
    samples.insert(0, production)  # First row = baseline

    output_path = OUTPUT_DIR / f"v2_stage1_{grade}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        list(PARAM_SPACE.keys())
        + [f"brier_{f['name']}" for f in FOLDS]
        + [f"logloss_{f['name']}" for f in FOLDS]
        + [f"accuracy_{f['name']}" for f in FOLDS]
        + [f"n_{f['name']}" for f in FOLDS]
        + ["brier_mean", "logloss_mean", "accuracy_mean"]
    )

    completed = 0
    total = len(samples)
    t0 = time.time()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        if workers is None:
            import os
            workers = max(1, os.cpu_count() - 1)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_config, cfg, matches): i
                for i, cfg in enumerate(samples)
            }
            for future in as_completed(futures):
                result = future.result()
                writer.writerow(result)
                f.flush()
                completed += 1
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                if completed % 50 == 0 or completed == total:
                    print(f"  [{completed}/{total}] "
                          f"best so far: {result.get('brier_mean', '?'):.4f} "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Sort and print top 10
    with open(output_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["brier_mean"]))
    # Rewrite sorted
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Stage 1] Complete. Results: {output_path}")
    print("[Stage 1] Top 10 configurations:")
    for i, row in enumerate(rows[:10]):
        print(f"  {i+1}. Brier={float(row['brier_mean']):.4f}  "
              f"K_I={row['K_FACTOR_INITIAL']} K_S={row['K_FACTOR_SETTLED']} "
              f"K_T={row['K_TRANSITION_GAMES']} HFA={row['HOME_FIELD_ADVANTAGE']} "
              f"C1={row['MOV_C1']} C2={row['MOV_C2']} "
              f"REG={row['PRIOR_REGRESSION_FACTOR']} RESET={row['RESET_PLAYED_PER_SEASON']} "
              f"ETG={row['ELO_TO_GOAL_RATIO']} XGA={row['XG_ASYMMETRY_FACTOR']} "
              f"MXG={row['MIN_XG']}")

    return rows


# ---------------------------------------------------------------------------
# Stage 2: Refined grid around Stage 1 winner
# ---------------------------------------------------------------------------

def _neighbours(values: list, current, include_current: bool = True) -> list:
    """Return the current value and its immediate neighbours in the list."""
    if current not in values:
        return values  # fallback: search entire range
    idx = values.index(current)
    result = []
    if idx > 0:
        result.append(values[idx - 1])
    if include_current:
        result.append(values[idx])
    if idx < len(values) - 1:
        result.append(values[idx + 1])
    return result


def run_stage2(grade: str, workers: int | None = None):
    """Stage 2: Refined grid around Stage 1 winner."""
    stage1_path = OUTPUT_DIR / f"v2_stage1_{grade}.csv"
    if not stage1_path.exists():
        print(f"[Stage 2] Stage 1 results not found: {stage1_path}")
        print("          Run --stage 1 first.")
        sys.exit(1)

    with open(stage1_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["brier_mean"]))
    best = rows[0]

    print(f"[Stage 2] Stage 1 winner: Brier={float(best['brier_mean']):.4f}")

    # Build refined grid: ±1 step from best for each parameter
    refined_space = {}
    for name, values in PARAM_SPACE.items():
        raw = best[name]
        # Convert types
        if isinstance(values[0], bool):
            current = raw == "True" or raw is True
            refined_space[name] = [True, False]  # always test both
        elif isinstance(values[0], float):
            current = float(raw)
            refined_space[name] = _neighbours(values, current)
        elif isinstance(values[0], int):
            current = int(raw)
            refined_space[name] = _neighbours(values, current)
        else:
            refined_space[name] = values

    # Generate full grid
    from itertools import product
    keys = list(refined_space.keys())
    combos = list(product(*[refined_space[k] for k in keys]))
    configs = [dict(zip(keys, combo)) for combo in combos]

    print(f"[Stage 2] Refined grid: {len(configs)} combinations")
    for name, vals in refined_space.items():
        print(f"  {name}: {vals}")

    print(f"\n[Stage 2] Loading {grade} data...")
    matches = load_all_matches(grade)

    output_path = OUTPUT_DIR / f"v2_stage2_{grade}.csv"

    fieldnames = (
        list(PARAM_SPACE.keys())
        + [f"brier_{f['name']}" for f in FOLDS]
        + [f"logloss_{f['name']}" for f in FOLDS]
        + [f"accuracy_{f['name']}" for f in FOLDS]
        + [f"n_{f['name']}" for f in FOLDS]
        + ["brier_mean", "logloss_mean", "accuracy_mean"]
    )

    completed = 0
    total = len(configs)
    t0 = time.time()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        if workers is None:
            import os
            workers = max(1, os.cpu_count() - 1)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_config, cfg, matches): i
                for i, cfg in enumerate(configs)
            }
            for future in as_completed(futures):
                result = future.result()
                writer.writerow(result)
                f.flush()
                completed += 1
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                if completed % 100 == 0 or completed == total:
                    print(f"  [{completed}/{total}] "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Sort
    with open(output_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["brier_mean"]))
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    champion = rows[0]
    print(f"\n[Stage 2] Complete. Results: {output_path}")
    print(f"[Stage 2] Champion: Brier={float(champion['brier_mean']):.4f}")
    for name in PARAM_SPACE:
        print(f"  {name} = {champion[name]}")

    return rows


# ---------------------------------------------------------------------------
# Stage 3: Statistical validation
# ---------------------------------------------------------------------------

def run_stage3(grade: str):
    """Stage 3: Bootstrap CIs, reserve grade cross-check, comparison."""
    stage2_path = OUTPUT_DIR / f"v2_stage2_{grade}.csv"
    if not stage2_path.exists():
        print(f"[Stage 3] Stage 2 results not found: {stage2_path}")
        sys.exit(1)

    with open(stage2_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["brier_mean"]))
    champion_row = rows[0]

    # Parse champion config
    champion = {}
    for name, values in PARAM_SPACE.items():
        raw = champion_row[name]
        if isinstance(values[0], bool):
            champion[name] = raw == "True" or raw is True
        elif isinstance(values[0], float):
            champion[name] = float(raw)
        elif isinstance(values[0], int):
            champion[name] = int(raw)

    # Production baseline
    production = {
        "K_FACTOR_INITIAL": 30, "K_FACTOR_SETTLED": 35,
        "K_TRANSITION_GAMES": 10, "HOME_FIELD_ADVANTAGE": 30,
        "MOV_C1": 0.001, "MOV_C2": 2.0,
        "PRIOR_REGRESSION_FACTOR": 0.4, "RESET_PLAYED_PER_SEASON": False,
        "ELO_TO_GOAL_RATIO": 75, "XG_ASYMMETRY_FACTOR": 0.85,
        "XG_BLEND_WEIGHT": 0.5, "MIN_XG": 0.2,
    }

    print("[Stage 3] Champion config:")
    for k, v in champion.items():
        changed = " ← CHANGED" if v != production[k] else ""
        print(f"  {k} = {v}{changed}")

    # --- 3a. Run both configs on primary grade ---
    print(f"\n[Stage 3] Evaluating on {grade}...")
    matches = load_all_matches(grade)
    champ_result = evaluate_config(champion, matches)
    prod_result = evaluate_config(production, matches)

    print(f"\n{'Metric':<25} {'Production':>12} {'Champion':>12} {'Delta':>10}")
    print("-" * 62)
    for fold in FOLDS:
        fn = fold["name"]
        pb = float(prod_result[f"brier_{fn}"])
        cb = float(champ_result[f"brier_{fn}"])
        delta = cb - pb
        marker = "✓" if delta < 0 else "✗"
        print(f"  Brier {fn:<18} {pb:>12.4f} {cb:>12.4f} {delta:>+10.4f} {marker}")
    pb_mean = float(prod_result["brier_mean"])
    cb_mean = float(champ_result["brier_mean"])
    delta_mean = cb_mean - pb_mean
    print(f"  {'Brier mean':<23} {pb_mean:>12.4f} {cb_mean:>12.4f} {delta_mean:>+10.4f}")
    print(f"  {'Log-loss mean':<23} {float(prod_result['logloss_mean']):>12.4f} "
          f"{float(champ_result['logloss_mean']):>12.4f}")
    print(f"  {'Accuracy mean':<23} {float(prod_result['accuracy_mean']):>12.1%} "
          f"{float(champ_result['accuracy_mean']):>12.1%}")

    # --- 3b. Bootstrap CI ---
    print("\n[Stage 3] Bootstrap confidence interval (1,000 resamples)...")
    rng = np.random.default_rng(42)
    bootstrap_deltas = []

    # Collect per-match Brier differences across all folds
    # Re-run to get per-match predictions
    all_prod_briers = []
    all_champ_briers = []

    for config, brier_list in [(production, all_prod_briers), (champion, all_champ_briers)]:
        import engine.elo as elo_mod
        from config.constants import BASE_ELO
        from engine.elo import GrassrootsEloEngine

        elo_mod.K_FACTOR_INITIAL = config["K_FACTOR_INITIAL"]
        elo_mod.K_FACTOR_SETTLED = config["K_FACTOR_SETTLED"]
        elo_mod.K_TRANSITION_GAMES = config["K_TRANSITION_GAMES"]
        elo_mod.HOME_FIELD_ADVANTAGE = config["HOME_FIELD_ADVANTAGE"]
        elo_mod.MOV_C1 = config["MOV_C1"]
        elo_mod.MOV_C2 = config["MOV_C2"]
        elo_mod.ELO_TO_GOAL_RATIO = config["ELO_TO_GOAL_RATIO"]
        elo_mod.XG_ASYMMETRY_FACTOR = config["XG_ASYMMETRY_FACTOR"]
        elo_mod.XG_BLEND_WEIGHT = config["XG_BLEND_WEIGHT"]
        elo_mod.MIN_XG = config["MIN_XG"]
        elo_mod.PRIOR_REGRESSION_FACTOR = config["PRIOR_REGRESSION_FACTOR"]
        regression = config["PRIOR_REGRESSION_FACTOR"]
        reset_played = config["RESET_PLAYED_PER_SEASON"]

        for fold in FOLDS:
            engine = GrassrootsEloEngine()
            for i, season in enumerate(fold["train"]):
                if season not in matches:
                    continue
                for m in matches[season]:
                    engine.process_match(
                        m["home_team_id"], m["away_team_id"],
                        int(m["home_goals"]), int(m["away_goals"]),
                        round_label=m.get("full_round"),
                    )
                if i < len(fold["train"]) - 1:
                    for team in engine.teams.values():
                        team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - regression)
                    if engine.teams:
                        avg_elo = sum(t.elo for t in engine.teams.values()) / len(engine.teams)
                        offset = avg_elo - BASE_ELO
                        if abs(offset) > 0.05:
                            for team in engine.teams.values():
                                team.elo -= offset
                    if reset_played:
                        engine.reset_played()

            for team in engine.teams.values():
                team.elo = BASE_ELO + (team.elo - BASE_ELO) * (1 - regression)
            if engine.teams:
                avg_elo = sum(t.elo for t in engine.teams.values()) / len(engine.teams)
                offset = avg_elo - BASE_ELO
                if abs(offset) > 0.05:
                    for team in engine.teams.values():
                        team.elo -= offset
            if reset_played:
                engine.reset_played()

            if fold["val"] not in matches:
                continue
            for m in matches[fold["val"]]:
                pred = engine.predict_match(m["home_team_id"], m["away_team_id"])
                h_goals = int(m["home_goals"])
                a_goals = int(m["away_goals"])
                if h_goals > a_goals:
                    o = 1
                elif h_goals < a_goals:
                    o = -1
                else:
                    o = 0
                oh = 1.0 if o == 1 else 0.0
                od = 1.0 if o == 0 else 0.0
                oa = 1.0 if o == -1 else 0.0
                b = (pred["home_win"] - oh)**2 + (pred["draw"] - od)**2 + (pred["away_win"] - oa)**2
                brier_list.append(b)
                engine.process_match(
                    m["home_team_id"], m["away_team_id"], h_goals, a_goals,
                    round_label=m.get("full_round"),
                )

    prod_arr = np.array(all_prod_briers)
    champ_arr = np.array(all_champ_briers)
    n_matches = len(prod_arr)

    for _ in range(1000):
        idx = rng.integers(0, n_matches, size=n_matches)
        bootstrap_deltas.append(champ_arr[idx].mean() - prod_arr[idx].mean())

    bootstrap_deltas.sort()
    ci_lo = bootstrap_deltas[24]   # 2.5th percentile
    ci_hi = bootstrap_deltas[974]  # 97.5th percentile
    mean_delta = np.mean(bootstrap_deltas)

    print(f"  Mean Brier delta: {mean_delta:+.4f}")
    print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    if ci_hi < 0:
        print("  → Improvement is STATISTICALLY SIGNIFICANT (entire CI below zero)")
    elif ci_lo > 0:
        print("  → Champion is WORSE (entire CI above zero)")
    else:
        print("  → Improvement is NOT statistically significant (CI spans zero)")

    # --- 3c. Reserve grade cross-check ---
    other_grade = "reserve_grade" if grade == "first_grade" else "first_grade"
    print(f"\n[Stage 3] Reserve grade cross-check ({other_grade})...")
    other_matches = load_all_matches(other_grade)
    if other_matches:
        other_prod = evaluate_config(production, other_matches)
        other_champ = evaluate_config(champion, other_matches)
        print(f"  Production Brier ({other_grade}): {float(other_prod['brier_mean']):.4f}")
        print(f"  Champion Brier ({other_grade}):   {float(other_champ['brier_mean']):.4f}")
        other_delta = float(other_champ["brier_mean"]) - float(other_prod["brier_mean"])
        print(f"  Delta: {other_delta:+.4f}")
        if other_delta > 0.01:
            print(f"  ⚠ WARNING: Champion degrades {other_grade} by >{0.01:.3f} Brier")
        else:
            print(f"  ✓ {other_grade} is acceptable")
    else:
        print(f"  No data for {other_grade}")

    # --- 3d. Save comparison ---
    comp_path = OUTPUT_DIR / "v2_champion_comparison.csv"
    with open(comp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "grade", "brier_F1", "brier_F2", "brier_F3",
                          "brier_mean", "logloss_mean", "accuracy_mean"])
        for label, res in [("production", prod_result), ("champion", champ_result)]:
            writer.writerow([
                label, grade,
                res.get("brier_F1", ""), res.get("brier_F2", ""), res.get("brier_F3", ""),
                res["brier_mean"], res["logloss_mean"], res["accuracy_mean"],
            ])

    print(f"\n[Stage 3] Comparison saved: {comp_path}")

    # --- 3e. Decision ---
    print(f"\n{'='*62}")
    print("DECISION FRAMEWORK")
    print(f"{'='*62}")
    checks = []
    checks.append(("Mean Brier improvement > 0", delta_mean < 0))
    checks.append(("Bootstrap 95% CI entirely below zero", ci_hi < 0))
    if other_matches:
        checks.append((f"{other_grade} not degraded >0.01", other_delta <= 0.01))

    # Check no fold is dramatically worse
    fold_ok = True
    for fold in FOLDS:
        fn = fold["name"]
        pb = float(prod_result[f"brier_{fn}"])
        cb = float(champ_result[f"brier_{fn}"])
        if cb > pb * 1.1:  # >10% worse
            fold_ok = False
    checks.append(("No fold >10% worse", fold_ok))

    all_pass = True
    for desc, ok in checks:
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}: {desc}")

    print()
    if all_pass:
        print("  → RECOMMENDATION: ADOPT new parameters")
        print("    Next steps:")
        print("    1. Update config/constants.py with champion values")
        print("    2. Run: python run_audit.py")
        print("    3. Run: python generate_2026_priors.py")
    else:
        print("  → RECOMMENDATION: KEEP current production parameters")
        print("    The improvement is not convincing enough to justify the change.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Optimisation v2")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--grade", default="first_grade",
                        choices=["first_grade", "reserve_grade"])
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of LHS samples for Stage 1 (default: 2000)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    args = parser.parse_args()

    if args.stage == 1:
        run_stage1(args.grade, n_samples=args.samples, workers=args.workers)
    elif args.stage == 2:
        run_stage2(args.grade, workers=args.workers)
    elif args.stage == 3:
        run_stage3(args.grade)


if __name__ == "__main__":
    main()
