# Parameter Optimisation Plan — 16 April 2026

## 1. Motivation

The current production parameters were optimised in a two-stage grid search (D1) using a walk-forward methodology: train on 2022–2023, validate on 2024, holdout on 2025. The champion configuration achieved a Brier score of **0.496** (random baseline ≈ 0.667).

Three gaps have been identified that warrant a re-optimisation:

| Gap | Description | Impact |
|-----|------------|--------|
| **G1 — K-factor never re-evaluated** | `K_FACTOR_INITIAL` (40), `K_TRANSITION_GAMES` (10) were hand-set. Only `K_FACTOR_SETTLED` was grid-searched. | May be leaving accuracy on the table; early-season calibration speed is untested. |
| **G2 — `played` accumulates across seasons** | `team.played` never resets between seasons, so the K=40 early-game boost only fires for a team's first-ever 10 games, not per-season. | New-season uncertainty (roster turnover, coaching changes) is not captured for returning teams. |
| **G3 — Skellam parameters never swept** | `XG_ASYMMETRY_FACTOR` (0.75), `MIN_XG` (0.2), `SKELLAM_TAIL_RANGE` (20) were hand-set. | These directly control win/draw/loss probability calculation — the scoring metric's input. |

Additionally, we now have **four full seasons** (2022–2025) of completed data, whereas the original search only had three (2022–2024) with 2025 as a blind holdout. This extra season provides a richer validation set.

---

## 2. Data Inventory

| Season | First Grade | Reserve Grade | Total |
|--------|-------------|---------------|-------|
| 2022   | 93          | 83            | 176   |
| 2023   | 94          | 94            | 188   |
| 2024   | 93          | 90            | 183   |
| 2025   | 94          | 94            | 188   |
| **Total** | **374**  | **361**       | **735** |

Source: `data/processed/all_seasons.csv`

---

## 3. Validation Methodology

### 3.1 Walk-Forward Protocol

The original protocol (train 2022–23, validate 2024, holdout 2025) is updated to exploit all four seasons via **expanding-window walk-forward cross-validation**:

| Fold | Train | Validate | Purpose |
|------|-------|----------|---------|
| F1   | 2022  | 2023     | Can the model generalise from a single season? |
| F2   | 2022–2023 | 2024 | Original training window → next-year accuracy |
| F3   | 2022–2024 | 2025 | Most data → truest out-of-sample test |

**Primary metric:** Mean Brier score across F1, F2, F3 (weighted equally).

**Why expanding-window, not sliding-window?** The Elo system is inherently cumulative — team ratings carry forward. Dropping early seasons would discard the foundational calibration period. Expanding-window mirrors production reality.

**Why equal-weight folds?** Each fold tests a different aspect: F1 tests cold-start generalisation, F2 tests mid-life accuracy, F3 tests mature accuracy. Weighting toward F3 would overfit to "lots of data" scenarios and undervalue early-season robustness.

### 3.2 Scoring Metrics

| Metric | Formula | Role |
|--------|---------|------|
| **Multi-class Brier** (primary) | $\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \{W,D,L\}}(p_{ic} - o_{ic})^2$ | Measures probability calibration. Lower = better. |
| **Log-loss** (secondary) | $-\frac{1}{N}\sum_{i=1}^{N}\sum_{c}\; o_{ic}\log(p_{ic})$ | Heavily penalises confident wrong predictions. Guards against overconfident models. |
| **ECE** (diagnostic) | $\sum_{b=1}^{B}\frac{n_b}{N}\lvert\bar{p}_b - \bar{o}_b\rvert$ | Expected calibration error — are 70% predictions correct 70% of the time? |
| **Accuracy** (informational) | % of matches where $\arg\max(p_W, p_D, p_L)$ matches actual outcome | Useful for intuition but **not** an optimisation target (insensitive to probability quality). |

### 3.3 Between-Season Regression

At the end of each training season, all team Elo ratings are regressed toward `BASE_ELO`:

$$\text{elo}_{\text{new}} = \text{BASE\_ELO} + (\text{elo}_{\text{old}} - \text{BASE\_ELO}) \times (1 - r)$$

where $r$ is `PRIOR_REGRESSION_FACTOR`. This is a searched parameter.

### 3.4 Per-Season `played` Reset (New)

**Hypothesis under test:** Resetting `team.played` to 0 at the start of each season — so the K=40→30 transition applies fresh each year — will improve early-season predictions because returning teams face genuine uncertainty (roster turnover, off-season changes).

This is tested as a **boolean parameter** `RESET_PLAYED_PER_SEASON ∈ {True, False}` in the grid search.

---

## 4. Parameter Space

### 4.1 Elo Update Parameters

| Parameter | Current | Search Range | Rationale |
|-----------|---------|-------------|-----------|
| `K_FACTOR_INITIAL` | 40 | {30, 40, 50, 60} | **Never searched.** Controls convergence speed for unsettled teams. Higher = faster but noisier. |
| `K_FACTOR_SETTLED` | 30 | {20, 25, 30, 35} | Previously searched {15,20,25,30}. Extending upward to test if grassroots volatility warrants more reactivity. |
| `K_TRANSITION_GAMES` | 10 | {5, 8, 10, 15} | **Never searched.** How many games before a team is "settled". |
| `HOME_FIELD_ADVANTAGE` | 50 | {30, 40, 50, 60} | Previously searched {30,40,50}. Extending to 60 based on observed home win rates in the data. |
| `MOV_C1` | 0.001 | {0.0005, 0.001, 0.002} | Dampening coefficient for expected blowouts. Previously searched. |
| `MOV_C2` | 3.0 | {2.0, 3.0, 4.0} | Baseline dampening denominator. Extending from {2.2, 3.0} to include 4.0. |
| `PRIOR_REGRESSION_FACTOR` | 0.2 | {0.1, 0.2, 0.3, 0.4} | How much to regress toward 1500 between seasons. 0.1 = 90% retention, 0.4 = 60% retention. |
| `RESET_PLAYED_PER_SEASON` | False | {True, False} | **New.** Whether to reset `team.played` each season. |

### 4.2 Skellam Prediction Parameters

| Parameter | Current | Search Range | Rationale |
|-----------|---------|-------------|-----------|
| `ELO_TO_GOAL_RATIO` | 75 | {50, 75, 100, 125} | Previously searched. Maps Elo gap to expected goal difference. |
| `XG_ASYMMETRY_FACTOR` | 0.75 | {0.5, 0.65, 0.75, 0.85, 1.0} | **Never searched.** Controls how aggressively xG splits between home and away. Higher = more decisive predictions. |
| `MIN_XG` | 0.2 | {0.1, 0.2, 0.3} | **Never searched.** Floor for expected goals. Too low → Skellam edge cases; too high → compressed predictions. |

### 4.3 Fixed Parameters (Not Searched)

| Parameter | Value | Why Fixed |
|-----------|-------|-----------|
| `BASE_ELO` | 1500 | Convention. Shifting it changes nothing (all relative). |
| `SKELLAM_TAIL_RANGE` | 20 | Evaluating PMF from -20 to +20 goal difference captures >99.99% of probability mass in any realistic grassroots match. |
| `LEAGUE_AVG_GOALS` | 7.0 | Overridden at runtime by adaptive computation. Only used as fallback for the first match. |
| `USE_ADJ_RATES` | True | Previously searched. True won decisively. No reason to re-test. |

### 4.4 Total Search Space

$$4 \times 4 \times 4 \times 4 \times 3 \times 3 \times 4 \times 2 \times 4 \times 5 \times 3 = 442{,}368 \text{ combinations}$$

This is too large for exhaustive search. We will use a **two-stage approach** (see §5).

---

## 5. Search Strategy

### Stage 1 — Coarse Random Search (Target: ~2,000 evaluations)

**Method:** Latin Hypercube Sampling (LHS) across the full parameter space. LHS ensures even coverage of each parameter's marginal distribution while keeping evaluation count tractable.

**Why not pure grid?** At 442K combinations × 3 folds × ~735 matches, exhaustive search would take days. LHS with 2,000 samples provides excellent coverage at <1% of the cost.

**Output:** Top 50 configurations by mean Brier score. Identify which parameters have the most influence via **marginal Brier plots** (one-parameter-at-a-time analysis).

### Stage 2 — Refined Grid Search (Target: ~3,000 evaluations)

**Method:** Take the top-performing region from Stage 1 and run a full grid over a narrowed parameter range (±1 step from the best values). This confirms the optimum is genuine and not a sampling artifact.

**Output:** Champion configuration with Brier, log-loss, ECE, and accuracy per fold.

### Stage 3 — Statistical Validation

For the champion configuration:

1. **Per-fold Brier breakdown** — ensure no single fold is carrying the average.
2. **Comparison to current production** — run the current params through the same 3-fold protocol for an apples-to-apples comparison.
3. **Bootstrap confidence interval** — resample each fold's predictions 1,000 times to compute a 95% CI on the Brier score improvement. If the CI includes zero, the improvement is not significant and we keep the current parameters.
4. **Calibration plot** — bucket predictions into deciles and plot predicted vs actual win rate. A well-calibrated model should track the diagonal.
5. **Reserve grade cross-check** — run the champion first-grade params on reserve grade data. If Brier degrades significantly, the params are overfitting to first-grade dynamics and we should consider grade-specific tuning.

---

## 6. Implementation Plan

### 6.1 Script: `optimise_v2.py`

```
Usage:  python optimise_v2.py [--stage 1|2|3] [--grade first_grade|reserve_grade]
```

**Architecture:**
- Reads `data/processed/all_seasons.csv`
- For each parameter combination:
  - Creates a fresh `GrassrootsEloEngine` with the candidate constants monkey-patched
  - Runs the 3-fold expanding-window backtest
  - Logs Brier, log-loss, accuracy per fold + mean
- Outputs to `backtest_logs/v2_stage{N}_{grade}.csv`
- Parallelised via `concurrent.futures.ProcessPoolExecutor` (each evaluation is independent)

**Key design decision:** The engine reads constants from `config.constants` at import time. To test different parameter values, we will pass them as a dict and override the module-level constants within each worker process. This avoids modifying the production constants file during the search.

### 6.2 Modifications to `engine/elo.py`

A single change is needed: support for per-season `played` reset. This will be implemented as an optional `reset_season_stats()` method on the engine that zeros `team.played` for all teams (preserving Elo, attack/defence rates, and all other state). The between-season regression step in the optimiser will call this conditionally based on `RESET_PLAYED_PER_SEASON`.

No other engine changes are required — the optimiser overrides constants externally.

### 6.3 Outputs

| File | Contents |
|------|----------|
| `backtest_logs/v2_stage1_first_grade.csv` | LHS results: all params + per-fold Brier + mean Brier |
| `backtest_logs/v2_stage2_first_grade.csv` | Refined grid results |
| `backtest_logs/v2_stage1_reserve_grade.csv` | LHS results for reserve grade |
| `backtest_logs/v2_champion_comparison.csv` | Old vs new champion, per-fold breakdown |
| `backtest_logs/v2_bootstrap_ci.csv` | Bootstrap Brier CIs |

---

## 7. Decision Framework

After Stage 3, the decision to adopt new parameters follows this flowchart:

1. **Is the mean Brier improvement > 0?** If no → keep current parameters.
2. **Is the bootstrap 95% CI entirely below the current Brier?** If no → improvement is not statistically significant. Keep current parameters unless the new config is simpler (fewer parameters or more intuitive values).
3. **Does reserve grade also improve (or at least not degrade)?** If reserve grade degrades by >0.01 Brier → consider grade-specific parameters.
4. **Is any single fold dramatically worse?** If one fold's Brier is >10% worse than the current config on that fold → the improvement is unstable. Investigate before adopting.
5. **All checks pass** → adopt new parameters, update `config/constants.py`, re-run `run_audit.py` to regenerate the audit report, and re-generate 2026 priors with `generate_2026_priors.py`.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting to 4 seasons of ~370 first-grade matches | The expanding-window protocol ensures each fold validates on unseen data. Bootstrap CIs quantify uncertainty. Reserve grade cross-check catches first-grade-specific overfitting. |
| `RESET_PLAYED_PER_SEASON=True` may hurt late-season accuracy by keeping K too high | The grid search will reveal this — if True wins on F1 (cold-start) but loses on F3 (mature), the trade-off is visible in the per-fold breakdown. |
| Skellam parameter changes may break display xG aesthetics | Display xG is a cosmetic layer (`_display_xg()` in predictions.py) independent of the model. No impact on scoring metrics. |
| Original optimisation scripts are deleted — we can't reproduce the exact D1 methodology | This plan defines a new, more rigorous methodology. The D1 results in `backtest_logs/d1_grid_results.csv` serve as a historical benchmark, not a dependency. |
| LHS may miss narrow optima | Stage 2 refined grid around the LHS winner catches this. If the refined grid's best is far from the LHS winner, we expand the refined range and re-run. |

---

## 9. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Mean Brier score (3-fold) | < 0.496 (current production) |
| No single fold Brier | > 0.55 |
| Reserve grade Brier | ≤ current + 0.01 (no significant degradation) |
| Bootstrap 95% CI upper bound | < current production Brier |
| ECE | < 0.05 |

---

## 10. Timeline

| Step | Description |
|------|-------------|
| **S1** | Implement `optimise_v2.py` with LHS sampling and 3-fold protocol |
| **S2** | Run Stage 1 (LHS, ~2,000 evals) on first grade |
| **S3** | Analyse Stage 1 results, define refined grid |
| **S4** | Run Stage 2 (refined grid, ~3,000 evals) on first grade |
| **S5** | Run Stage 3 (bootstrap CIs, reserve grade cross-check, calibration plots) |
| **S6** | Apply decision framework (§7). If adopting: update constants, re-run audit, regenerate priors |

---

## Appendix A — Current Production Parameters (Baseline)

```python
BASE_ELO               = 1500
K_FACTOR_INITIAL        = 40
K_FACTOR_SETTLED        = 30
K_TRANSITION_GAMES      = 10
HOME_FIELD_ADVANTAGE    = 50
MOV_C1                  = 0.001
MOV_C2                  = 3.0
ELO_TO_GOAL_RATIO       = 75
XG_ASYMMETRY_FACTOR     = 0.75
MIN_XG                  = 0.2
PRIOR_REGRESSION_FACTOR = 0.2
RESET_PLAYED_PER_SEASON = False  # (implicit — not currently implemented)
```

Production Brier (2025 holdout, first grade): **0.496**

## Appendix B — Previous D1 Grid Search Winner

From `backtest_logs/d1_grid_results.csv` (row 1, sorted by Brier ascending):

```
K_SETTLED=30, HFA=50, ELO_TO_GOAL_RATIO=75, REGRESSION_FACTOR=0.8,
MOV_C1=0.001, MOV_C2=3.0, USE_ADJ_RATES=True → Brier=0.5755
```

Note: The D1 Brier (0.5755) differs from the production audit Brier (0.496) because D1 was evaluated on a different fold structure (single-fold validation on 2024). The 0.496 figure comes from `run_audit.py`'s walk-forward over 2025 only.
