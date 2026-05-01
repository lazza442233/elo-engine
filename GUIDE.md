# Grassroots Football Elo Engine

A **Skellam-Elo hybrid rating system** built for amateur football. It ingests live match data from the Dribl API, computes Elo ratings for every team, and predicts upcoming match outcomes using a Skellam goal-difference distribution. Every parameter has been empirically optimized against four seasons (2022–2025) of walk-forward validated data.

**Brier Score: 0.479** on the 2025 holdout season (random guessing = 0.667).

---

## Getting started

### Run the CLI

```bash
python3 main.py                           # auto-detect round, first grade
python3 main.py --league prem-res         # reserve grade
python3 main.py --priors                  # carry forward last season's ratings
python3 main.py --round 8                 # specific round fixtures
python3 main.py --output json             # machine-readable output
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

The dashboard provides an interactive web UI with rankings, predictions for upcoming fixtures, and Elo trajectory charts. Use the sidebar to switch leagues, select rounds, and toggle carry-forward ratings.

### Run the test suite

```bash
python3 -m pytest tests/ -v              # 77 tests
```

---

## CLI reference

| Flag | Effect |
|---|---|
| `--priors` | Load carry-forward ratings from `data/end_of_season_elos_{grade}.json` |
| `--round N` | Fetch fixtures for round N (default: auto-detect) |
| `--league KEY` | League to analyse: `prem-men` (default) or `prem-res` |
| `--output json` | Machine-readable JSON output |
| `--export-ratings` | Save regressed end-of-season Elos to JSON for next season |
| `--export-history` | Save per-match Elo trajectory to CSV |
| `--log-results` | Log predictions for completed matches to calibration CSV |
| `--calibrate` | Show Brier score from previously logged predictions |

---

## Project structure

```
main.py                        CLI entry point
dashboard.py                   Streamlit web dashboard
run_audit.py                   Model performance audit script

config/
  constants.py                 All hyperparameters and league configs
  teams.py                     Team colours, abbreviations, badge paths

engine/
  elo.py                       GrassrootsEloEngine — core rating & prediction
  calibration.py               Brier score computation

models/
  team.py                      Team data model with adaptive league avg goals

api/
  client.py                    Dribl API client

display/
  output.py                    Terminal rendering (rankings + predictions)

dashboard/
  data.py                      Data loading with Streamlit caching
  helpers.py                   Form dots, date parsing, closeness scoring
  components/
    header.py                  Stats bar (leader, biggest swing, closest matchup)
    rankings.py                League table with form indicators
    predictions.py             Prediction cards with confidence bars
    sidebar.py                 League selector, round control, priors toggle
    elo_history.py             Elo trajectory chart

persistence/
  db.py                        SQLite storage for matches and ratings

data/
  seasons.json                 API season/competition IDs
  processed/all_seasons.csv    Clean match data (2022–2025, both grades)
  raw/                         Immutable raw JSON from Dribl API
  end_of_season_elos_*.json    Carry-forward priors files

backtest_logs/                 Validation artifacts and audit logs
tests/test_elo.py              67 unit tests
```

---

## How the model works

### Elo ratings

Every team starts at **1500**. After each match, ratings shift based on three factors:

1. **Result** — Win, draw, or loss against the expected outcome
2. **Margin of victory** — Logarithmic scaling (`log(GD + 1)`), with diminishing returns for blowouts. An asymmetric dampener (`MoV / (|Δelo| × C1 + C2)`) reduces the bonus when the Elo gap already predicted the blowout
3. **Adaptive K-factor** — K=30 for a team's first 10 games (fast convergence), ramping linearly to K=35 once settled

**Home-field advantage** adds 30 Elo points to the home team's effective rating. This was empirically optimized from four seasons of data showing a consistent home edge in this league.

### Predictions

Match probabilities are computed via the **Skellam distribution** — the difference of two independent Poisson random variables (home goals minus away goals). This naturally handles the high-scoring, high-variance nature of grassroots football.

Each team's expected goals (xG) comes from blending two signals:
- **Opponent-adjusted rates**: team attack and defence rates normalised by the quality of opponents faced (Massey method)
- **Elo-derived xG**: the rating gap converted to an expected goal difference

The 50/50 blend of these two signals produces the final xG for each team, which feeds the Skellam PMF to generate win/draw/loss probabilities.

### Adaptive league average

The league-wide goals per game is **computed dynamically** from processed matches rather than using a fixed constant. This allows the model to self-correct when league scoring trends shift — for example, the 2025 season averaged 4.03 goals/game versus 3.37 in 2023.

### Carry-forward ratings

When enabled (via `--priors` on CLI or the dashboard toggle), teams retain **60% of their Elo delta** from the previous season. This:
- Dramatically improves early-season predictions by starting with informed ratings
- Naturally decays — each new match result dilutes the prior
- Automatically filters out teams no longer in the competition
- Seeds new entrants at a manually assessed starting level

Priors are stored in `data/end_of_season_elos_first_grade.json` and `data/end_of_season_elos_reserve_grade.json`.

---

## Season maintenance workflow

### During the season

The system is fully automatic once running. The Dribl API provides live match data, and ratings update after every completed round.

### End of season

```bash
python3 main.py --export-ratings                    # first grade (default)
python3 main.py --export-ratings --league prem-res  # reserve grade
```

This writes regressed Elo ratings to `data/end_of_season_elos.json`. Rename this file to the appropriate grade-specific path before the next season starts.

### New season checklist

1. Verify the `season` ID in `config/constants.py` matches the new Dribl season
2. Run `--export-ratings` at end of previous season if not already done
3. For new teams entering the competition, add their starting Elo to the priors JSON file:
   - Strong entrants from other competitions: **1540–1550**
   - Promoted from lower division: **1500** (league average)
4. Run `python3 run_audit.py` at the end of each season to assess model health

---

## Hyperparameters

All constants live in `config/constants.py`. These have been optimized via a 2,000-sample Latin Hypercube search followed by a refined grid, with 3-fold expanding-window walk-forward validation (train 2022–2023, validate 2024, holdout 2025).

| Parameter | Value | Purpose |
|---|---|---|
| `BASE_ELO` | 1500 | Starting rating for all teams |
| `K_FACTOR_INITIAL` | 30 | K-factor for first 10 games (fast convergence) |
| `K_FACTOR_SETTLED` | 35 | K-factor after 10 games (stability) |
| `K_TRANSITION_GAMES` | 10 | Games to transition from initial to settled K |
| `HOME_FIELD_ADVANTAGE` | 30 | Elo points added for home team |
| `MOV_C1` | 0.001 | Asymmetric MoV dampener — Elo gap coefficient |
| `MOV_C2` | 2.0 | Asymmetric MoV dampener — baseline denominator |
| `ELO_TO_GOAL_RATIO` | 75 | Elo points per expected goal difference |
| `XG_ASYMMETRY_FACTOR` | 0.85 | How aggressively xG splits between home/away |
| `PRIOR_REGRESSION_FACTOR` | 0.4 | Retain 60% of Elo delta across seasons |
| `LEAGUE_AVG_GOALS` | 7.0 | Initial seed (overridden adaptively at runtime) |
| `MIN_XG` | 0.2 | Floor for expected goals |
| `SKELLAM_TAIL_RANGE` | 20 | Max goal difference evaluated in Skellam PMF |

**Do not change these casually.** They were chosen to minimize the Brier score across four seasons of data. If you suspect drift, run `python3 run_audit.py` before re-tuning.

---

## Understanding the output

### Rankings table

| Column | Meaning |
|---|---|
| **Elo** | Current rating. 1500 = league average |
| **Drift** | Position change since last round |
| **Form** | Last 5 results (green = win, grey = draw, red = loss) |
| **MP / W / D / L** | Matches played, wins, draws, losses |
| **GF / GA / GD** | Goals for, against, difference |
| **PTS** | League points (3 for win, 1 for draw) |

### Match predictions

| Field | Meaning |
|---|---|
| **Win / Draw / Loss %** | Probabilities from the Skellam distribution |
| **Confidence label** | STRONG (>75%), LIKELY (60–75%), LEAN (50–60%), TOSS-UP (<50%) |
| **Δ Elo** | Rating gap between the two teams |
| **xG** | Expected goals per team. For extreme mismatches (>95% win prob), a display adjustment is applied (marked ⓘ) — the underlying model prediction remains unchanged |

---

## Match filtering

The engine skips matches with status: `forfeit`, `abandoned`, `postponed`, `upcoming`, or `bye`. Forfeits are excluded intentionally — they inflate ratings without a ball being kicked.

---

## Model audit

Run a comprehensive performance audit across all historical seasons:

```bash
python3 run_audit.py
python3 run_audit.py --grade reserve_grade
```

This produces:
- Per-season Brier score, log-loss, and accuracy
- Calibration analysis (expected calibration error)
- League dynamics comparison (home win %, goals/game, draw %)
- End-of-season Elo distribution statistics
- Automated health assessment and recommendation

Audit logs are saved to `backtest_logs/audit_{grade}_{year}_log.csv`.

---

## Data pipeline

1. **Historical base**: `data/processed/all_seasons.csv` provides the walk-forward training and audit dataset
2. **Live**: `main.py` / `dashboard.py` fetch current season data directly from the Dribl API
3. **Replay**: normalized match records are processed through the shared kickoff-aware replay path
4. **Persist**: cached completed matches plus replay metadata are stored in SQLite (`data/elo_engine.db`) for offline replay

---

## Known limitations

- **Early-season noise**: With fewer than ~8 rounds played, ratings are volatile. Carry-forward ratings significantly mitigate this
- **xG ceiling**: The Skellam model intentionally caps xG to preserve global calibration. A presentation-layer adjustment handles extreme mismatches
- **New teams**: Teams entering mid-season or from unknown competitions start at 1500 unless manually seeded in the priors file
- **Single competition**: The model is calibrated for the Northern Districts Premier League. Applying it to a different league would require re-tuning hyperparameters
