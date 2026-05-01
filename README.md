# Elo Engine

A match prediction system for grassroots football. It rates every team using a **Skellam-Elo hybrid model**, then predicts upcoming results — win/draw/loss probabilities and expected scorelines — via a live Streamlit dashboard.

Built for the **Northern Districts Premier League** (Sydney, Australia). All parameters are empirically optimised against four seasons (2022–2025) of walk-forward validated data.

**Brier Score: 0.479** on the 2025 holdout season (random guessing ≈ 0.667).

<!-- TODO: add a dashboard screenshot here
![Dashboard](assets/screenshots/dashboard.png)
-->

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd elo-engine
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run dashboard.py

# 3. Or use the CLI
python3 main.py                  # current round, first grade
python3 main.py --priors         # use carry-forward ratings from last season
```

The dashboard provides live rankings, match predictions with confidence levels, and full Elo trajectory charts for every team across all seasons.

---

## What happens when you run this?

1. **Fetches live data** from the [Dribl API](https://www.dribl.com/) — the platform used by this league for fixtures and results
2. **Replays every completed match** through the Elo engine, updating team ratings after each game
3. **Predicts upcoming fixtures** using a Skellam distribution over each team's expected goals
4. **Displays results** in an interactive dashboard (or CLI table)

No API key is required — the Dribl endpoints used are public.

---

## How it works

Every team starts at **1500 Elo**. After each match, ratings shift based on result, margin of victory (with diminishing returns for blowouts), and an adaptive K-factor that starts low for new teams.

Match predictions use the **Skellam distribution** — the difference of two Poisson variables — which naturally models the high-scoring, high-variance nature of amateur football. Each team's expected goals blend Elo-derived expectations with opponent-adjusted attacking and defensive rates.

At season boundaries, teams retain **60% of their Elo delta** (40% regression toward the mean), giving the model a head start on early-season predictions.

For the full technical breakdown, see [GUIDE.md](GUIDE.md).

---

## Usage

### Dashboard

```bash
streamlit run dashboard.py
```

Use the sidebar to switch between First Grade and Reserve Grade, select rounds, and toggle carry-forward ratings.

### CLI

```bash
python3 main.py                           # auto-detect round, first grade
python3 main.py --league prem-res         # reserve grade
python3 main.py --priors                  # carry-forward ratings from last season
python3 main.py --round 8                 # specific round
python3 main.py --output json             # machine-readable JSON
python3 main.py --calibrate               # show Brier score from logged predictions
python3 main.py --export-ratings          # save end-of-season Elos for next year
```

### Tests

```bash
python3 -m pytest tests/ -v              # 77 tests
```

### Model audit

```bash
python3 run_audit.py                     # audits both grades by default
python3 run_audit.py --grade first_grade # single-grade audit
```

---

## Project structure

```
main.py                 CLI entry point
dashboard.py            Streamlit dashboard
run_audit.py            Model performance audit

engine/
  elo.py                Core rating & prediction engine
  calibration.py        Brier score computation

config/
  constants.py          All hyperparameters
  teams.py              Team colours, abbreviations, badges

api/
  client.py             Dribl API client

dashboard/
  data.py               Data loading with Streamlit caching
  components/           UI tabs (rankings, predictions, Elo history)

data/
  processed/            Clean match data (2022–2025)
  raw/                  Immutable JSON from Dribl API
  *.json                Season config & carry-forward priors

persistence/
  db.py                 SQLite cache for normalized match replay data

tests/
  test_elo.py           77 unit tests (invariants, calibration, persistence)
```

---

## Data source

All match data comes from the [Dribl API](https://www.dribl.com/), the official platform for Northern Districts football fixtures and results. The API endpoints used are public and require no authentication.

Historical data (2022–2025) is stored in `data/processed/all_seasons.csv` and `data/raw/` for reproducibility. Live data is fetched on each run.

---

## For developers

See [GUIDE.md](GUIDE.md) for:
- Full hyperparameter reference and tuning methodology
- Season maintenance workflow (end-of-season export, new-season checklist)
- Model audit interpretation
- Known limitations and design decisions

Optimisation tooling lives in `optimise_v2.py` — a Latin Hypercube search with 3-fold expanding-window walk-forward validation. See [docs/OPTIMISATION-PLAN-2026-04-16.md](docs/OPTIMISATION-PLAN-2026-04-16.md) for the methodology.
