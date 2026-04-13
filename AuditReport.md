# Analysis Report: Grassroots Football Elo Engine

### **Executive Summary**

The Grassroots Football Elo Engine is a deeply considered, scientifically rigorous rating and prediction system for amateur football — a project born from the conviction that the analytical respect afforded to elite sport should be available to the local game. In its current state, it is a remarkably well-validated statistical engine (Brier 0.496 on holdout, 63 passing tests, walk-forward discipline across four seasons) surrounded by an increasingly capable but structurally adolescent operational shell. The most critical recommendation is to formalize the project's deployment and lifecycle infrastructure — version control, CI, environment isolation — so that the engine's intellectual maturity is matched by the resilience of the system that delivers it.

---

### **Part I: The Soul of the Project**

*   **Stated Mission:** "A Skellam-Elo hybrid rating system built for amateur football. It ingests live match data from the Dribl API, computes Elo ratings for every team, and predicts upcoming match outcomes using a Skellam goal-difference distribution. Every parameter has been empirically optimized against four seasons of walk-forward validated data."

*   **Inferred Animating Purpose:** To prove — to oneself and to one's community — that grassroots football is worthy of the same quantitative care and intellectual honesty that professionals demand of elite leagues, and that a single person with the right methodology can build something genuinely predictive from the noise of amateur sport.

*   **The Core Question:** Can a solo-built, first-principles statistical model extract reliable signal from the chaos of grassroots football, and can that signal be delivered to a community in a way that enriches how they understand and experience their local game?

---

### **Part II: The Anatomy of the Project — A Critical Assessment**

*   **Foundations (Strengths):**

    *   **Methodological Integrity — Walk-Forward Validation as Religion:** This project does not merely claim to be validated; it structurally *enforces* it. The strict temporal separation between tuning (2022–23), validation (2024), and holdout (2025) seasons — encoded in `run_audit.py` and `generate_2026_priors.py` — is a discipline many professional data science teams fail to maintain. The Brier score of 0.496 against a 0.667 random baseline is honest, meaningful, and reproducible. This is the project's deepest strength.

    *   **Three-Directive Validation Architecture:** The progression from D1 (refined Elo) through D2 (LightGBM feature model) to D3 (stacking ensemble), with each directive having its own grid search, holdout evaluation, and CSV artifact trail, shows a builder who thinks in terms of *competing hypotheses* rather than blind feature stuffing. The fact that the pure Elo model won and became production — with the ML directives serving as benchmarks rather than replacements — demonstrates uncommon restraint and intellectual honesty.

    *   **The Skellam-Elo Hybrid Design:** The fusion of Elo's rating dynamics with Skellam's goal-difference distribution is not a standard textbook formula — it's an original synthesis. The variable xG that blends opponent-adjusted attack/defence rates with Elo-derived expected goal difference, floored at `MIN_XG = 0.2`, shows an understanding that amateur football's high-scoring, high-variance nature requires models that respect fat tails. The asymmetric MoV dampening (`MoV / (|elo_diff| * C1 + C2)`) prevents the common failure mode of Elo systems in amateur sport: runaway ratings from predictable blowouts.

    *   **Comprehensive Test Suite:** 63 tests across 500+ lines covering everything from Elo conservation laws to SQLite persistence round-trips to Brier score computation — all passing in 1.16 seconds. The test design is remarkably mature: it tests *invariants* (conservation, symmetry, monotonicity) not just I/O, which means the tests actually catch regressions in mathematical correctness, not just interface breakage.

    *   **Clean Separation of Concerns:** The architecture — `engine/` for core logic, `models/` for data, `config/` for constants, `api/` for external data, `display/` for rendering, `persistence/` for storage, `dashboard/` for web UI — is textbook clean architecture for a Python project of this scale. No module reaches across boundaries inappropriately. The engine has zero knowledge of the dashboard; the dashboard imports the engine. This is a project that could be refactored without tearing it apart.

    *   **The Audit System:** `run_audit.py` is a production-grade model monitoring tool: it computes per-season Brier, log-loss, accuracy, expected calibration error (ECE), league dynamics drift, and Elo distribution statistics, then renders automated health verdicts (HEALTHY / REQUIRES RECALIBRATION / DEGRADED). Many deployed ML systems lack anything this thorough.

*   **Fissures (Weaknesses):**

    *   **No Version Control:** The project is not a git repository. For a codebase of over 5,600 lines with validated hyperparameters, immutable data artifacts, and a walk-forward backtest system built on reproducibility, the absence of git is a critical structural gap. A single accidental edit to `config/constants.py` could invalidate all prior validation with no way to detect or revert it.

    *   **No CI or Automated Quality Gates:** The 63 tests exist but run only when someone remembers to invoke pytest. There is no pre-commit hook, no GitHub Actions workflow, no automated lint pass. The test suite is a loaded gun in a drawer — effective only if you pick it up.

    *   **Implicit Dependency Management:** `requirements.txt` lists 6 packages (scipy, streamlit, altair, pandas, pytest) but `run_post_validation.py` uses lightgbm and sklearn, which are undeclared. There is no virtual environment lockfile (`poetry.lock`, `uv.lock`, or `pip freeze`). On a different machine or a future Python version, the project may not reproduce.

    *   **Orphaned / Dead Code:** `backtest.py`, `backtest_v2.py`, `optimize.py`, `run_post_validation.py`, `ingest_raw_data.py`, and `process_data.py` are top-level scripts referenced in the repo memory inventory but absent from the current workspace file tree. If they were removed intentionally, their ghosts haunt the documentation; if they exist elsewhere, the project boundary is unclear.

    *   **Single-Grade Audit Bias:** `run_audit.py` hardcodes `GRADE = "first_grade"`. Reserve grade has its own data, priors files, and league configuration, but receives no automated audit coverage. If reserve grade's league dynamics diverge (different team count, different scoring patterns), the model may silently degrade for that segment.

*   **Tension Points (Internal Conflicts):**

    *   **Reproducibility vs. Operational Fragility:** The engine's *intellectual* approach to reproducibility is exemplary (immutable raw data, deterministic walk-forward replay, logged artifacts). But the *operational* reproducibility is brittle: no version control, no environment pinning, no deployment pipeline. The project demands its data be immutable while allowing its own code to be mutable without audit trails. This tension is the project's central paradox.

    *   **Production Ambition vs. Solo-Operator Scale:** The codebase has production patterns — adaptive caching in Streamlit, SQLite offline fallback, exponential-backoff API retries, multi-league configuration, badge asset embedding. But it is maintained by one person running scripts manually from a terminal. The sophistication of the engine pulls the project toward "product"; the operational reality keeps it in "personal tool." Neither is wrong, but the project has not explicitly chosen which it is, and this indecision creates friction (e.g., building a full offline-mode SQLite fallback that no one else will ever trigger).

    *   **Grassroots vs. Generalization:** The model is exquisitely tuned for one league ecosystem (Parramatta & District, Premier League, ~12 teams, 4 seasons). Constants like `LEAGUE_AVG_GOALS = 7.0`, `HOME_FIELD_ADVANTAGE = 50`, and `ELO_TO_GOAL_RATIO = 75` are district-specific. The architecture *looks* general (multi-league config, team identity registry), but the parameters are hyper-local. Applying this to another grassroots league would require a full re-optimization, and there's no tooling to support that.

    *   **Display-layer xG Inflation vs. Model Honesty:** The `_display_xg()` function in `predictions.py` deliberately inflates xG for extreme mismatches so the UI looks more "realistic" to users. This creates a subtle crack in the project's otherwise strict commitment to model honesty. A user seeing an inflated xG of 9.2 when the model actually computed 5.8 is being shown a lie — a well-intentioned one, but a lie nonetheless.

*   **Latent Potential (Hidden Gems):**

    *   **The `generate_2026_priors.py` Origin-Classification System:** The classification of teams into RETURNING, HIATUS, TRANSFERRED, and PROMOTED origins — each with a documented anchoring strategy — is a genuinely novel approach to the cold-start problem in sports rating systems. This taxonomy could be formalized into the engine itself as a first-class prior injection strategy, rather than living in a standalone script.

    *   **The Opponent-Adjusted Rate System (Massey Method):** The `adj_attack_rate` and `adj_defence_rate` properties, accumulated in real-time during `process_match()`, are essentially a lightweight strength-of-schedule adjustment baked into the prediction at the team level. This is more sophisticated than most amateur rating systems and could be exposed as a standalone analytical feature (e.g., "Who has the hardest remaining schedule?").

    *   **The Calibration Infrastructure:** The `engine/calibration.py` module, combined with the calibration bucket analysis in `run_audit.py`, forms a complete expected-calibration-error pipeline. This infrastructure could power a live "model confidence dashboard" that shows users not just predictions, but how *trustworthy* the model has been historically in different probability ranges.

    *   **The Match Log as a Dataset:** `engine.match_log` records every processed match with scores, round labels, and teams. Combined with `elo_history`, this constitutes a complete, enriched match dataset that could power features like head-to-head history lookups, venue analysis, or form streaks — without re-querying the API.

---

### **Part III: The State of the Union**

*   **Current State Metaphor:** A Swiss chronometer movement running inside a hand-carved wooden case — the precision of the mechanism is world-class, but the housing that protects and presents it has not yet caught up.

*   **Narrative Assessment:**

    The Grassroots Football Elo Engine is, at its core, an act of applied epistemology: *What can we actually know about grassroots football, and how rigorously can we know it?* The answer this project gives — a validated Brier score of 0.496, earned through honest walk-forward testing, with no leakage, no cherry-picking, and no ensemble inflation — is a compelling one. The model works. It works because its builder understood that the hardest part of prediction isn't building a complex model; it's building a simple one that you can trust.

    The engine layer reflects this philosophy perfectly. The Skellam-Elo hybrid is elegant: Elo handles the macro dynamics (team strength, momentum, home advantage), while the Skellam distribution handles the micro (goal-by-goal probability mass, fat tails for amateur blowouts). The opponent-adjusted rates add genuine analytical depth without overcomplicating the system. The adaptive K-factor respects the epistemic reality that new teams carry more uncertainty. The asymmetric MoV dampening solves a problem specific to grassroots sport that most Elo implementations ignore entirely. Every design choice has a clear justification, and the justification is usually traceable to a specific pathology of amateur football.

    Where the project falters is not in its *thinking* but in its *infrastructure*. There is no git repository, no CI, no environment lock, no deployment pipeline. The prior generation script for 2026 — a document of remarkable methodological care — was run from a terminal session. The audit system produces rich diagnostics but reports to no one; the dashboard serves data but is deployed nowhere persistent. The project has built a prediction engine that outperforms random chance by a statistically significant margin, and then housed it in a system where a single `rm -rf` would erase years of validated work with no recovery path. This is not a criticism of the builder's skill — it is a measure of how far the engine has outgrown its shell.

*   **The Core Challenge:** The project must evolve its operational maturity to match its intellectual maturity — giving the engine a durable, versioned, reproducible home that protects the validated work it contains and enables it to be shared, deployed, and extended without risk of silent regression.

---

### **Part IV: Navigating the Future — Counsel & Direction**

*   **Immediate Refinements (The Next 3 Steps):**

    1.  **Action:** Initialize git, commit the current state as a baseline, push to a private GitHub repository, and add a minimal GitHub Actions workflow that runs `pytest` on every push. **Rationale:** This is the single highest-leverage action available. It transforms the project from a collection of files into a versioned, recoverable artifact. Every future change becomes auditable. The test suite already exists and is fast (1.16s) — hooking it into CI is trivial and immediately protective. This also enables the prior generation outputs and backtest logs to be versioned alongside the code that produced them, closing the reproducibility gap.

    2.  **Action:** Pin all dependencies in a lockfile (e.g., `uv lock` or `pip freeze > requirements.lock`), including the implicit lightgbm/sklearn dependencies used by the post-validation system. Add a note to `requirements.txt` distinguishing core runtime deps from development/validation deps. **Rationale:** The project's most dangerous silent failure mode is a dependency update that changes numerical output. Scipy, numpy, and lightgbm all have version-sensitive numerical behavior. A lockfile lets you reproduce the exact environment that produced the validated Brier score of 0.496.

    3.  **Action:** Extend `run_audit.py` to audit reserve grade alongside first grade, either by parameterizing `GRADE` or by running the full pipeline for both. **Rationale:** Reserve grade is a first-class citizen in the config, the data pipeline, the priors system, and the dashboard — but has zero automated quality monitoring. If reserve grade's league dynamics differ enough to degrade predictions (plausible: different team count, different scoring patterns, different roster stability), you won't know until someone complains.

*   **Strategic Shifts (The Next Big Leap):**

    *   **Commit to identity: personal analytical tool or community product.** The project currently inhabits an uncomfortable middle ground. If it's a personal analytical tool, strip the offline SQLite fallback, the multi-league config complexity, and the badge system — they're overhead. If it's a community product, add persistent deployment (Streamlit Cloud or a VPS), a public URL, and a mechanism for users to see historical model performance alongside predictions (the calibration infrastructure is already there). The codebase's architecture already supports the "product" direction — the question is whether the builder wants to carry that operational burden, and what the community relationship would look like. Either choice is valid; the current non-choice is where energy is wasted.

    *   **Formalize the prior injection framework as a first-class engine feature.** The origin-classification system in `generate_2026_priors.py` (RETURNING, HIATUS, TRANSFERRED, PROMOTED) is the most innovative piece of applied rating theory in the project, yet it lives in a standalone script that runs once a year. Integrating this taxonomy into `GrassrootsEloEngine` — so that `inject_priors()` accepts not just `{team: elo}` but `{team: {elo, origin, anchor_method}}` — would make the cold-start handling a documented, testable, reproducible part of the engine rather than a manual process.

*   **Provocative Questions for the Creator:**

    1.  **Who is this for — and what changes if they see it?** Right now, the engine's predictions flow to a Streamlit dashboard and a CLI. If your teammates, opponents, or league organizers started using these predictions to inform decisions — team selection, match expectations, even league restructuring — how would that change your relationship to the model's accuracy and your responsibility for its outputs?

    2.  **What is the relationship between prediction and understanding?** The engine predicts outcomes with a Brier score of 0.496. But does a prediction of "Pennant Hills 62% / Draw 18% / Epping 20%" actually help someone *understand* their league? Is the purpose of this project to tell people what will happen, or to give them a richer language for talking about what *might* happen and *why*? The answer determines whether you invest in accuracy or in explanation.

    3.  **If you had to delete one-third of the codebase tomorrow, which third would you keep, and what does that tell you about where the project's soul actually lives?** The engine? The dashboard? The validation framework? The answer reveals what you're actually building — and what's accumulated around it as a byproduct of builder's momentum rather than intent.
