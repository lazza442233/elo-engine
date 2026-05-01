"""
Skellam-Elo Hybrid engine for grassroots football.

Design choices:
  - Adaptive K      : K=40 early (fast convergence), K=30 after 10 games
  - HFA = 50 pts    : empirically optimized via D1 grid search
  - Log MoV (dampened): diminishing returns via ln(GD+1), with
                        asymmetric dampening MoV/(|Δelo|·C1+C2) so
                        expected blowouts don't over-inflate ratings
  - Dixon-Coles multiplicative xG: λ = μ × α_att × β_def × γ_hfa
                                   with MLE simultaneous solver,
                                   Bayesian shrinkage, and winsorization
  - Skellam pmf     : better than Probit for high-scoring amateur football
"""

import json
import math
import statistics
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skellam

from config.constants import (
    BASE_ELO,
    ELO_TO_GOAL_RATIO,
    HFA_MULTIPLIER,
    HOME_FIELD_ADVANTAGE,
    K_FACTOR_INITIAL,
    K_FACTOR_SETTLED,
    K_TRANSITION_GAMES,
    LEAGUE_AVG_GOALS,
    MIN_XG,
    MOV_C1,
    MOV_C2,
    PRIOR_REGRESSION_FACTOR,
    RIDGE_LAMBDA,
    SHRINKAGE_GAMES_FULL_TRUST,
    SKELLAM_TAIL_RANGE,
    WINSORIZE_GOALS_CAP,
    XG_ASYMMETRY_FACTOR,
    XG_BLEND_WEIGHT,
)
from engine.calibration import log_prediction
from engine.match_record import MatchRecord, normalize_match_records, shorten_team_name
from models.team import Team


class GrassrootsEloEngine:
    """
    Skellam-Elo Hybrid engine for grassroots football.

    PRIORS — how to manually seed ratings before the season starts
    -------------------------------------------------------------
    Call inject_priors() with a dict of {team_name: starting_elo}.
    The easiest way to tune this is to look at the team list after
    the first run, then add entries here based on last season's
    table, coaching changes, etc.

    Example:
        engine.inject_priors({
            "Kellyville Kolts Soccer Club": 1560,   # last season's champs
            "Putney Rangers FC":            1430,   # newly promoted
        })

    Only teams present in the data need priors — everyone else
    defaults to BASE_ELO (1500).
    """

    def __init__(self):
        self.teams: dict[str, Team] = {}
        self.processed_matches: int = 0
        self._total_goals: int = 0  # running total for adaptive league avg
        self._match_goals: list[int] = []  # per-match total goals for median calc
        self.elo_history: list[dict[str, float]] = []  # snapshot after each match
        self.match_log: list[dict] = []  # per-match metadata
        self.initial_elos: dict[str, float] = {}  # starting Elos after priors
        self._state_version: int = 0
        self._prediction_context: dict | None = None
        self._prediction_context_version: int = -1

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @property
    def league_avg_goals(self) -> float:
        """Adaptive league-average goals per game, computed from processed matches."""
        total = getattr(self, "_total_goals", 0)
        if self.processed_matches == 0 or total == 0:
            return LEAGUE_AVG_GOALS
        return total / self.processed_matches

    @property
    def league_median_goals(self) -> float:
        """Median total goals per match — robust to blowout outliers."""
        match_goals = getattr(self, "_match_goals", [])
        if not match_goals:
            return LEAGUE_AVG_GOALS
        return float(statistics.median(match_goals))

    def _get_or_create(self, name: str) -> Team:
        if name not in self.teams:
            self.teams[name] = Team(name)
            self._invalidate_prediction_context()
        return self.teams[name]

    @staticmethod
    def _shorten_name(raw: str) -> str:
        return shorten_team_name(raw)

    def _invalidate_prediction_context(self):
        """Drop cached derived prediction state after any engine mutation."""
        self._state_version += 1
        self._prediction_context = None
        self._prediction_context_version = -1

    # ------------------------------------------------------------------ #
    # Adaptive K-factor                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def k_factor(games_played: int) -> float:
        """
        Adaptive K-factor: high early for fast convergence, lower once settled.
        Linear interpolation between K_FACTOR_INITIAL and K_FACTOR_SETTLED.
        """
        if games_played >= K_TRANSITION_GAMES:
            return K_FACTOR_SETTLED
        t = games_played / K_TRANSITION_GAMES
        return K_FACTOR_INITIAL + t * (K_FACTOR_SETTLED - K_FACTOR_INITIAL)

    # ------------------------------------------------------------------ #
    # Expected score (standard Elo)                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """P(team_a wins) given ratings, before HFA is applied."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    # ------------------------------------------------------------------ #
    # Margin-of-Victory multiplier                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def mov_multiplier(goal_diff: int, total_goals: int = 0) -> float:
        """
        Uncapped logarithmic MoV multiplier.
          - goal_diff is the raw score difference.
          - total_goals: for draws (goal_diff=0), a high-scoring draw
            carries more information than a 0-0.  A mild ln(total+1) boost
            differentiates them without overshadowing the dampening term.
          - No hard cap: massive grassroots blowouts (e.g. 15-0) are
            allowed to influence Elo via diminishing log returns.
        """
        effective_gd = abs(goal_diff)
        if effective_gd <= 1:
            if goal_diff == 0 and total_goals > 0:
                return 1.0 + 0.1 * math.log(total_goals + 1)
            return 1.0
        return math.log(effective_gd + 1)

    # ------------------------------------------------------------------ #
    # Priors injection                                                      #
    # ------------------------------------------------------------------ #

    def inject_priors(self, priors: dict[str, float], quiet: bool = False):
        """
        Manually seed starting Elo ratings before processing matches.

        Call this BEFORE process_matches() so the priors flow through
        every historical game.

        Args:
            priors: dict mapping team name (short, no league suffix)
                    to their desired starting Elo.
            quiet:  suppress console output (for JSON mode).
        """
        for name, elo in priors.items():
            team = self._get_or_create(name)
            team.elo = elo
            self.initial_elos[name] = elo
        self._invalidate_prediction_context()
        if not quiet:
            print(f"[Priors] Injected {len(priors)} team rating(s).")

    def reset_played(self):
        """Reset games-played counters for all teams (preserves Elo and rates).

        Call between seasons so the K-factor transition (K_INITIAL → K_SETTLED)
        applies fresh each year.
        """
        for team in self.teams.values():
            team.played = 0
        self._invalidate_prediction_context()

    # ------------------------------------------------------------------ #
    # Process a single match                                               #
    # ------------------------------------------------------------------ #

    def process_match(
        self,
        home_name: str,
        away_name: str,
        home_score: int,
        away_score: int,
        round_label: str | None = None,
        match_date: str | None = None,
    ):
        home = self._get_or_create(home_name)
        away = self._get_or_create(away_name)

        # Expected win probability (home gets HFA boost)
        elo_home_adj = home.elo + HOME_FIELD_ADVANTAGE
        e_home = self.expected_score(elo_home_adj, away.elo)
        1.0 - e_home

        # Actual score (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            s_home, _s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, _s_away = 0.0, 1.0
        else:
            s_home, _s_away = 0.5, 0.5

        # MoV multiplier with autocorrelation dampening
        # Dampens MoV bonus when the Elo gap already predicted the blowout
        raw_mult = self.mov_multiplier(home_score - away_score,
                                       total_goals=home_score + away_score)
        elo_diff = abs(elo_home_adj - away.elo)
        mult = raw_mult / (elo_diff * MOV_C1 + MOV_C2)

        # Adaptive K — average of both teams' K-factors
        k = (self.k_factor(home.played) + self.k_factor(away.played)) / 2.0

        # Capture opponent rates BEFORE updating stats (avoid current-match feedback)
        _half = self.league_avg_goals / 2.0
        _away_def = away.defence_rate if away.played > 0 else _half
        _away_att = away.attack_rate if away.played > 0 else _half
        _home_def = home.defence_rate if home.played > 0 else _half
        _home_att = home.attack_rate if home.played > 0 else _half

        # Elo update
        delta = k * mult * (s_home - e_home)
        home.elo += delta
        away.elo -= delta

        # Stats
        home.played += 1
        away.played += 1
        home.gf += home_score
        home.ga += away_score
        away.gf += away_score
        away.ga += home_score
        # Winsorized stats for rate/xG calculations
        home.gf_capped += min(home_score, WINSORIZE_GOALS_CAP)
        home.ga_capped += min(away_score, WINSORIZE_GOALS_CAP)
        away.gf_capped += min(away_score, WINSORIZE_GOALS_CAP)
        away.ga_capped += min(home_score, WINSORIZE_GOALS_CAP)

        if home_score > away_score:
            home.wins += 1
            away.losses += 1
        elif home_score < away_score:
            away.wins += 1
            home.losses += 1
        else:
            home.draws += 1
            away.draws += 1

        # Opponent-quality tracking (using pre-match rates)
        home.opp_def_sum += _away_def
        home.opp_att_sum += _away_att
        away.opp_def_sum += _home_def
        away.opp_att_sum += _home_att

        self.processed_matches += 1
        self._total_goals += home_score + away_score
        if not hasattr(self, "_match_goals"):
            self._match_goals = []
        self._match_goals.append(home_score + away_score)

        # Propagate adaptive league average to all team instances
        avg = self.league_avg_goals
        for t in self.teams.values():
            t.league_avg_goals = avg

        # Record Elo snapshot for history tracking
        self.elo_history.append({name: t.elo for name, t in self.teams.items()})

        # Record match metadata
        self.match_log.append({
            "home": home_name,
            "away": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "round": round_label,
            "date": match_date,
        })
        self._invalidate_prediction_context()

    # ------------------------------------------------------------------ #
    # Process full match list from API data                                #
    # ------------------------------------------------------------------ #

    def _group_match_batches(self, matches: list[MatchRecord]) -> list[list[MatchRecord]]:
        """Group normalized matches by exact kickoff timestamp."""
        if not matches:
            return []

        batches: list[list[MatchRecord]] = []
        current_batch: list[MatchRecord] = [matches[0]]
        current_kickoff = matches[0].kickoff

        for match in matches[1:]:
            if match.kickoff == current_kickoff:
                current_batch.append(match)
                continue

            batches.append(current_batch)
            current_batch = [match]
            current_kickoff = match.kickoff

        batches.append(current_batch)
        return batches

    def replay_matches(
        self,
        raw_data: Iterable[MatchRecord | dict],
        quiet: bool = False,
        collect_predictions: bool = False,
        log_calibration: bool = False,
    ) -> list[dict]:
        """Replay normalized matches in kickoff-aware batches."""
        matches, skipped = normalize_match_records(raw_data)
        replay_log: list[dict] = []

        for batch in self._group_match_batches(matches):
            for match in batch:
                self._get_or_create(match.home_team)
                self._get_or_create(match.away_team)

            if not collect_predictions and not log_calibration:
                for match in batch:
                    self.process_match(
                        match.home_team,
                        match.away_team,
                        match.home_score,
                        match.away_score,
                        round_label=match.round_label,
                        match_date=match.match_date,
                    )
                continue

            batch_entries: list[tuple[MatchRecord, dict, dict]] = []
            for match in batch:
                home_pre = self.teams[match.home_team].elo
                away_pre = self.teams[match.away_team].elo
                prediction = self.predict_match(match.home_team, match.away_team)

                if match.home_score > match.away_score:
                    outcome = 1
                elif match.home_score < match.away_score:
                    outcome = -1
                else:
                    outcome = 0

                replay_entry = {
                    "season": match.season,
                    "grade": match.grade,
                    "round": match.round_number if match.round_number is not None else match.round_label,
                    "round_label": match.round_label,
                    "home": match.home_team,
                    "away": match.away_team,
                    "home_win": prediction["home_win"],
                    "draw": prediction["draw"],
                    "away_win": prediction["away_win"],
                    "prob_win": prediction["home_win"],
                    "prob_draw": prediction["draw"],
                    "prob_loss": prediction["away_win"],
                    "xg_home": prediction["xg_home"],
                    "xg_away": prediction["xg_away"],
                    "expected_gd": prediction["expected_gd"],
                    "actual_home_goals": match.home_score,
                    "actual_away_goals": match.away_score,
                    "outcome": outcome,
                    "home_elo_pre": home_pre,
                    "away_elo_pre": away_pre,
                    "date": match.match_date,
                }
                batch_entries.append((match, replay_entry, prediction))

                if log_calibration:
                    log_prediction(
                        prediction,
                        match.home_team,
                        match.away_team,
                        match.home_score,
                        match.away_score,
                    )

            for match, _, _ in batch_entries:
                self.process_match(
                    match.home_team,
                    match.away_team,
                    match.home_score,
                    match.away_score,
                    round_label=match.round_label,
                    match_date=match.match_date,
                )

            if collect_predictions:
                replay_log.extend(entry for _, entry, _ in batch_entries)

        if skipped and not quiet:
            print(f"[Ingest] Skipped {skipped} match(es) (bye/forfeit/invalid).")

        return replay_log

    def process_matches(
        self,
        raw_data: Iterable[MatchRecord | dict],
        quiet: bool = False,
        log_calibration: bool = False,
    ):
        """Ingest and replay matches, optionally logging calibration on the fly."""
        self.replay_matches(
            raw_data,
            quiet=quiet,
            collect_predictions=False,
            log_calibration=log_calibration,
        )

    # ------------------------------------------------------------------ #
    # Dixon-Coles MLE solver                                               #
    # ------------------------------------------------------------------ #

    def _solve_attack_defence(self) -> tuple[dict[str, float], dict[str, float]]:
        """
        Simultaneously fit attack (α) and defence (β) multipliers for all
        teams via maximum-likelihood estimation on a Poisson log-linear model
        with L2 (Ridge) regularisation.

        Returns (attack_dict, defence_dict) where values are multipliers
        relative to 1.0 (league average).  e.g. α=1.5 → 50 % above average.
        """
        team_names = sorted(self.teams.keys())
        n = len(team_names)
        idx = {name: i for i, name in enumerate(team_names)}

        if n == 0 or not self.match_log:
            return {}, {}

        baseline = self.league_median_goals / 2.0

        # Build match arrays (using capped goals)
        home_idx, away_idx, home_goals, away_goals = [], [], [], []
        for m in self.match_log:
            hi, ai = idx.get(m["home"]), idx.get(m["away"])
            if hi is None or ai is None:
                continue
            home_idx.append(hi)
            away_idx.append(ai)
            home_goals.append(min(m["home_score"], WINSORIZE_GOALS_CAP))
            away_goals.append(min(m["away_score"], WINSORIZE_GOALS_CAP))

        home_idx = np.array(home_idx)
        away_idx = np.array(away_idx)
        home_goals = np.array(home_goals, dtype=float)
        away_goals = np.array(away_goals, dtype=float)

        # x = [α_0 .. α_{n-1}, β_0 .. β_{n-1}]  (log-space for unconstrained optimisation)
        x0 = np.zeros(2 * n)

        def neg_log_lik(x):
            log_alpha = x[:n]
            log_beta = x[n:]

            # λ_home = baseline × α_home × β_away × HFA
            log_lam_h = (
                np.log(baseline)
                + log_alpha[home_idx]
                + log_beta[away_idx]
                + np.log(HFA_MULTIPLIER)
            )
            # λ_away = baseline × α_away × β_home
            log_lam_a = (
                np.log(baseline)
                + log_alpha[away_idx]
                + log_beta[home_idx]
            )

            lam_h = np.exp(np.clip(log_lam_h, -10, 5))
            lam_a = np.exp(np.clip(log_lam_a, -10, 5))

            # Poisson log-likelihood: y*log(λ) - λ - log(y!)
            ll = (
                home_goals * np.log(lam_h + 1e-10) - lam_h
                + away_goals * np.log(lam_a + 1e-10) - lam_a
            )

            # L2 regularisation — pull α, β toward 0 (i.e. multiplier = 1.0)
            penalty = RIDGE_LAMBDA * (np.sum(log_alpha ** 2) + np.sum(log_beta ** 2))

            return -np.sum(ll) + penalty

        result = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 200})
        log_alpha = result.x[:n]
        log_beta = result.x[n:]

        attack = {name: float(np.exp(log_alpha[i])) for i, name in enumerate(team_names)}
        defence = {name: float(np.exp(log_beta[i])) for i, name in enumerate(team_names)}
        return attack, defence

    def _get_prediction_context(self) -> dict:
        """Return cached derived state used across repeated predictions."""
        if (
            self._prediction_context is not None
            and self._prediction_context_version == self._state_version
        ):
            return self._prediction_context

        attack_mults, defence_mults = self._solve_attack_defence()
        context = {
            "attack_mults": attack_mults,
            "defence_mults": defence_mults,
            "baseline": self.league_median_goals / 2.0,
        }
        self._prediction_context = context
        self._prediction_context_version = self._state_version
        return context

    # ------------------------------------------------------------------ #
    # Bayesian shrinkage                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _shrinkage_weight(games_played: int) -> float:
        """
        How much to trust observed data vs prior expectation.
        Returns 0.0 (all prior) → 1.0 (all observed).
        """
        return min(1.0, games_played / SHRINKAGE_GAMES_FULL_TRUST)

    def _shrink_multiplier(self, alpha_raw: float, beta_raw: float, team_elo: float, games_played: int) -> tuple[float, float]:
        """Shrink attack and defence multipliers toward the team's Elo-implied prior."""
        # 1. Calculate a much stronger Prior based on Elo
        # (A 100 Elo point difference now equals a 25% shift, not 10%)
        prior_attack = 1.0 + ((team_elo - BASE_ELO) / 400.0)

        # 2. Safely clamp the attack prior so it doesn't spiral to infinity
        prior_attack = max(0.5, min(2.5, prior_attack))

        # 3. Calculate Defense Prior using mathematical inversion (never negative)
        prior_defense = 1.0 / prior_attack

        # 4. Calculate Weight
        weight = self._shrinkage_weight(games_played)

        # 5. Apply Shrinkage
        alpha_final = (weight * alpha_raw) + ((1.0 - weight) * prior_attack)
        beta_final = (weight * beta_raw) + ((1.0 - weight) * prior_defense)

        return alpha_final, beta_final

    # ------------------------------------------------------------------ #
    # Skellam prediction (v3: Dixon-Coles multiplicative)                  #
    # ------------------------------------------------------------------ #

    def predict_match(
        self, home_name: str, away_name: str, neutral: bool = False
    ) -> dict:
        """
        Predict Win/Draw/Loss probabilities using the Skellam distribution
        with Dixon-Coles multiplicative xG model.

        xG derivation (v3):
          1. MLE solver fits attack (α) and defence (β) multipliers for all
             teams simultaneously from winsorized match results.
          2. Bayesian shrinkage pulls α/β toward 1.0 when sample is small.
          3. xG = median_baseline × α_att × β_def × HFA (multiplicative).
          4. Blend with Elo-derived xG for robustness.
          5. Feed into Skellam PMF for W/D/L probabilities.

        Returns:
            dict with keys: home_win, draw, away_win (sum ≈ 1.0),
            plus xg_home, xg_away, expected_gd.
        """
        home = self._get_or_create(home_name)
        away = self._get_or_create(away_name)

        hfa = 0 if neutral else HOME_FIELD_ADVANTAGE

        # ── Step 1: Elo-to-expected goal difference ──
        elo_diff = (home.elo + hfa) - away.elo
        expected_gd = elo_diff / ELO_TO_GOAL_RATIO

        # ── Step 2: Dixon-Coles multiplicative xG ──
        context = self._get_prediction_context()
        attack_mults = context["attack_mults"]
        defence_mults = context["defence_mults"]
        baseline = context["baseline"]

        # ── Step 3: Elo-derived xG anchor (uses median baseline) ──
        elo_home_xg = baseline + (expected_gd * XG_ASYMMETRY_FACTOR)
        elo_away_xg = baseline - (expected_gd * XG_ASYMMETRY_FACTOR)

        if attack_mults and home_name in attack_mults:
            # Raw MLE multipliers
            alpha_home = attack_mults[home_name]
            beta_away = defence_mults[away_name]
            alpha_away = attack_mults[away_name]
            beta_home = defence_mults[home_name]

            # Apply Elo-aware Bayesian shrinkage
            alpha_home, beta_home = self._shrink_multiplier(alpha_home, beta_home, home.elo, home.played)
            alpha_away, beta_away = self._shrink_multiplier(alpha_away, beta_away, away.elo, away.played)

            hfa_mult = HFA_MULTIPLIER if not neutral else 1.0
            dc_home_xg = baseline * alpha_home * beta_away * hfa_mult
            dc_away_xg = baseline * alpha_away * beta_home
        else:
            # Fallback: no match data yet
            dc_home_xg = elo_home_xg
            dc_away_xg = elo_away_xg

        # ── Step 4: Blend Dixon-Coles xG with Elo-derived xG ──
        w = XG_BLEND_WEIGHT  # 0 = pure Elo, 1 = pure Dixon-Coles
        mu_home = max(MIN_XG, w * dc_home_xg + (1 - w) * elo_home_xg)
        mu_away = max(MIN_XG, w * dc_away_xg + (1 - w) * elo_away_xg)

        # ── Step 5: Skellam PMF ──
        p_home_win = sum(
            skellam.pmf(k, mu_home, mu_away)
            for k in range(1, SKELLAM_TAIL_RANGE + 1)
        )
        p_draw = float(skellam.pmf(0, mu_home, mu_away))
        p_away_win = sum(
            skellam.pmf(k, mu_home, mu_away)
            for k in range(-SKELLAM_TAIL_RANGE, 0)
        )

        # ── Step 6: Normalise ──
        total = p_home_win + p_draw + p_away_win
        return {
            "home_win": p_home_win / total,
            "draw": p_draw / total,
            "away_win": p_away_win / total,
            "xg_home": mu_home,
            "xg_away": mu_away,
            "expected_gd": expected_gd,
        }

    # ------------------------------------------------------------------ #
    # Elo history export                                                   #
    # ------------------------------------------------------------------ #

    def export_elo_history(self, path: str = "data/elo_history.csv"):
        """
        Export per-match Elo snapshots to CSV for plotting trajectories.
        Columns: match_number, team_1_elo, team_2_elo, ...
        """
        import csv as _csv

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not self.elo_history:
            print("[History] No matches processed — nothing to export.")
            return

        team_names = sorted(self.teams.keys())
        with open(out, "w", newline="") as f:
            writer = _csv.writer(f)
            writer.writerow(["match"] + team_names)
            for i, snapshot in enumerate(self.elo_history, start=1):
                row = [i] + [round(snapshot.get(name, BASE_ELO), 1) for name in team_names]
                writer.writerow(row)

        print(f"[History] Wrote {len(self.elo_history)} snapshots to {out}")

    # ------------------------------------------------------------------ #
    # Export / Import ratings for next-season priors                        #
    # ------------------------------------------------------------------ #

    def export_ratings(self, path: str = "data/end_of_season_elos.json"):
        """
        Export current Elo ratings to JSON for use as next-season priors.
        Ratings are regressed toward BASE_ELO by PRIOR_REGRESSION_FACTOR.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        regressed = {}
        for name, team in self.teams.items():
            regressed[name] = round(
                BASE_ELO + (team.elo - BASE_ELO) * (1 - PRIOR_REGRESSION_FACTOR), 1
            )

        # Normalize so pool average == BASE_ELO (Elo conservation)
        if len(regressed) > 1:
            avg_elo = statistics.mean(regressed.values())
            offset = avg_elo - BASE_ELO
            if abs(offset) > 0.05:
                regressed = {k: round(v - offset, 1) for k, v in regressed.items()}
                print(f"[Export] Normalization: shifted all ratings by {-offset:+.1f}")

        out.write_text(json.dumps(regressed, indent=2))
        print(f"[Export] Wrote {len(regressed)} regressed ratings to {out}")
        return regressed

    @staticmethod
    def load_priors_from_file(path: str = "data/end_of_season_elos.json") -> dict[str, float]:
        """Load priors from a previously exported JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No prior ratings file at {p}")
        return json.loads(p.read_text())

    # ------------------------------------------------------------------ #
    # Sorted standings                                                     #
    # ------------------------------------------------------------------ #

    def standings(self) -> list[Team]:
        """Return teams sorted by Elo (descending)."""
        return sorted(self.teams.values(), key=lambda t: t.elo, reverse=True)
