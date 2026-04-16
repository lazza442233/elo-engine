"""
Skellam-Elo Hybrid engine for grassroots football.

Design choices:
  - Adaptive K      : K=40 early (fast convergence), K=30 after 10 games
  - HFA = 50 pts    : empirically optimized via D1 grid search
  - Log MoV (dampened): diminishing returns via ln(GD+1), with
                        asymmetric dampening MoV/(|Δelo|·C1+C2) so
                        expected blowouts don't over-inflate ratings
  - Opponent-adjusted xG: team attack/defence rates adjusted for
                          opponent quality (Massey-style)
  - Skellam pmf     : better than Probit for high-scoring amateur football
"""

import json
import math
from datetime import datetime
from pathlib import Path

from scipy.stats import skellam

from config.constants import (
    BASE_ELO,
    ELO_TO_GOAL_RATIO,
    HOME_FIELD_ADVANTAGE,
    K_FACTOR_INITIAL,
    K_FACTOR_SETTLED,
    K_TRANSITION_GAMES,
    LEAGUE_AVG_GOALS,
    MIN_XG,
    MOV_C1,
    MOV_C2,
    PRIOR_REGRESSION_FACTOR,
    SKELLAM_TAIL_RANGE,
    XG_ASYMMETRY_FACTOR,
)
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
        self.elo_history: list[dict[str, float]] = []  # snapshot after each match
        self.match_log: list[dict] = []  # per-match metadata
        self.initial_elos: dict[str, float] = {}  # starting Elos after priors

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

    def _get_or_create(self, name: str) -> Team:
        if name not in self.teams:
            self.teams[name] = Team(name)
        return self.teams[name]

    @staticmethod
    def _shorten_name(raw: str) -> str:
        """
        Strip the league/grade suffix that Dribl appends to every team name.
        e.g. 'Kellyville Kolts Soccer Club Premier League Men Reserve Grade'
             -> 'Kellyville Kolts Soccer Club'
        """
        suffixes = [
            " Premier League Men First Grade",
            " Premier League Men Reserve Grade",
            " Premier League First Grade",
            " Premier League Reserve Grade",
            " Premier League Men",
            " Premier League",
            " First Grade",
            " Reserve Grade",
        ]
        for sfx in suffixes:
            if raw.endswith(sfx):
                return raw[: -len(sfx)].strip()
        return raw.strip()

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
    def mov_multiplier(goal_diff: int) -> float:
        """
        Uncapped logarithmic MoV multiplier.
          - goal_diff is the raw score difference.
          - No hard cap: massive grassroots blowouts (e.g. 15-0) are
            allowed to influence Elo via diminishing log returns.
        """
        effective_gd = abs(goal_diff)
        if effective_gd <= 1:
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
        if not quiet:
            print(f"[Priors] Injected {len(priors)} team rating(s).")

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
        e_away = 1.0 - e_home

        # Actual score (1 = win, 0.5 = draw, 0 = loss)
        if home_score > away_score:
            s_home, s_away = 1.0, 0.0
        elif home_score < away_score:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # MoV multiplier with autocorrelation dampening
        # Dampens MoV bonus when the Elo gap already predicted the blowout
        raw_mult = self.mov_multiplier(home_score - away_score)
        elo_diff = abs(elo_home_adj - away.elo)
        mult = raw_mult / (elo_diff * MOV_C1 + MOV_C2)

        # Adaptive K — average of both teams' K-factors
        k = (self.k_factor(home.played) + self.k_factor(away.played)) / 2.0

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

        if home_score > away_score:
            home.wins += 1
            away.losses += 1
        elif home_score < away_score:
            away.wins += 1
            home.losses += 1
        else:
            home.draws += 1
            away.draws += 1

        # Opponent-quality tracking for adjusted rates
        half = self.league_avg_goals / 2.0
        home.opp_def_sum += away.defence_rate if away.played > 1 else half
        home.opp_att_sum += away.attack_rate if away.played > 1 else half
        away.opp_def_sum += home.defence_rate if home.played > 1 else half
        away.opp_att_sum += home.attack_rate if home.played > 1 else half

        self.processed_matches += 1
        self._total_goals += home_score + away_score

        # Propagate adaptive league average to Team class
        Team.league_avg_goals = self.league_avg_goals

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

    # ------------------------------------------------------------------ #
    # Process full match list from API data                                #
    # ------------------------------------------------------------------ #

    def process_matches(self, raw_data: list[dict], quiet: bool = False):
        """
        Ingest the Dribl API 'data' list chronologically.
        Skips: Forfeit, Abandoned, Postponed, Upcoming, Bye.
        """
        SKIP_STATUSES = {"forfeit", "abandoned", "postponed", "upcoming", "bye"}

        # Sort by date ascending so early rounds update ratings first
        def parse_date(m):
            try:
                return datetime.strptime(
                    m["attributes"]["date"], "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                return datetime.min

        matches = sorted(raw_data, key=parse_date)

        skipped = 0
        for match in matches:
            attrs = match["attributes"]

            # Skip byes
            if attrs.get("bye_flag", 0):
                skipped += 1
                continue

            # Skip non-played statuses
            status = attrs.get("status", "").lower()
            if status in SKIP_STATUSES:
                skipped += 1
                continue

            # Extract and validate scores
            home_score = attrs.get("home_score")
            away_score = attrs.get("away_score")
            if home_score is None or away_score is None:
                skipped += 1
                continue

            home_name = self._shorten_name(attrs["home_team_name"])
            away_name = self._shorten_name(attrs["away_team_name"])

            round_label = attrs.get("full_round")
            match_date = attrs.get("date")
            self.process_match(home_name, away_name, int(home_score), int(away_score),
                               round_label=round_label, match_date=match_date)

        if skipped and not quiet:
            print(f"[Ingest] Skipped {skipped} match(es) (bye/forfeit/invalid).")

    # ------------------------------------------------------------------ #
    # Skellam prediction                                                   #
    # ------------------------------------------------------------------ #

    def predict_match(
        self, home_name: str, away_name: str, neutral: bool = False
    ) -> dict:
        """
        Predict Win/Draw/Loss probabilities using the Skellam distribution.

        Uses variable xG derived from each team's attack/defence rates,
        blended with Elo-derived expected goal difference.

        Args:
            home_name:  short team name (no suffix)
            away_name:  short team name (no suffix)
            neutral:    set True for cup finals / neutral venues (disables HFA)

        Returns:
            dict with keys: home_win, draw, away_win (sum = 1.0),
            plus xg_home, xg_away, expected_gd.
        """
        home = self._get_or_create(home_name)
        away = self._get_or_create(away_name)

        hfa = 0 if neutral else HOME_FIELD_ADVANTAGE

        # Step 1: Elo-to-expected-goal-difference
        elo_diff = (home.elo + hfa) - away.elo
        expected_gd = elo_diff / ELO_TO_GOAL_RATIO

        # Step 2: Variable xG from opponent-adjusted attack/defence rates
        raw_home_xg = (home.adj_attack_rate + away.adj_defence_rate) / 2.0
        raw_away_xg = (away.adj_attack_rate + home.adj_defence_rate) / 2.0

        # Step 3: Blend raw xG toward Elo-derived xG
        league_avg_half = self.league_avg_goals / 2.0
        elo_home_xg = league_avg_half + (expected_gd * XG_ASYMMETRY_FACTOR)
        elo_away_xg = league_avg_half - (expected_gd * XG_ASYMMETRY_FACTOR)
        mu_home = max(MIN_XG, (raw_home_xg + elo_home_xg) / 2.0)
        mu_away = max(MIN_XG, (raw_away_xg + elo_away_xg) / 2.0)

        # Step 4: Skellam pmf with wide tails for grassroots blowouts
        p_home_win = sum(
            skellam.pmf(k, mu_home, mu_away)
            for k in range(1, SKELLAM_TAIL_RANGE + 1)
        )
        p_draw = float(skellam.pmf(0, mu_home, mu_away))
        p_away_win = sum(
            skellam.pmf(k, mu_home, mu_away)
            for k in range(-SKELLAM_TAIL_RANGE, 0)
        )

        # Step 5: Normalise
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
