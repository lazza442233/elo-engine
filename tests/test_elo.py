"""
Unit tests for the Grassroots Elo Engine.

Run with:  python -m pytest tests/ -v
"""

import csv
import json
import math
import sqlite3
from pathlib import Path

import pytest

from config.constants import (
    BASE_ELO,
    DEFAULT_LEAGUE,
    LEAGUES,
    K_FACTOR_INITIAL,
    K_FACTOR_SETTLED,
    K_TRANSITION_GAMES,
    PRIOR_REGRESSION_FACTOR,
)
from engine.calibration import compute_brier_score, log_prediction
from engine.elo import GrassrootsEloEngine
from models.team import Team


@pytest.fixture
def engine():
    return GrassrootsEloEngine()


# ------------------------------------------------------------------ #
# _shorten_name                                                        #
# ------------------------------------------------------------------ #

class TestShortenName:
    def test_strips_premier_league_men_reserve_grade(self):
        raw = "Kellyville Kolts Soccer Club Premier League Men Reserve Grade"
        assert GrassrootsEloEngine._shorten_name(raw) == "Kellyville Kolts Soccer Club"

    def test_strips_premier_league_reserve_grade(self):
        raw = "Epping FC Premier League Reserve Grade"
        assert GrassrootsEloEngine._shorten_name(raw) == "Epping FC"

    def test_strips_premier_league_men(self):
        raw = "Pennant Hills FC Premier League Men"
        assert GrassrootsEloEngine._shorten_name(raw) == "Pennant Hills FC"

    def test_strips_premier_league(self):
        raw = "West Ryde Rovers FC Premier League"
        assert GrassrootsEloEngine._shorten_name(raw) == "West Ryde Rovers FC"

    def test_strips_reserve_grade(self):
        raw = "Gladesville Ravens SC Reserve Grade"
        assert GrassrootsEloEngine._shorten_name(raw) == "Gladesville Ravens SC"

    def test_no_suffix_unchanged(self):
        raw = "Putney Rangers FC"
        assert GrassrootsEloEngine._shorten_name(raw) == "Putney Rangers FC"

    def test_strips_whitespace(self):
        assert GrassrootsEloEngine._shorten_name("  Some FC  ") == "Some FC"

    def test_longest_suffix_matched_first(self):
        raw = "Test FC Premier League Men Reserve Grade"
        assert GrassrootsEloEngine._shorten_name(raw) == "Test FC"


# ------------------------------------------------------------------ #
# expected_score                                                       #
# ------------------------------------------------------------------ #

class TestExpectedScore:
    def test_equal_ratings_returns_0_5(self):
        assert GrassrootsEloEngine.expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_favoured(self):
        score = GrassrootsEloEngine.expected_score(1600, 1400)
        assert score > 0.5

    def test_lower_rating_underdog(self):
        score = GrassrootsEloEngine.expected_score(1400, 1600)
        assert score < 0.5

    def test_symmetry(self):
        a = GrassrootsEloEngine.expected_score(1600, 1400)
        b = GrassrootsEloEngine.expected_score(1400, 1600)
        assert a + b == pytest.approx(1.0)

    def test_400_point_gap(self):
        score = GrassrootsEloEngine.expected_score(1900, 1500)
        assert score == pytest.approx(10.0 / 11.0, abs=1e-6)


# ------------------------------------------------------------------ #
# mov_multiplier                                                       #
# ------------------------------------------------------------------ #

class TestMovMultiplier:
    def test_draw_returns_1(self):
        assert GrassrootsEloEngine.mov_multiplier(0) == 1.0

    def test_one_goal_returns_1(self):
        assert GrassrootsEloEngine.mov_multiplier(1) == 1.0
        assert GrassrootsEloEngine.mov_multiplier(-1) == 1.0

    def test_two_goals(self):
        assert GrassrootsEloEngine.mov_multiplier(2) == pytest.approx(math.log(3))

    def test_large_blowout(self):
        result = GrassrootsEloEngine.mov_multiplier(15)
        assert result == pytest.approx(math.log(16))
        assert result > 1.0

    def test_negative_diff_same_as_positive(self):
        assert GrassrootsEloEngine.mov_multiplier(-5) == GrassrootsEloEngine.mov_multiplier(5)

    def test_diminishing_returns(self):
        # Each additional goal should contribute less than the previous one
        m2 = GrassrootsEloEngine.mov_multiplier(2)
        m3 = GrassrootsEloEngine.mov_multiplier(3)
        m4 = GrassrootsEloEngine.mov_multiplier(4)
        m5 = GrassrootsEloEngine.mov_multiplier(5)
        assert m3 - m2 > m4 - m3 > m5 - m4


# ------------------------------------------------------------------ #
# Adaptive K-factor                                                    #
# ------------------------------------------------------------------ #

class TestAdaptiveK:
    def test_zero_games_returns_initial(self):
        assert GrassrootsEloEngine.k_factor(0) == K_FACTOR_INITIAL

    def test_at_transition_returns_settled(self):
        assert GrassrootsEloEngine.k_factor(K_TRANSITION_GAMES) == K_FACTOR_SETTLED

    def test_above_transition_returns_settled(self):
        assert GrassrootsEloEngine.k_factor(20) == K_FACTOR_SETTLED

    def test_midpoint_interpolation(self):
        mid = K_TRANSITION_GAMES // 2
        k = GrassrootsEloEngine.k_factor(mid)
        assert K_FACTOR_SETTLED < k < K_FACTOR_INITIAL

    def test_monotonically_decreasing(self):
        k_values = [GrassrootsEloEngine.k_factor(g) for g in range(K_TRANSITION_GAMES + 1)]
        for i in range(len(k_values) - 1):
            assert k_values[i] >= k_values[i + 1]


# ------------------------------------------------------------------ #
# Elo conservation                                                     #
# ------------------------------------------------------------------ #

class TestEloConservation:
    def test_single_match_conserves_elo(self, engine):
        engine._get_or_create("Team A")
        engine._get_or_create("Team B")
        total_before = sum(t.elo for t in engine.teams.values())

        engine.process_match("Team A", "Team B", 3, 1)

        total_after = sum(t.elo for t in engine.teams.values())
        assert total_after == pytest.approx(total_before)

    def test_multiple_matches_conserve_elo(self, engine):
        teams = ["Alpha", "Bravo", "Charlie", "Delta"]
        for name in teams:
            engine._get_or_create(name)
        total_before = sum(t.elo for t in engine.teams.values())

        engine.process_match("Alpha", "Bravo", 2, 0)
        engine.process_match("Charlie", "Delta", 1, 1)
        engine.process_match("Alpha", "Charlie", 0, 3)
        engine.process_match("Bravo", "Delta", 4, 2)

        total_after = sum(t.elo for t in engine.teams.values())
        assert total_after == pytest.approx(total_before)

    def test_conservation_with_priors(self, engine):
        engine.inject_priors({"X": 1600, "Y": 1400})
        total_before = sum(t.elo for t in engine.teams.values())

        engine.process_match("X", "Y", 1, 0)
        engine.process_match("Y", "X", 2, 2)

        total_after = sum(t.elo for t in engine.teams.values())
        assert total_after == pytest.approx(total_before)


# ------------------------------------------------------------------ #
# Team.league_avg_goals isolation                                      #
# ------------------------------------------------------------------ #

class TestTeamLeagueAvgGoals:
    def test_new_team_has_league_avg_goals(self):
        t = Team("Test")
        assert hasattr(t, "league_avg_goals")
        assert t.league_avg_goals > 0

    def test_deserialized_team_falls_back_to_class_default(self):
        """Simulate a cached/deserialized Team missing the instance attr."""
        t = Team("Cached")
        del t.__dict__["league_avg_goals"]  # remove instance attr
        # Should fall back to class-level default without AttributeError
        assert t.league_avg_goals > 0

    def test_engine_instances_dont_share_league_avg(self):
        """Two engines must not bleed league_avg_goals across Team objects."""
        e1 = GrassrootsEloEngine()
        e2 = GrassrootsEloEngine()
        e1.process_match("A", "B", 5, 0)
        e2._get_or_create("C")
        # Team C (engine 2) should still have the default, not engine 1's adaptive value
        from config.constants import LEAGUE_AVG_GOALS
        assert e2.teams["C"].league_avg_goals == LEAGUE_AVG_GOALS

    def test_adj_rates_work_on_fresh_team(self, engine):
        """predict_match on never-seen teams must not raise AttributeError."""
        pred = engine.predict_match("NewHome", "NewAway")
        assert pred["home_win"] + pred["draw"] + pred["away_win"] == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------ #
# predict_match                                                        #
# ------------------------------------------------------------------ #

class TestPredictMatch:
    def test_probabilities_sum_to_one(self, engine):
        engine._get_or_create("Home")
        engine._get_or_create("Away")
        result = engine.predict_match("Home", "Away")
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert total == pytest.approx(1.0)

    def test_equal_teams_roughly_symmetric(self, engine):
        engine._get_or_create("A")
        engine._get_or_create("B")
        result = engine.predict_match("A", "B")
        # With HFA, home should be slightly favoured
        assert result["home_win"] > result["away_win"]

    def test_neutral_venue_symmetric(self, engine):
        engine._get_or_create("A")
        engine._get_or_create("B")
        result = engine.predict_match("A", "B", neutral=True)
        assert result["home_win"] == pytest.approx(result["away_win"], abs=0.01)
        assert result["expected_gd"] == pytest.approx(0.0)

    def test_stronger_team_favoured(self, engine):
        engine.inject_priors({"Strong": 1700, "Weak": 1300})
        result = engine.predict_match("Strong", "Weak")
        assert result["home_win"] > 0.7
        assert result["xg_home"] > result["xg_away"]
        assert result["expected_gd"] > 0

    def test_xg_floors_at_min(self, engine):
        engine.inject_priors({"Dom": 2000, "Sub": 1000})
        result = engine.predict_match("Dom", "Sub")
        assert result["xg_away"] >= 0.2

    def test_creates_unknown_teams(self, engine):
        result = engine.predict_match("NewTeamA", "NewTeamB", neutral=True)
        assert "NewTeamA" in engine.teams
        assert "NewTeamB" in engine.teams
        assert result["home_win"] == pytest.approx(result["away_win"], abs=0.01)

    def test_variable_xg_differs_from_flat(self, engine):
        # After games, attack/defence rates should influence xG
        engine.process_match("Attacker", "Defender", 8, 0)
        engine.process_match("Attacker", "Defender", 6, 1)
        result = engine.predict_match("Attacker", "Defender")
        # Attacker's high attack rate should push their xG above league avg/2
        assert result["xg_home"] > 3.5


# ------------------------------------------------------------------ #
# Export / Import ratings                                               #
# ------------------------------------------------------------------ #

class TestExportImport:
    def test_export_creates_file(self, engine, tmp_path):
        engine.inject_priors({"A": 1600, "B": 1400})
        path = str(tmp_path / "ratings.json")
        result = engine.export_ratings(path)
        assert Path(path).exists()
        assert "A" in result
        assert "B" in result

    def test_export_regresses_toward_base(self, engine, tmp_path):
        engine.inject_priors({"A": 1600})
        path = str(tmp_path / "ratings.json")
        result = engine.export_ratings(path)
        expected = BASE_ELO + (1600 - BASE_ELO) * (1 - PRIOR_REGRESSION_FACTOR)
        assert result["A"] == pytest.approx(expected, abs=0.1)

    def test_roundtrip(self, engine, tmp_path):
        engine.inject_priors({"A": 1600, "B": 1400})
        path = str(tmp_path / "ratings.json")
        engine.export_ratings(path)
        loaded = GrassrootsEloEngine.load_priors_from_file(path)
        assert "A" in loaded
        assert "B" in loaded

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            GrassrootsEloEngine.load_priors_from_file("/nonexistent/path.json")


# ------------------------------------------------------------------ #
# Calibration                                                          #
# ------------------------------------------------------------------ #

class TestCalibration:
    def test_log_and_compute(self, tmp_path, monkeypatch):
        log_path = tmp_path / "cal.csv"
        monkeypatch.setattr("engine.calibration.CALIBRATION_LOG", log_path)

        prediction = {
            "home_win": 0.6, "draw": 0.2, "away_win": 0.2,
            "xg_home": 3.5, "xg_away": 2.5,
        }
        log_prediction(prediction, "A", "B", 2, 1)

        result = compute_brier_score()
        assert result is not None
        assert result["n_predictions"] == 1
        # Brier = (0.6-1)^2 + (0.2-0)^2 + (0.2-0)^2 = 0.16+0.04+0.04 = 0.24
        assert result["brier_score"] == pytest.approx(0.24)

    def test_perfect_prediction_scores_zero(self, tmp_path, monkeypatch):
        log_path = tmp_path / "cal.csv"
        monkeypatch.setattr("engine.calibration.CALIBRATION_LOG", log_path)

        prediction = {
            "home_win": 1.0, "draw": 0.0, "away_win": 0.0,
            "xg_home": 5.0, "xg_away": 1.0,
        }
        log_prediction(prediction, "A", "B", 3, 0)

        result = compute_brier_score()
        assert result["brier_score"] == pytest.approx(0.0)

    def test_no_log_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("engine.calibration.CALIBRATION_LOG", tmp_path / "nope.csv")
        assert compute_brier_score() is None


# ------------------------------------------------------------------ #
# process_matches (integration)                                        #
# ------------------------------------------------------------------ #

class TestProcessMatches:
    def _make_match(self, home, away, home_score, away_score, status="played"):
        return {
            "attributes": {
                "date": "2025-04-01 15:00:00",
                "home_team_name": home,
                "away_team_name": away,
                "home_score": home_score,
                "away_score": away_score,
                "status": status,
                "bye_flag": 0,
            }
        }

    def test_skips_forfeits(self, engine):
        data = [self._make_match("A", "B", 3, 0, status="forfeit")]
        engine.process_matches(data)
        assert engine.processed_matches == 0

    def test_skips_byes(self, engine):
        match = self._make_match("A", "B", None, None)
        match["attributes"]["bye_flag"] = 1
        engine.process_matches([match])
        assert engine.processed_matches == 0

    def test_skips_null_scores(self, engine):
        data = [self._make_match("A", "B", None, None)]
        engine.process_matches(data)
        assert engine.processed_matches == 0

    def test_processes_valid_match(self, engine):
        data = [self._make_match("A", "B", 2, 1)]
        engine.process_matches(data)
        assert engine.processed_matches == 1
        assert engine.teams["A"].wins == 1
        assert engine.teams["B"].losses == 1


# ------------------------------------------------------------------ #
# Elo history tracking                                                 #
# ------------------------------------------------------------------ #

class TestEloHistory:
    def test_history_recorded_per_match(self, engine):
        engine.process_match("A", "B", 2, 0)
        engine.process_match("A", "B", 1, 1)
        assert len(engine.elo_history) == 2

    def test_history_contains_all_teams(self, engine):
        engine.process_match("A", "B", 3, 1)
        snapshot = engine.elo_history[0]
        assert "A" in snapshot
        assert "B" in snapshot

    def test_history_reflects_elo_changes(self, engine):
        engine.process_match("A", "B", 5, 0)
        snapshot = engine.elo_history[0]
        assert snapshot["A"] > BASE_ELO
        assert snapshot["B"] < BASE_ELO

    def test_export_history_creates_csv(self, engine, tmp_path):
        engine.process_match("X", "Y", 2, 1)
        engine.process_match("X", "Y", 0, 1)
        path = str(tmp_path / "history.csv")
        engine.export_elo_history(path)
        assert Path(path).exists()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 data rows

    def test_match_log_recorded(self, engine):
        engine.process_match("A", "B", 3, 1, round_label="Round 1")
        assert len(engine.match_log) == 1
        entry = engine.match_log[0]
        assert entry["home"] == "A"
        assert entry["away"] == "B"
        assert entry["round"] == "Round 1"
        assert entry["home_score"] == 3
        assert entry["away_score"] == 1

    def test_match_log_tracks_round_from_api(self, engine):
        data = [
            {"attributes": {"date": "2025-04-01 15:00:00",
                            "home_team_name": "X", "away_team_name": "Y",
                            "home_score": 2, "away_score": 0,
                            "status": "played", "bye_flag": 0,
                            "full_round": "Round 3"}},
        ]
        engine.process_matches(data)
        assert engine.match_log[0]["round"] == "Round 3"

    def test_match_log_none_round_when_missing(self, engine):
        engine.process_match("A", "B", 1, 0)
        assert engine.match_log[0]["round"] is None


# ------------------------------------------------------------------ #
# JSON output                                                          #
# ------------------------------------------------------------------ #

class TestJsonOutput:
    def test_output_json_structure(self, engine, capsys):
        from display.output import output_json
        engine.process_match("Home FC", "Away FC", 3, 1)
        output_json(engine, [])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "rankings" in data
        assert "predictions" in data
        assert len(data["rankings"]) == 2
        assert data["rankings"][0]["team"] == "Home FC"


# ------------------------------------------------------------------ #
# Multi-league config                                                  #
# ------------------------------------------------------------------ #

class TestMultiLeague:
    def test_default_league_exists(self):
        assert DEFAULT_LEAGUE in LEAGUES

    def test_league_has_required_keys(self):
        for key, cfg in LEAGUES.items():
            assert "name" in cfg, f"{key} missing 'name'"
            assert "season" in cfg, f"{key} missing 'season'"
            assert "competition" in cfg, f"{key} missing 'competition'"
            assert "league" in cfg, f"{key} missing 'league'"
            assert "tenant" in cfg, f"{key} missing 'tenant'"
            assert "priors" in cfg, f"{key} missing 'priors'"

    def test_build_api_url_uses_league_ids(self):
        from config.constants import _build_api_url
        url = _build_api_url("prem-men")
        cfg = LEAGUES["prem-men"]
        assert cfg["season"] in url
        assert cfg["competition"] in url
        assert cfg["league"] in url

    def test_fixtures_url_uses_league_ids(self):
        from config.constants import fixtures_url
        url = fixtures_url(5, "prem-men")
        cfg = LEAGUES["prem-men"]
        assert cfg["season"] in url
        assert "roundrobin_5" in url

    def test_priors_backward_compat(self):
        from config.constants import PRIOR_RATINGS
        assert PRIOR_RATINGS == LEAGUES[DEFAULT_LEAGUE]["priors"]


# ------------------------------------------------------------------ #
# SQLite persistence                                                   #
# ------------------------------------------------------------------ #

class TestPersistence:
    def test_init_creates_tables(self, tmp_path):
        from persistence.db import init_db
        db = tmp_path / "test.db"
        init_db(db)
        conn = sqlite3.connect(str(db))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "matches" in tables
        assert "elo_snapshots" in tables
        assert "team_ratings" in tables

    def test_save_and_load_matches(self, tmp_path):
        from persistence.db import init_db, save_matches, load_matches
        db = tmp_path / "test.db"
        init_db(db)
        matches = [
            {"date": "2025-04-01 15:00:00", "home_team": "A", "away_team": "B",
             "home_score": 3, "away_score": 1},
            {"date": "2025-04-08 15:00:00", "home_team": "C", "away_team": "D",
             "home_score": 0, "away_score": 0},
        ]
        inserted = save_matches("test-league", matches, db)
        assert inserted == 2

        loaded = load_matches("test-league", db)
        assert len(loaded) == 2
        assert loaded[0]["home_team"] == "A"

    def test_duplicate_matches_skipped(self, tmp_path):
        from persistence.db import init_db, save_matches, get_match_count
        db = tmp_path / "test.db"
        init_db(db)
        matches = [
            {"date": "2025-04-01 15:00:00", "home_team": "A", "away_team": "B",
             "home_score": 3, "away_score": 1},
        ]
        save_matches("test-league", matches, db)
        save_matches("test-league", matches, db)  # duplicate
        assert get_match_count("test-league", db) == 1

    def test_save_and_load_team_ratings(self, tmp_path, engine):
        from persistence.db import init_db, save_team_ratings, load_team_ratings
        db = tmp_path / "test.db"
        init_db(db)
        engine.process_match("Alpha", "Bravo", 2, 0)
        save_team_ratings("test-league", engine.teams, db)

        loaded = load_team_ratings("test-league", db)
        assert len(loaded) == 2
        names = {r["team_name"] for r in loaded}
        assert "Alpha" in names
        assert "Bravo" in names

    def test_leagues_isolated(self, tmp_path):
        from persistence.db import init_db, save_matches, get_match_count
        db = tmp_path / "test.db"
        init_db(db)
        matches = [
            {"date": "2025-04-01 15:00:00", "home_team": "A", "away_team": "B",
             "home_score": 1, "away_score": 0},
        ]
        save_matches("league-1", matches, db)
        save_matches("league-2", matches, db)
        assert get_match_count("league-1", db) == 1
        assert get_match_count("league-2", db) == 1
