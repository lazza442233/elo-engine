"""
Team model for the Grassroots Elo Engine.
"""

from config.constants import BASE_ELO, LEAGUE_AVG_GOALS


class Team:
    # Class-level fallback for deserialized/cached objects missing the instance attr
    league_avg_goals: float = LEAGUE_AVG_GOALS

    def __init__(self, name: str, elo: float = BASE_ELO):
        self.name = name
        self.elo = elo
        self.league_avg_goals: float = LEAGUE_AVG_GOALS
        self.played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.gf = 0   # goals for
        self.ga = 0   # goals against
        self.opp_def_sum = 0.0  # sum of opponents' defence rates (for adj attack)
        self.opp_att_sum = 0.0  # sum of opponents' attack rates (for adj defence)

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def attack_rate(self) -> float:
        """Goals scored per game (or league average if no games played)."""
        if self.played == 0:
            return self.league_avg_goals / 2.0
        return self.gf / self.played

    @property
    def defence_rate(self) -> float:
        """Goals conceded per game (or league average if no games played)."""
        if self.played == 0:
            return self.league_avg_goals / 2.0
        return self.ga / self.played

    @property
    def adj_attack_rate(self) -> float:
        """Opponent-adjusted attack: goals scored normalised by opponent defence quality."""
        half = self.league_avg_goals / 2.0
        if self.played == 0:
            return half
        avg_opp_def = self.opp_def_sum / self.played
        if avg_opp_def <= 0:
            return self.attack_rate
        return self.attack_rate * (half / avg_opp_def)

    @property
    def adj_defence_rate(self) -> float:
        """Opponent-adjusted defence: goals conceded normalised by opponent attack quality."""
        half = self.league_avg_goals / 2.0
        if self.played == 0:
            return half
        avg_opp_att = self.opp_att_sum / self.played
        if avg_opp_att <= 0:
            return self.defence_rate
        return self.defence_rate * (half / avg_opp_att)

    def __repr__(self):
        return f"Team({self.name!r}, elo={self.elo:.1f}, played={self.played})"
