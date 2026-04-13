"""
Constants and configuration for the Grassroots Elo Engine.
"""

# ---------------------------------------------------------------------------
# ELO PARAMETERS
# ---------------------------------------------------------------------------
BASE_ELO = 1500
K_FACTOR_INITIAL = 40       # K for teams with < K_TRANSITION_GAMES played
K_FACTOR_SETTLED = 30       # K for teams with >= K_TRANSITION_GAMES played (D1 optimized)
K_TRANSITION_GAMES = 10     # games threshold for K decay
HOME_FIELD_ADVANTAGE = 50   # pts — empirically optimized via D1 grid search
MOV_C1 = 0.001              # asymmetric MoV dampener: MoV / (|elo_diff| * C1 + C2)
MOV_C2 = 3.0                # baseline dampening denominator
LEAGUE_AVG_GOALS = 7.0      # total goals per game (derived from match data)

# ---------------------------------------------------------------------------
# SKELLAM PREDICTION PARAMETERS
# ---------------------------------------------------------------------------
ELO_TO_GOAL_RATIO = 75      # Elo pts per expected goal difference (D1 optimized)
XG_ASYMMETRY_FACTOR = 0.75  # how aggressively xG splits between home/away
MIN_XG = 0.2                # floor for expected goals (avoid Skellam edge cases)
SKELLAM_TAIL_RANGE = 20     # max goal diff evaluated in Skellam PMF

# ---------------------------------------------------------------------------
# PRIOR REGRESSION
# ---------------------------------------------------------------------------
PRIOR_REGRESSION_FACTOR = 0.2  # retain 80% of Elo delta across seasons (D1 optimized)

# ---------------------------------------------------------------------------
# LEAGUE CONFIGURATIONS
# ---------------------------------------------------------------------------
LEAGUES = {
    "prem-men": {
        "name": "Premier League First Grade",
        "season": "7MNGzV2mAz",
        "competition": "LBdDxbrxdb",
        "league": "6lNb5Zlgmx",
        "tenant": "PLBdD94mb7",
        "priors": {},
    },
    "prem-res": {
        "name": "Premier League Reserve Grade",
        "season": "7MNGzV2mAz",
        "competition": "LBdDxbrxdb",
        "league": "bgdMn55OmE",
        "tenant": "PLBdD94mb7",
        "priors": {},
    },
}

DEFAULT_LEAGUE = "prem-men"

# ---------------------------------------------------------------------------
# BACKWARD-COMPAT ALIASES (derived from default league)
# ---------------------------------------------------------------------------
PRIOR_RATINGS = LEAGUES[DEFAULT_LEAGUE]["priors"]

DEFAULT_ROUND = 5


def _build_api_url(league_key: str = DEFAULT_LEAGUE) -> str:
    cfg = LEAGUES[league_key]
    return (
        "https://mc-api.dribl.com/api/results"
        "?date_range=default"
        f"&season={cfg['season']}"
        f"&competition={cfg['competition']}"
        f"&league={cfg['league']}"
        f"&tenant={cfg['tenant']}"
        "&results=1"
        "&timezone=Australia%2FSydney"
    )


def fixtures_url(round_number: int = DEFAULT_ROUND, league_key: str = DEFAULT_LEAGUE) -> str:
    cfg = LEAGUES[league_key]
    return (
        "https://mc-api.dribl.com/api/fixtures"
        f"?season={cfg['season']}"
        "&date_range=default"
        f"&competition={cfg['competition']}"
        f"&league={cfg['league']}"
        f"&type_round=roundrobin_{round_number}"
        f"&tenant={cfg['tenant']}"
        "&timezone=Australia%2FSydney"
    )


API_URL = _build_api_url()
