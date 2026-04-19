"""
Constants and configuration for the Grassroots Elo Engine.
"""

# ---------------------------------------------------------------------------
# ELO PARAMETERS
# ---------------------------------------------------------------------------
BASE_ELO = 1500
K_FACTOR_INITIAL = 30       # K for teams with < K_TRANSITION_GAMES played
K_FACTOR_SETTLED = 35       # K for teams with >= K_TRANSITION_GAMES played
K_TRANSITION_GAMES = 10     # games threshold for K decay
HOME_FIELD_ADVANTAGE = 30   # pts — v2 optimised (was 50)
MOV_C1 = 0.001              # asymmetric MoV dampener: MoV / (|elo_diff| * C1 + C2)
MOV_C2 = 2.0                # baseline dampening denominator
LEAGUE_AVG_GOALS = 7.0      # total goals per game (derived from match data)

# ---------------------------------------------------------------------------
# SKELLAM PREDICTION PARAMETERS
# ---------------------------------------------------------------------------
ELO_TO_GOAL_RATIO = 75      # Elo pts per expected goal difference
XG_ASYMMETRY_FACTOR = 0.85  # how aggressively xG splits between home/away (v2 optimised, was 0.75)
XG_BLEND_WEIGHT = 1.0       # blend ratio: 0 = pure Elo xG, 1 = pure opponent-adjusted xG
MIN_XG = 0.2                # floor for expected goals (avoid Skellam edge cases)
SKELLAM_TAIL_RANGE = 20     # max goal diff evaluated in Skellam PMF

# ---------------------------------------------------------------------------
# DIXON-COLES / ROBUST PREDICTION PARAMETERS (v3)
# ---------------------------------------------------------------------------
WINSORIZE_GOALS_CAP = 5     # cap per-team goals at this value for rate/xG calc
                             # (a 24-0 proves the same dominance as 5-0)
SHRINKAGE_GAMES_FULL_TRUST = 8   # games before observed rates fully replace priors
                                  # (amateur seasons are short; quality shows by R5-6)
RIDGE_LAMBDA = 0.01         # L2 regularisation — light touch, shrinkage does the heavy lifting
HFA_MULTIPLIER = 1.05       # home-field advantage as scoring multiplier
                             # (amateur leagues: small crowds, short travel → modest HFA)

# ---------------------------------------------------------------------------
# PRIOR REGRESSION
# ---------------------------------------------------------------------------
PRIOR_REGRESSION_FACTOR = 0.4  # retain 60% of Elo delta across seasons (v2 optimised, was 0.2)

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
