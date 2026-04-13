"""
Prediction calibration: log predictions vs actual results, compute Brier score.
"""

import csv
from pathlib import Path


CALIBRATION_LOG = Path("data/calibration_log.csv")
_FIELDNAMES = [
    "home", "away", "p_home_win", "p_draw", "p_away_win",
    "xg_home", "xg_away", "actual_home_score", "actual_away_score", "actual_outcome",
]


def log_prediction(prediction: dict, home: str, away: str,
                   home_score: int, away_score: int):
    """Append a single prediction + actual result to the calibration CSV."""
    CALIBRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CALIBRATION_LOG.exists()

    if home_score > away_score:
        outcome = "home_win"
    elif home_score < away_score:
        outcome = "away_win"
    else:
        outcome = "draw"

    row = {
        "home": home,
        "away": away,
        "p_home_win": f"{prediction['home_win']:.4f}",
        "p_draw": f"{prediction['draw']:.4f}",
        "p_away_win": f"{prediction['away_win']:.4f}",
        "xg_home": f"{prediction['xg_home']:.2f}",
        "xg_away": f"{prediction['xg_away']:.2f}",
        "actual_home_score": home_score,
        "actual_away_score": away_score,
        "actual_outcome": outcome,
    }

    with open(CALIBRATION_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_brier_score() -> dict | None:
    """
    Compute Brier score from the calibration log.

    Brier score = mean of sum( (p_i - o_i)^2 ) for each outcome category.
    Lower is better. 0 = perfect, 1 = worst possible.
    Random guessing (1/3 each) gives ~0.667.
    """
    if not CALIBRATION_LOG.exists():
        return None

    total_bs = 0.0
    n = 0

    with open(CALIBRATION_LOG, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_h = float(row["p_home_win"])
            p_d = float(row["p_draw"])
            p_a = float(row["p_away_win"])
            outcome = row["actual_outcome"]

            o_h = 1.0 if outcome == "home_win" else 0.0
            o_d = 1.0 if outcome == "draw" else 0.0
            o_a = 1.0 if outcome == "away_win" else 0.0

            total_bs += (p_h - o_h) ** 2 + (p_d - o_d) ** 2 + (p_a - o_a) ** 2
            n += 1

    if n == 0:
        return None

    return {"brier_score": total_bs / n, "n_predictions": n}
