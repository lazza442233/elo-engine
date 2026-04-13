"""
SQLite persistence layer for the Grassroots Elo Engine.

Stores processed match results and per-match Elo snapshots so the engine
can skip re-fetching the full API on subsequent runs.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/elo_engine.db")


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """Create tables if they don't already exist."""
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            league_key  TEXT    NOT NULL,
            match_date  TEXT    NOT NULL,
            home_team   TEXT    NOT NULL,
            away_team   TEXT    NOT NULL,
            home_score  INTEGER NOT NULL,
            away_score  INTEGER NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'played',
            api_hash    TEXT    NOT NULL,
            UNIQUE(league_key, api_hash)
        );

        CREATE TABLE IF NOT EXISTS elo_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            league_key  TEXT    NOT NULL,
            match_id    INTEGER NOT NULL REFERENCES matches(id),
            snapshot    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS team_ratings (
            league_key  TEXT NOT NULL,
            team_name   TEXT NOT NULL,
            elo         REAL NOT NULL,
            played      INTEGER NOT NULL DEFAULT 0,
            wins        INTEGER NOT NULL DEFAULT 0,
            draws       INTEGER NOT NULL DEFAULT 0,
            losses      INTEGER NOT NULL DEFAULT 0,
            gf          INTEGER NOT NULL DEFAULT 0,
            ga          INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (league_key, team_name)
        );
    """)
    conn.commit()
    conn.close()


def _match_hash(date: str, home: str, away: str) -> str:
    """Deterministic hash for deduplication (no crypto needed — just identity)."""
    import hashlib
    raw = f"{date}|{home}|{away}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_matches(league_key: str, matches: list[dict], db_path: Path = DB_PATH) -> int:
    """
    Persist processed matches, skipping duplicates.
    Each match dict must have: date, home_team, away_team, home_score, away_score.
    Returns count of newly inserted rows.
    """
    conn = _connect(db_path)
    inserted = 0
    for m in matches:
        h = _match_hash(m["date"], m["home_team"], m["away_team"])
        try:
            conn.execute(
                """INSERT INTO matches (league_key, match_date, home_team, away_team,
                   home_score, away_score, status, api_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (league_key, m["date"], m["home_team"], m["away_team"],
                 m["home_score"], m["away_score"], m.get("status", "played"), h),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate — skip
    conn.commit()
    conn.close()
    return inserted


def load_matches(league_key: str, db_path: Path = DB_PATH) -> list[dict]:
    """Load all stored matches for a league, ordered by date."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM matches WHERE league_key = ? ORDER BY match_date",
        (league_key,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_elo_snapshot(league_key: str, match_id: int, snapshot: dict[str, float],
                      db_path: Path = DB_PATH):
    """Save a single Elo snapshot (team→rating dict) linked to a match."""
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO elo_snapshots (league_key, match_id, snapshot) VALUES (?, ?, ?)",
        (league_key, match_id, json.dumps(snapshot)),
    )
    conn.commit()
    conn.close()


def save_team_ratings(league_key: str, teams: dict, db_path: Path = DB_PATH):
    """Upsert current team ratings (for quick resume without reprocessing)."""
    conn = _connect(db_path)
    for name, team in teams.items():
        conn.execute(
            """INSERT INTO team_ratings (league_key, team_name, elo, played, wins, draws, losses, gf, ga)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(league_key, team_name) DO UPDATE SET
                   elo=excluded.elo, played=excluded.played, wins=excluded.wins,
                   draws=excluded.draws, losses=excluded.losses, gf=excluded.gf, ga=excluded.ga""",
            (league_key, name, team.elo, team.played, team.wins, team.draws,
             team.losses, team.gf, team.ga),
        )
    conn.commit()
    conn.close()


def load_team_ratings(league_key: str, db_path: Path = DB_PATH) -> list[dict]:
    """Load saved team ratings for a league."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM team_ratings WHERE league_key = ? ORDER BY elo DESC",
        (league_key,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_match_count(league_key: str, db_path: Path = DB_PATH) -> int:
    """Return count of stored matches for a league."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM matches WHERE league_key = ?",
        (league_key,),
    ).fetchone()
    conn.close()
    return row["cnt"]
