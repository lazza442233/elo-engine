"""
SQLite persistence layer for the Grassroots Elo Engine.

Stores processed match results and replay metadata so the engine can skip
re-fetching the full API on subsequent runs without degrading offline replay.
"""

import sqlite3
from pathlib import Path

from engine.match_record import MatchRecord, normalize_match_records

DB_PATH = Path("data/elo_engine.db")

_MATCH_TABLE_COLUMNS = {
    "round_label": "TEXT",
}


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]):
    """Apply additive schema migrations for existing SQLite files."""
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_type in columns.items():
        if column_name in existing:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


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
            round_label TEXT,
            api_hash    TEXT    NOT NULL,
            UNIQUE(league_key, api_hash)
        );
    """)
    _ensure_columns(conn, "matches", _MATCH_TABLE_COLUMNS)
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
    Each match dict must have match_date/date, home_team, away_team, home_score,
    away_score, and may optionally include round_label.
    Returns count of newly inserted rows.
    """
    conn = _connect(db_path)
    inserted = 0
    for m in matches:
        match_date = m.get("match_date", m.get("date"))
        if match_date is None:
            raise KeyError("Each match must include 'match_date' or 'date'.")

        h = _match_hash(match_date, m["home_team"], m["away_team"])
        try:
            conn.execute(
                """INSERT INTO matches (league_key, match_date, home_team, away_team,
                   home_score, away_score, status, round_label, api_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    league_key,
                    match_date,
                    m["home_team"],
                    m["away_team"],
                    m["home_score"],
                    m["away_score"],
                    m.get("status", "played"),
                    m.get("round_label"),
                    h,
                ),
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
        "SELECT * FROM matches WHERE league_key = ? ORDER BY match_date, home_team, away_team",
        (league_key,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_match_records(league_key: str, db_path: Path = DB_PATH) -> list[MatchRecord]:
    """Load stored matches as normalized replay records."""
    rows = load_matches(league_key, db_path)
    records, _ = normalize_match_records(rows)
    return records


def get_match_count(league_key: str, db_path: Path = DB_PATH) -> int:
    """Return count of stored matches for a league."""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM matches WHERE league_key = ?",
        (league_key,),
    ).fetchone()
    conn.close()
    return row["cnt"]
