"""SQLite results database for match processing state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from wpv.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    match_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    work_dir TEXT,
    track_path TEXT,
    render_path TEXT,
    segments_path TEXT,
    highlights_path TEXT,
    track_coverage_pct REAL,
    quality_passed INTEGER,
    youtube_video_id TEXT,
    youtube_url TEXT,
    uploaded_at TEXT
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'setup',
    playlist_name TEXT,
    playlist_id TEXT,
    default_mask TEXT,
    error_message TEXT,
    current_stage TEXT,
    progress_pct REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS game_clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL REFERENCES games(game_id),
    clip_index INTEGER NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL DEFAULT '',
    uploaded INTEGER NOT NULL DEFAULT 0,
    mask_override TEXT,
    frame_path TEXT,
    UNIQUE(game_id, clip_index)
);
"""


@dataclass
class MatchRecord:
    match_id: str
    source_path: str
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    work_dir: str | None = None
    track_path: str | None = None
    render_path: str | None = None
    segments_path: str | None = None
    highlights_path: str | None = None
    track_coverage_pct: float | None = None
    quality_passed: bool | None = None
    youtube_video_id: str | None = None
    youtube_url: str | None = None
    uploaded_at: str | None = None


def _default_db_path() -> Path:
    return settings.results_db_path.expanduser()


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the results table if it doesn't exist."""
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> MatchRecord:
    d = dict(row)
    # SQLite stores bool as int
    if d.get("quality_passed") is not None:
        d["quality_passed"] = bool(d["quality_passed"])
    return MatchRecord(**d)


def upsert_record(record: MatchRecord, db_path: Path | None = None) -> None:
    """Insert or replace a match record."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        d = asdict(record)
        # Convert bool to int for SQLite
        if d.get("quality_passed") is not None:
            d["quality_passed"] = int(d["quality_passed"])
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        conn.execute(
            f"INSERT OR REPLACE INTO results ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        conn.commit()
    finally:
        conn.close()


def get_record(match_id: str, db_path: Path | None = None) -> MatchRecord | None:
    """Fetch a single match record by ID."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM results WHERE match_id = ?", (match_id,)
        ).fetchone()
        return _row_to_record(row) if row else None
    finally:
        conn.close()


def get_all_records(db_path: Path | None = None) -> list[MatchRecord]:
    """Fetch all match records."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM results ORDER BY match_id").fetchall()
        return [_row_to_record(r) for r in rows]
    finally:
        conn.close()


def update_status(
    match_id: str, status: str, error_message: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Update the status (and optional error) for a match."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE results SET status = ?, error_message = ? WHERE match_id = ?",
            (status, error_message, match_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_field(
    match_id: str, field: str, value: object, db_path: Path | None = None,
) -> None:
    """Update a single field on a match record."""
    valid_fields = {f.name for f in fields(MatchRecord)} - {"match_id"}
    if field not in valid_fields:
        raise ValueError(f"Invalid field: {field}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        if field == "quality_passed" and value is not None:
            value = int(value)
        conn.execute(
            f"UPDATE results SET {field} = ? WHERE match_id = ?",  # noqa: S608
            (value, match_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Game records (web UI)
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    game_id: str
    name: str
    created_at: str
    status: str = "setup"
    playlist_name: str | None = None
    playlist_id: str | None = None
    default_mask: str | None = None  # JSON polygon
    error_message: str | None = None
    current_stage: str | None = None
    progress_pct: float = 0


@dataclass
class GameClipRecord:
    game_id: str
    clip_index: int
    id: int | None = None
    filename: str = ""
    source_path: str = ""
    uploaded: bool = False
    mask_override: str | None = None  # JSON polygon or None (use game default)
    frame_path: str | None = None


def _row_to_game(row: sqlite3.Row) -> GameRecord:
    d = dict(row)
    return GameRecord(**d)


def _row_to_clip(row: sqlite3.Row) -> GameClipRecord:
    d = dict(row)
    d["uploaded"] = bool(d.get("uploaded", 0))
    return GameClipRecord(**d)


def create_game(game: GameRecord, db_path: Path | None = None) -> None:
    """Insert a new game record."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        d = asdict(game)
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" for _ in d)
        conn.execute(
            f"INSERT INTO games ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        conn.commit()
    finally:
        conn.close()


def get_game(game_id: str, db_path: Path | None = None) -> GameRecord | None:
    """Fetch a single game by ID."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        return _row_to_game(row) if row else None
    finally:
        conn.close()


def get_all_games(db_path: Path | None = None) -> list[GameRecord]:
    """Fetch all games ordered by creation time (newest first)."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM games ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_game(r) for r in rows]
    finally:
        conn.close()


def update_game_field(
    game_id: str, field: str, value: object, db_path: Path | None = None,
) -> None:
    """Update a single field on a game record."""
    valid = {f.name for f in fields(GameRecord)} - {"game_id"}
    if field not in valid:
        raise ValueError(f"Invalid game field: {field}")
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            f"UPDATE games SET {field} = ? WHERE game_id = ?",  # noqa: S608
            (value, game_id),
        )
        conn.commit()
    finally:
        conn.close()


def add_game_clip(clip: GameClipRecord, db_path: Path | None = None) -> None:
    """Insert or replace a game clip record."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO game_clips "
            "(game_id, clip_index, filename, source_path, uploaded, mask_override, frame_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                clip.game_id, clip.clip_index, clip.filename, clip.source_path,
                int(clip.uploaded), clip.mask_override, clip.frame_path,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_game_clips(game_id: str, db_path: Path | None = None) -> list[GameClipRecord]:
    """Fetch all clips for a game, ordered by clip_index."""
    init_db(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM game_clips WHERE game_id = ? ORDER BY clip_index",
            (game_id,),
        ).fetchall()
        return [_row_to_clip(r) for r in rows]
    finally:
        conn.close()


def update_clip_field(
    game_id: str, clip_index: int, field: str, value: object,
    db_path: Path | None = None,
) -> None:
    """Update a single field on a game clip."""
    valid = {f.name for f in fields(GameClipRecord)} - {"game_id", "clip_index", "id"}
    if field not in valid:
        raise ValueError(f"Invalid clip field: {field}")
    if field == "uploaded" and value is not None:
        value = int(value)
    init_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            f"UPDATE game_clips SET {field} = ? "  # noqa: S608
            "WHERE game_id = ? AND clip_index = ?",
            (value, game_id, clip_index),
        )
        conn.commit()
    finally:
        conn.close()


def get_clip_mask(game_id: str, clip_index: int, db_path: Path | None = None) -> list | None:
    """Get the effective mask for a clip (override or game default). Returns parsed polygon."""
    game = get_game(game_id, db_path)
    if game is None:
        return None
    clips = get_game_clips(game_id, db_path)
    for c in clips:
        if c.clip_index == clip_index:
            if c.mask_override:
                return json.loads(c.mask_override)
            break
    if game.default_mask:
        return json.loads(game.default_mask)
    return None
