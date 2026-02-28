"""SQLite database for persisting transcriptions and video clip metadata."""

import sqlite3
import time
from pathlib import Path


class CleoSQLite:
    """Thread-safe SQLite wrapper for transcription and video clip storage.

    Uses WAL mode for concurrent reads and ``check_same_thread=False`` so the
    connection can be shared across gRPC handler threads.
    """

    def __init__(self, db_path: str = "data/cleo.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                confidence REAL,
                start_time REAL,
                end_time REAL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS video_clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_path TEXT NOT NULL,
                faiss_id INTEGER,
                start_timestamp REAL,
                end_timestamp REAL,
                num_frames INTEGER,
                embedding_dimension INTEGER,
                created_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def insert_transcription(
        self,
        text: str,
        confidence: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """Insert a transcription row. Returns the new row ID."""
        cur = self._conn.execute(
            "INSERT INTO transcriptions (text, confidence, start_time, end_time, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (text, confidence, start_time, end_time, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def insert_video_clip(
        self,
        clip_path: str,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
        num_frames: int | None = None,
        embedding_dimension: int | None = None,
    ) -> int:
        """Insert a video clip metadata row. Returns the new row ID."""
        cur = self._conn.execute(
            "INSERT INTO video_clips "
            "(clip_path, start_timestamp, end_timestamp, num_frames, embedding_dimension, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (clip_path, start_timestamp, end_timestamp, num_frames, embedding_dimension, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_clip_faiss_id(self, clip_id: int, faiss_id: int) -> None:
        """Set the FAISS vector ID on an existing clip row."""
        self._conn.execute(
            "UPDATE video_clips SET faiss_id = ? WHERE id = ?",
            (faiss_id, clip_id),
        )
        self._conn.commit()

    def get_clip_metadata(self, clip_id: int) -> dict | None:
        """Return clip metadata as a dict, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM video_clips WHERE id = ?", (clip_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_clip_by_faiss_id(self, faiss_id: int) -> dict | None:
        """Look up clip metadata by its FAISS vector ID."""
        row = self._conn.execute(
            "SELECT * FROM video_clips WHERE faiss_id = ?", (faiss_id,)
        ).fetchone()
        return dict(row) if row else None

    def query_transcriptions(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """Return paginated transcription rows and total count."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM transcriptions"
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    def close(self):
        self._conn.close()
