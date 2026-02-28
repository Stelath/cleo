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

            CREATE TABLE IF NOT EXISTS apps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                app_type TEXT NOT NULL DEFAULT 'on_demand',
                grpc_address TEXT NOT NULL,
                input_schema_json TEXT NOT NULL DEFAULT '{}',
                enabled INTEGER NOT NULL DEFAULT 1,
                registered_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tracked_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                normalized_title TEXT NOT NULL UNIQUE,
                embedding_json TEXT NOT NULL,
                reference_image_path TEXT NOT NULL,
                registered_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS note_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_text TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS food_macros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                brand TEXT,
                barcode TEXT,
                basis TEXT NOT NULL,
                calories_kcal REAL,
                protein_g REAL,
                fat_g REAL,
                carbs_g REAL,
                serving_size TEXT,
                serving_quantity REAL,
                recorded_at REAL NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                faiss_id INTEGER,
                thumbnail_path TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                display_note TEXT NOT NULL DEFAULT '',
                confidence REAL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                seen_at REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (face_id) REFERENCES faces(id)
            );
            """
        )
        face_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(faces)").fetchall()
        }
        if "display_name" not in face_columns:
            self._conn.execute(
                "ALTER TABLE faces ADD COLUMN display_name TEXT NOT NULL DEFAULT ''"
            )
        if "display_note" not in face_columns:
            self._conn.execute(
                "ALTER TABLE faces ADD COLUMN display_note TEXT NOT NULL DEFAULT ''"
            )
        self._conn.execute(
            """
            INSERT INTO face_sightings (face_id, image_path, seen_at, created_at)
            SELECT faces.id, faces.thumbnail_path, faces.first_seen, faces.created_at
            FROM faces
            WHERE NOT EXISTS (
                SELECT 1
                FROM face_sightings
                WHERE face_sightings.face_id = faces.id
            )
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

    # ── App registration ──

    def upsert_app(
        self,
        name: str,
        description: str,
        app_type: str,
        grpc_address: str,
        input_schema_json: str,
    ) -> tuple[int, bool]:
        """Insert or update an app by name.

        On first insert, ``enabled`` defaults to 1 (true).
        On update, ``enabled`` is preserved — only description, app_type,
        grpc_address, input_schema_json, and updated_at are refreshed.

        Returns:
            (row_id, created) — *created* is True when a new row was inserted.
        """
        now = time.time()
        existing = self._conn.execute(
            "SELECT id FROM apps WHERE name = ?", (name,)
        ).fetchone()

        if existing is None:
            cur = self._conn.execute(
                "INSERT INTO apps "
                "(name, description, app_type, grpc_address, input_schema_json, enabled, registered_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (name, description, app_type, grpc_address, input_schema_json, now, now),
            )
            self._conn.commit()
            return cur.lastrowid, True

        row_id = existing["id"]
        self._conn.execute(
            "UPDATE apps SET description = ?, app_type = ?, grpc_address = ?, "
            "input_schema_json = ?, updated_at = ? WHERE id = ?",
            (description, app_type, grpc_address, input_schema_json, now, row_id),
        )
        self._conn.commit()
        return row_id, False

    def list_apps(
        self, enabled_only: bool = False, app_type: str = ""
    ) -> list[dict]:
        """Return registered apps, optionally filtered."""
        clauses: list[str] = []
        params: list = []
        if enabled_only:
            clauses.append("enabled = 1")
        if app_type:
            clauses.append("app_type = ?")
            params.append(app_type)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM apps{where} ORDER BY name", params
        ).fetchall()
        return [dict(r) for r in rows]

    def set_app_enabled(self, name: str, enabled: bool) -> bool:
        """Toggle the enabled flag for an app.

        Returns True if the app existed, False otherwise.
        """
        cur = self._conn.execute(
            "UPDATE apps SET enabled = ?, updated_at = ? WHERE name = ?",
            (1 if enabled else 0, time.time(), name),
        )
        self._conn.commit()
        return cur.rowcount > 0
    def query_transcriptions_in_range(
        self, start_timestamp: float, end_timestamp: float
    ) -> list[dict]:
        """Return transcription rows that overlap the requested time window."""
        rows = self._conn.execute(
            """
            SELECT *
            FROM transcriptions
            WHERE COALESCE(end_time, created_at) >= ?
              AND COALESCE(start_time, created_at) <= ?
            ORDER BY COALESCE(start_time, created_at) ASC, id ASC
            """,
            (start_timestamp, end_timestamp),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_video_clips_in_range(
        self, start_timestamp: float, end_timestamp: float
    ) -> list[dict]:
        """Return video clips that overlap the requested time window."""
        rows = self._conn.execute(
            """
            SELECT *
            FROM video_clips
            WHERE COALESCE(end_timestamp, created_at) >= ?
              AND COALESCE(start_timestamp, created_at) <= ?
            ORDER BY COALESCE(start_timestamp, created_at) ASC, id ASC
            """,
            (start_timestamp, end_timestamp),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_note_summary(
        self,
        summary_text: str,
        start_timestamp: float,
        end_timestamp: float,
    ) -> int:
        """Insert a generated note summary row. Returns the new row ID."""
        cur = self._conn.execute(
            """
            INSERT INTO note_summaries (summary_text, start_timestamp, end_timestamp, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (summary_text, start_timestamp, end_timestamp, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def query_note_summaries(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """Return paginated note summaries and total count."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM note_summaries"
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT * FROM note_summaries ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    def insert_food_macros(
        self,
        *,
        product_name: str,
        brand: str | None = None,
        barcode: str | None = None,
        basis: str,
        calories_kcal: float | None = None,
        protein_g: float | None = None,
        fat_g: float | None = None,
        carbs_g: float | None = None,
        serving_size: str | None = None,
        serving_quantity: float | None = None,
        recorded_at: float | None = None,
    ) -> int:
        """Insert a food macro row. Returns the new row ID."""
        cur = self._conn.execute(
            """
            INSERT INTO food_macros (
                product_name,
                brand,
                barcode,
                basis,
                calories_kcal,
                protein_g,
                fat_g,
                carbs_g,
                serving_size,
                serving_quantity,
                recorded_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product_name,
                brand,
                barcode,
                basis,
                calories_kcal,
                protein_g,
                fat_g,
                carbs_g,
                serving_size,
                serving_quantity,
                recorded_at if recorded_at is not None else time.time(),
                time.time(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def query_food_macros(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """Return paginated stored food macro rows and total count."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM food_macros"
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT * FROM food_macros ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    # ── Tracked items ──

    def insert_tracked_item(
        self,
        *,
        title: str,
        normalized_title: str,
        embedding_json: str,
        reference_image_path: str,
        registered_at: float,
    ) -> tuple[int, bool]:
        """Insert a tracked item if it does not exist.

        Returns:
            (item_id, created) where created is False for existing items.
        """
        existing = self._conn.execute(
            "SELECT id FROM tracked_items WHERE normalized_title = ?",
            (normalized_title,),
        ).fetchone()
        if existing is not None:
            return int(existing["id"]), False

        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO tracked_items (
                title,
                normalized_title,
                embedding_json,
                reference_image_path,
                registered_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                normalized_title,
                embedding_json,
                reference_image_path,
                registered_at,
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid), True

    def get_tracked_item_by_normalized_title(self, normalized_title: str) -> dict | None:
        """Return tracked item metadata by normalized title."""
        row = self._conn.execute(
            "SELECT * FROM tracked_items WHERE normalized_title = ?",
            (normalized_title,),
        ).fetchone()
        return dict(row) if row else None

    # ── Preferences ──

    def get_preference(self, key: str) -> str | None:
        """Get a preference value by key."""
        row = self._conn.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_preference(self, key: str, value: str) -> None:
        """Set a preference value by key."""
        self._conn.execute(
            "INSERT INTO preferences (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, time.time()),
        )
        self._conn.commit()

    # ── Faces ──

    def insert_face(
        self,
        thumbnail_path: str,
        confidence: float | None = None,
        first_seen: float | None = None,
    ) -> int:
        """Insert a face row. Returns the new row ID."""
        now = time.time()
        first_seen = first_seen or now
        cur = self._conn.execute(
            "INSERT INTO faces (thumbnail_path, confidence, first_seen, last_seen, seen_count, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (thumbnail_path, confidence, first_seen, first_seen, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_face_faiss_id(self, face_id: int, faiss_id: int | None) -> None:
        """Set the FAISS vector ID on an existing face row."""
        self._conn.execute(
            "UPDATE faces SET faiss_id = ? WHERE id = ?",
            (faiss_id, face_id),
        )
        self._conn.commit()

    def update_face_seen(self, face_id: int, last_seen: float) -> None:
        """Increment seen_count and update last_seen timestamp."""
        self._conn.execute(
            "UPDATE faces SET last_seen = ?, seen_count = seen_count + 1 WHERE id = ?",
            (last_seen, face_id),
        )
        self._conn.commit()

    def get_face_by_faiss_id(self, faiss_id: int) -> dict | None:
        """Look up face metadata by its FAISS vector ID."""
        row = self._conn.execute(
            "SELECT * FROM faces WHERE faiss_id = ?", (faiss_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_face(self, face_id: int) -> dict | None:
        """Look up a face row by its primary key."""
        row = self._conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_faces(self, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """Return paginated face rows ordered by most recent sightings."""
        total = self._conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        rows = self._conn.execute(
            "SELECT * FROM faces ORDER BY last_seen DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    def list_all_faces(self) -> list[dict]:
        """Return all stored face rows."""
        rows = self._conn.execute(
            "SELECT * FROM faces ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_face_metadata(
        self,
        face_id: int,
        display_name: str,
        display_note: str,
    ) -> bool:
        """Assign or clear stored face metadata."""
        cur = self._conn.execute(
            "UPDATE faces SET display_name = ?, display_note = ? WHERE id = ?",
            (display_name, display_note, face_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def insert_face_sighting(
        self,
        face_id: int,
        image_path: str,
        seen_at: float,
    ) -> int:
        """Insert a stored image for one face sighting."""
        cur = self._conn.execute(
            "INSERT INTO face_sightings (face_id, image_path, seen_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (face_id, image_path, seen_at, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_face_sightings(self, face_id: int, limit: int = 4) -> list[dict]:
        """Return the most recent stored sightings for a face."""
        rows = self._conn.execute(
            "SELECT * FROM face_sightings WHERE face_id = ? "
            "ORDER BY seen_at DESC, id DESC LIMIT ?",
            (face_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_face_sighting_by_index(self, face_id: int, sighting_index: int) -> dict | None:
        """Return one sighting by recency order for a face."""
        row = self._conn.execute(
            "SELECT * FROM face_sightings WHERE face_id = ? "
            "ORDER BY seen_at DESC, id DESC LIMIT 1 OFFSET ?",
            (face_id, sighting_index),
        ).fetchone()
        return dict(row) if row else None

    def clear_faces(self) -> tuple[int, int, list[str]]:
        """Delete all face rows and sightings, returning counts and file paths."""
        face_rows = self._conn.execute(
            "SELECT thumbnail_path FROM faces"
        ).fetchall()
        sighting_rows = self._conn.execute(
            "SELECT image_path FROM face_sightings"
        ).fetchall()
        image_paths = {
            str(row["thumbnail_path"])
            for row in face_rows
            if row["thumbnail_path"]
        }
        image_paths.update(
            str(row["image_path"])
            for row in sighting_rows
            if row["image_path"]
        )

        sighting_cur = self._conn.execute("DELETE FROM face_sightings")
        face_cur = self._conn.execute("DELETE FROM faces")
        sightings_deleted = sighting_cur.rowcount
        faces_deleted = face_cur.rowcount
        self._conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('faces', 'face_sightings')")
        self._conn.commit()
        return faces_deleted, sightings_deleted, sorted(image_paths)

    def delete_face(self, face_id: int) -> tuple[bool, int, list[str]]:
        """Delete one face row and its sightings, returning deletion status and file paths."""
        face_row = self._conn.execute(
            "SELECT thumbnail_path FROM faces WHERE id = ?",
            (face_id,),
        ).fetchone()
        if face_row is None:
            return False, 0, []

        sighting_rows = self._conn.execute(
            "SELECT image_path FROM face_sightings WHERE face_id = ?",
            (face_id,),
        ).fetchall()
        image_paths: set[str] = set()
        if face_row["thumbnail_path"]:
            image_paths.add(str(face_row["thumbnail_path"]))
        image_paths.update(
            str(row["image_path"])
            for row in sighting_rows
            if row["image_path"]
        )

        sighting_cur = self._conn.execute(
            "DELETE FROM face_sightings WHERE face_id = ?",
            (face_id,),
        )
        face_cur = self._conn.execute(
            "DELETE FROM faces WHERE id = ?",
            (face_id,),
        )
        self._conn.commit()
        return face_cur.rowcount > 0, sighting_cur.rowcount, sorted(image_paths)

    def close(self):
        self._conn.close()
