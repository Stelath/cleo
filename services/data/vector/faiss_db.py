"""Thread-safe FAISS vector database for storing and searching embeddings."""

import json
import threading
from pathlib import Path

import faiss
import numpy as np


class FaissDB:
    """Thread-safe wrapper around FAISS IndexFlatIP for cosine similarity search.

    Expects normalized vectors (L2 norm = 1) so that inner product == cosine similarity.
    Metadata is stored as a JSON sidecar file alongside the FAISS index.
    """

    def __init__(self, dimension: int, index_path: str | None = None):
        self._lock = threading.Lock()
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []
        self._index_path = Path(index_path) if index_path else None

        if self._index_path and self._index_path.exists():
            self.load(str(self._index_path))

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal

    def add(self, embedding: np.ndarray, metadata: dict | None = None) -> int:
        """Add a single embedding with optional metadata. Returns the assigned ID."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if embedding.shape[1] != self._dimension:
            raise ValueError(
                f"Expected dimension {self._dimension}, got {embedding.shape[1]}"
            )

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with self._lock:
            idx = self._index.ntotal
            self._index.add(embedding)
            self._metadata.append(metadata or {})
            return idx

    def search(
        self, query: np.ndarray, k: int = 5
    ) -> list[tuple[int, float, dict]]:
        """Search for k nearest neighbors. Returns list of (id, score, metadata)."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._dimension:
            raise ValueError(
                f"Expected dimension {self._dimension}, got {query.shape[1]}"
            )

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        with self._lock:
            if self._index.ntotal == 0:
                return []
            k = min(k, self._index.ntotal)
            scores, ids = self._index.search(query, k)
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                results.append((int(idx), float(score), self._metadata[idx]))
            return results

    def save(self, path: str | None = None):
        """Save the FAISS index and metadata sidecar to disk."""
        save_path = Path(path) if path else self._index_path
        if not save_path:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            faiss.write_index(self._index, str(save_path))
            meta_path = save_path.with_suffix(".meta.json")
            self._write_metadata(meta_path, self._metadata)

    def load(self, path: str):
        """Load a FAISS index and metadata sidecar from disk."""
        load_path = Path(path)
        meta_path = load_path.with_suffix(".meta.json")

        with self._lock:
            self._index = faiss.read_index(str(load_path))
            default_metadata = [{} for _ in range(self._index.ntotal)]
            self._metadata = self._load_metadata(meta_path, default_metadata)
            self._index_path = load_path

    def _load_metadata(self, meta_path: Path, default_metadata: list[dict]) -> list[dict]:
        if not meta_path.exists():
            return default_metadata

        try:
            raw = meta_path.read_text(encoding="utf-8")
            if not raw.strip():
                raise ValueError("Metadata sidecar is empty")

            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError("Metadata sidecar must be a JSON array")
            if len(payload) != len(default_metadata):
                raise ValueError(
                    "Metadata length does not match FAISS index size "
                    f"({len(payload)} != {len(default_metadata)})"
                )

            normalized: list[dict] = []
            for entry in payload:
                if isinstance(entry, dict):
                    normalized.append(entry)
                elif entry is None:
                    normalized.append({})
                else:
                    raise ValueError("Metadata entries must be objects")
            return normalized
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            try:
                self._write_metadata(meta_path, default_metadata)
            except OSError:
                pass
            return default_metadata

    @staticmethod
    def _write_metadata(meta_path: Path, metadata: list[dict]) -> None:
        temp_path = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
        temp_path.write_text(json.dumps(metadata), encoding="utf-8")
        temp_path.replace(meta_path)
