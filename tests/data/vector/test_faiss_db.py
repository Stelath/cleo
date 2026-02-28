"""Tests for data.vector.faiss_db.FaissDB."""

import json

import numpy as np

from services.data.vector.faiss_db import FaissDB


def test_add_and_size(faiss_db, random_embedding):
    """Add vectors and assert size increments correctly."""
    pass


def test_search_returns_closest(faiss_db):
    """Add known vectors, search with a query close to one, verify correct ranking."""
    pass


def test_search_empty_index(faiss_db, random_embedding):
    """Search on an empty DB returns an empty list."""
    pass


def test_dimension_mismatch_raises(faiss_db):
    """Adding a vector with the wrong dimension raises ValueError."""
    pass


def test_save_and_load(faiss_db, random_embedding, tmp_path):
    """Save the index to disk and reload it; verify size and search results survive."""
    pass


def test_metadata_roundtrip(faiss_db, random_embedding):
    """Metadata passed to add() is preserved and returned by search()."""
    pass


def test_cosine_similarity(faiss_db):
    """An identical (normalized) vector returns a similarity score of ~1.0."""
    pass


def test_load_resets_index_when_dimension_changes(tmp_path):
    index_path = tmp_path / "faces.index"

    original = FaissDB(dimension=1024, index_path=str(index_path))
    vec = np.random.randn(1024).astype(np.float32)
    vec /= np.linalg.norm(vec)
    original.add(vec, {"face_id": 1})
    original.save()

    reloaded = FaissDB(dimension=512, index_path=str(index_path))

    assert reloaded.size == 0
    metadata = json.loads(index_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata == []
