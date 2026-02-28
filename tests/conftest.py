"""Shared pytest fixtures for Cleo tests."""

import numpy as np
import pytest

from data.vector.faiss_db import FaissDB


@pytest.fixture
def faiss_db(tmp_path):
    """Create a FaissDB instance backed by a temp directory."""
    index_path = str(tmp_path / "test.index")
    return FaissDB(dimension=128, index_path=index_path)


@pytest.fixture
def random_embedding():
    """Factory that returns a random normalized float32 vector of a given dimension."""

    def _make(dim: int = 128) -> np.ndarray:
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    return _make
