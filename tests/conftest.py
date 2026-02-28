"""Shared pytest fixtures and hooks for Cleo tests."""

import os

import numpy as np
import pytest


def pytest_addoption(parser):
    """Register project-specific pytest CLI flags."""
    parser.addoption(
        "--run-transcription-e2e",
        action="store_true",
        default=False,
        help="Run opt-in Parakeet transcription end-to-end tests",
    )


def pytest_configure(config):
    """Register custom markers used across the suite."""
    config.addinivalue_line(
        "markers",
        "integration: integration tests that may rely on external services or hardware",
    )
    config.addinivalue_line(
        "markers",
        "transcription_e2e: opt-in end-to-end transcription tests that are skipped by default",
    )


def pytest_collection_modifyitems(config, items):
    """Skip expensive transcription E2E tests unless explicitly enabled."""
    if config.getoption("--run-transcription-e2e") or os.getenv("RUN_TRANSCRIPTION_E2E") == "1":
        return

    skip_transcription_e2e = pytest.mark.skip(
        reason=(
            "Use --run-transcription-e2e or set RUN_TRANSCRIPTION_E2E=1 "
            "to run transcription end-to-end tests"
        )
    )
    for item in items:
        if "transcription_e2e" in item.keywords:
            item.add_marker(skip_transcription_e2e)


@pytest.fixture
def faiss_db(tmp_path):
    """Create a FaissDB instance backed by a temp directory."""
    faiss = pytest.importorskip("faiss")
    del faiss
    from data.vector.faiss_db import FaissDB

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
