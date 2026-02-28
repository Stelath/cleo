"""Shared pytest fixtures for Cleo tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from assistant.registry import ToolDefinition, ToolRegistry
from data.vector.faiss_db import FaissDB


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


@pytest.fixture
def tool_registry():
    """Registry with a single test tool."""
    tool = ToolDefinition(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        grpc_address="localhost:50099",
    )
    return ToolRegistry(tools=[tool])


@pytest.fixture
def mock_grpc_context():
    """Mock gRPC servicer context."""
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx
