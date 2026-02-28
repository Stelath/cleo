"""Shared pytest fixtures for Cleo tests."""

import struct

import numpy as np
import pytest

from data.vector.faiss_db import FaissDB
from generated import sensor_pb2, transcription_pb2


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


@pytest.fixture
def mock_camera_frame():
    """Return a CameraFrame with synthetic 4x4 RGB data."""
    width, height = 4, 4
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
    return sensor_pb2.CameraFrame(
        data=data,
        width=width,
        height=height,
        timestamp=1000.0,
    )


@pytest.fixture
def mock_audio_chunk():
    """Return an AudioChunk with synthetic PCM float32 data."""
    sample_rate = 16000
    duration_s = 0.5
    num_samples = int(sample_rate * duration_s)
    audio = np.random.randn(num_samples).astype(np.float32)
    return sensor_pb2.AudioChunk(
        data=audio.tobytes(),
        sample_rate=sample_rate,
        num_samples=num_samples,
        timestamp=1000.0,
    )


@pytest.fixture
def mock_transcription_result():
    """Return a TranscriptionResult with sample data."""
    return transcription_pb2.TranscriptionResult(
        text="hello world",
        confidence=0.95,
        start_time=0.0,
        end_time=1.5,
        is_partial=False,
    )
