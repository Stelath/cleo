"""Fixtures for sensor service unit tests."""

import numpy as np
import pytest

from generated import sensor_pb2


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
