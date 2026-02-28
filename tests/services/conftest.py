"""Fixtures for sensor service unit tests."""

import numpy as np
import pytest

from generated import sensor_pb2
from services.media.camera_transport import encode_rgb_to_jpeg


@pytest.fixture
def mock_camera_frame():
    """Return a single-chunk CameraFrameChunk with synthetic 4x4 JPEG data."""
    width, height = 4, 4
    frame_rgb = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    data = encode_rgb_to_jpeg(frame_rgb)
    return sensor_pb2.CameraFrameChunk(
        data=data,
        frame_id="test-frame",
        chunk_index=0,
        is_last=True,
        width=width,
        height=height,
        timestamp=1000.0,
        encoding=sensor_pb2.FRAME_ENCODING_JPEG,
        key_frame=True,
    )


@pytest.fixture
def mock_audio_chunk():
    """Return an AudioChunk with synthetic PCM float32 data."""
    sample_rate = 48000
    duration_s = 0.5
    num_samples = int(sample_rate * duration_s)
    audio = np.random.randn(num_samples).astype(np.float32)
    return sensor_pb2.AudioChunk(
        data=audio.tobytes(),
        sample_rate=sample_rate,
        num_samples=num_samples,
        timestamp=1000.0,
    )
