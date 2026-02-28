"""Tests for video service: encoding, client, and pipeline."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from services.video.service import (
    VideoClipPipeline,
    VideoDataClient,
    _encode_frames_to_mp4,
)


def _make_rgb_frames(n: int, width: int = 64, height: int = 48) -> list[np.ndarray]:
    """Generate n synthetic RGB frames."""
    return [
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(n)
    ]


# ── _encode_frames_to_mp4 ──


def test_encode_frames_to_mp4_basic():
    frames = _make_rgb_frames(10)
    mp4_bytes = _encode_frames_to_mp4(frames, fps=5.0)
    assert len(mp4_bytes) > 0
    # MP4 files start with an ftyp box or mdat — just verify non-empty bytes
    assert isinstance(mp4_bytes, bytes)


def test_encode_frames_empty_list():
    result = _encode_frames_to_mp4([], fps=30.0)
    assert result == b""


# ── VideoDataClient ──


def test_video_data_client_store_clip_failure():
    """Unreachable DataService should return None, not raise."""
    client = VideoDataClient(address="localhost:1", timeout=0.5)
    try:
        result = client.store_clip(
            mp4_data=b"\x00" * 100,
            embed_data=b"\x00" * 50,
            start_timestamp=1.0,
            end_timestamp=16.0,
            num_frames=450,
        )
        assert result is None
    finally:
        client.close()


# ── VideoClipPipeline ──


def test_pipeline_stop_event():
    """Pipeline thread should terminate promptly when stop is called."""
    pipeline = VideoClipPipeline(
        sensor_address="localhost:1",
        data_address="localhost:1",
    )
    pipeline.start()
    time.sleep(0.1)
    pipeline.stop()
    pipeline.join(timeout=5)
    assert not pipeline.is_alive()


def test_accumulate_and_encode_integration():
    """Mock sensor yields 450 frames; verify DataService called with both MP4s."""
    from generated import sensor_pb2

    width, height = 64, 48
    num_frames = 450

    # Build mock CameraFrame objects
    mock_frames = []
    for i in range(num_frames):
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
        mock_frames.append(
            sensor_pb2.CameraFrame(
                data=data,
                width=width,
                height=height,
                timestamp=1000.0 + i / 30.0,
            )
        )

    # Patch the sensor stub to yield our mock frames then stop
    mock_sensor_stub = MagicMock()
    mock_sensor_stub.StreamCamera.return_value = iter(mock_frames)

    # Track DataService calls
    store_calls = []

    def fake_store_clip(**kwargs):
        store_calls.append(kwargs)
        from generated import data_pb2

        return data_pb2.StoreVideoClipResponse(clip_id=1, faiss_id=1)

    mock_data_client = MagicMock()
    mock_data_client.store_clip.side_effect = fake_store_clip

    pipeline = VideoClipPipeline(
        sensor_address="localhost:1",
        data_address="localhost:1",
    )

    # Run _stream_loop directly (not via thread) with mocked dependencies
    pipeline._stream_loop(mock_sensor_stub, mock_data_client)

    # Should have been called once for 450 frames
    assert mock_data_client.store_clip.call_count == 1

    call_kwargs = mock_data_client.store_clip.call_args
    # mp4_data (30 FPS full clip) should be non-empty
    assert len(call_kwargs.kwargs.get("mp4_data", call_kwargs[1].get("mp4_data", b""))) > 0
    # embed_data (5 FPS clip) should be non-empty
    assert len(call_kwargs.kwargs.get("embed_data", call_kwargs[1].get("embed_data", b""))) > 0
    # num_frames should be 450
    assert call_kwargs.kwargs.get("num_frames", call_kwargs[1].get("num_frames", 0)) == 450
