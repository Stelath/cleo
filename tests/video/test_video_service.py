"""Tests for video service: encoding, client, and pipeline."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest

from generated import data_pb2
from services.video.service import (
    VideoClipPipeline,
    VideoDataClient,
    _encode_frames_to_mp4,
    _open_mp4_writer,
)
from tests.video.conftest import requires_bedrock

VIDEO_DATA_DIR = Path(__file__).parent / "data"


def _downsample_mp4(mp4_path: Path, fps: int = 2, max_seconds: int = 10) -> bytes:
    """Read an MP4, subsample frames, and re-encode a shorter clip.

    This mirrors what VideoClipPipeline does: it sends a low-FPS,
    short clip to Bedrock for embedding rather than the full video.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(src_fps / fps)))
    max_frames = fps * max_seconds

    frames: list[np.ndarray] = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        idx += 1
    cap.release()

    return _encode_frames_to_mp4(frames, float(fps))


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


def test_open_mp4_writer_success():
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = True

    with patch("services.video.service.cv2.VideoWriter", return_value=mock_writer):
        writer = _open_mp4_writer("/tmp/test.mp4", fps=30.0, width=64, height=48)

    assert writer is mock_writer


def test_open_mp4_writer_returns_none_on_failure():
    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False

    with patch("services.video.service.cv2.VideoWriter", return_value=mock_writer):
        writer = _open_mp4_writer("/tmp/test.mp4", fps=30.0, width=64, height=48)

    assert writer is None
    mock_writer.release.assert_called_once()


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
    num_frames = 60

    # Build mock CameraFrameChunk objects
    mock_frames = []
    for i in range(num_frames):
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
        mock_frames.append(
            sensor_pb2.CameraFrameChunk(
                data=data,
                frame_id=f"frame-{i}",
                chunk_index=0,
                is_last=True,
                width=width,
                height=height,
                timestamp=1000.0 + i / 30.0,
                encoding=sensor_pb2.FRAME_ENCODING_RGB24,
                key_frame=True,
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
        clip_fps=30.0,
        clip_duration=2.0,
    )

    # Run _stream_loop directly (not via thread) with mocked dependencies
    with patch("services.video.service.encode_rgb_to_h264_annexb", return_value=b"\x00\x00\x00\x01frame"):
        with patch("services.video.service.h264_frames_to_mp4", return_value=b"mp4-bytes"):
            with patch("services.video.service.downsample_mp4_for_embedding", return_value=b"embed-bytes"):
                pipeline._stream_loop(mock_sensor_stub, mock_data_client)

    # Should have been called once for target frames
    assert mock_data_client.store_clip.call_count == 1

    call_kwargs = mock_data_client.store_clip.call_args
    # mp4_data (30 FPS full clip) should be non-empty
    assert len(call_kwargs.kwargs.get("mp4_data", call_kwargs[1].get("mp4_data", b""))) > 0
    # embed_data (5 FPS clip) should be non-empty
    assert len(call_kwargs.kwargs.get("embed_data", call_kwargs[1].get("embed_data", b""))) > 0
    # num_frames should be target clip frame count
    assert call_kwargs.kwargs.get("num_frames", call_kwargs[1].get("num_frames", 0)) == num_frames


# ── Bedrock integration tests ──


def _store_both_clips(video_data_client):
    """Store both test videos and return (run_clip_id, study_clip_id)."""
    run_path = VIDEO_DATA_DIR / "run_into_room_video.mp4"
    study_path = VIDEO_DATA_DIR / "study_video.mp4"

    run_mp4 = run_path.read_bytes()
    study_mp4 = study_path.read_bytes()

    # Downsample for Bedrock embedding (mirrors VideoClipPipeline behaviour)
    run_embed = _downsample_mp4(run_path)
    study_embed = _downsample_mp4(study_path)

    run_resp = video_data_client.store_clip(
        mp4_data=run_mp4,
        embed_data=run_embed,
        start_timestamp=0.0,
        end_timestamp=15.0,
        num_frames=450,
    )
    assert run_resp is not None, "Failed to store run_into_room clip"

    study_resp = video_data_client.store_clip(
        mp4_data=study_mp4,
        embed_data=study_embed,
        start_timestamp=15.0,
        end_timestamp=30.0,
        num_frames=450,
    )
    assert study_resp is not None, "Failed to store study clip"

    return run_resp.clip_id, study_resp.clip_id


@requires_bedrock
def test_embed_and_search_running_video(video_data_client, data_stub):
    """Text search for 'running' should rank the running video first."""
    run_clip_id, _ = _store_both_clips(video_data_client)

    resp = data_stub.Search(
        data_pb2.SearchRequest(text="person running down the hall", top_k=2),
        timeout=30.0,
    )
    assert len(resp.results) == 2
    assert resp.results[0].clip_id == run_clip_id


@requires_bedrock
def test_embed_and_search_studying_video(video_data_client, data_stub):
    """Text search for 'studying' should rank the study video first."""
    _, study_clip_id = _store_both_clips(video_data_client)

    resp = data_stub.Search(
        data_pb2.SearchRequest(text="student reading at desk", top_k=2),
        timeout=30.0,
    )
    assert len(resp.results) == 2
    assert resp.results[0].clip_id == study_clip_id
