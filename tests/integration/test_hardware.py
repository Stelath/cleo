"""Hardware integration tests for VITURE AR glasses.

All tests require VITURE_HARDWARE=1 to run; skipped otherwise.
"""

import hashlib
import time

import grpc
import numpy as np
import pytest

from generated import sensor_pb2, sensor_pb2_grpc

from tests.integration.conftest import requires_hardware


# ---------------------------------------------------------------------------
# Direct hardware access
# ---------------------------------------------------------------------------


@requires_hardware
def test_camera_capture_real_frame(usb_camera):
    """Capture a frame from the real camera and validate it."""
    frame = usb_camera.capture()

    # Shape: (H, W, 3) with positive dimensions
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.shape[0] > 0 and frame.shape[1] > 0

    # dtype is uint8
    assert frame.dtype == np.uint8

    # Non-trivial variance (not a dead sensor / solid color)
    assert frame.std() > 1.0, "Frame appears to be a solid color"

    # Two consecutive captures should differ (camera is live)
    frame2 = usb_camera.capture()
    assert not np.array_equal(frame, frame2), "Two captures are identical — camera may be frozen"


@requires_hardware
def test_audio_record_real_samples(audio_recorder):
    """Record audio from the real microphone and validate it."""
    audio = audio_recorder.record(duration_ms=500, sample_rate=16000)

    # Must be float32 with samples
    assert audio.dtype == np.float32
    assert len(audio) > 0

    # Not all zeros
    assert not np.allclose(audio, 0.0), "Audio is all zeros — mic may be dead"

    # Not clipping (all at +/-1.0)
    assert np.abs(audio).max() < 1.0 or audio.std() > 0.001, "Audio appears to be clipping"


# ---------------------------------------------------------------------------
# Sensor service gRPC layer (real hardware behind gRPC)
# ---------------------------------------------------------------------------


@requires_hardware
def test_grpc_capture_frame(sensor_server):
    """CaptureFrame RPC returns a valid image from real hardware."""
    server, port = sensor_server
    channel = grpc.insecure_channel(
        f"localhost:{port}",
        options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
    )
    stub = sensor_pb2_grpc.SensorServiceStub(channel)

    response = stub.CaptureFrame(sensor_pb2.CaptureRequest())

    assert response.width > 0
    assert response.height > 0

    expected_len = response.width * response.height * 3
    assert len(response.data) == expected_len

    frame = np.frombuffer(response.data, dtype=np.uint8).reshape(
        response.height, response.width, 3
    )
    assert frame.std() > 1.0, "gRPC frame appears to be a solid color"

    channel.close()


@requires_hardware
def test_grpc_record_audio(sensor_server):
    """RecordAudio RPC returns valid audio from real hardware."""
    server, port = sensor_server
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = sensor_pb2_grpc.SensorServiceStub(channel)

    response = stub.RecordAudio(
        sensor_pb2.RecordRequest(duration_ms=500, sample_rate=16000)
    )

    assert response.sample_rate > 0
    assert response.num_samples > 0

    audio = np.frombuffer(response.data, dtype=np.float32)
    assert len(audio) == response.num_samples

    channel.close()


@requires_hardware
def test_grpc_stream_camera_frames(sensor_server):
    """StreamCamera RPC delivers live frames with increasing timestamps."""
    server, port = sensor_server
    channel = grpc.insecure_channel(
        f"localhost:{port}",
        options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
    )
    stub = sensor_pb2_grpc.SensorServiceStub(channel)

    frames = []
    stream = stub.StreamCamera(sensor_pb2.StreamRequest(fps=2))
    for frame_msg in stream:
        frames.append(frame_msg)
        if len(frames) >= 3:
            stream.cancel()
            break

    assert len(frames) >= 3

    for f in frames:
        assert f.width > 0 and f.height > 0
        assert len(f.data) == f.width * f.height * 3

    timestamps = [f.timestamp for f in frames]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], "Timestamps are not monotonically increasing"

    channel.close()


# ---------------------------------------------------------------------------
# Frame processor → FAISS pipeline (real camera frames → FAISS)
# ---------------------------------------------------------------------------


@requires_hardware
def test_frame_processor_buffers_real_frames(sensor_server, tmp_path):
    """FrameProcessor with default (stub) embed_frame stores nothing in FAISS."""
    from core.frame_processor import FrameProcessor
    from data.vector.faiss_db import FaissDB

    server, port = sensor_server
    db = FaissDB(dimension=512, index_path=str(tmp_path / "test.index"))
    fp = FrameProcessor(
        sensor_address=f"localhost:{port}",
        faiss_db=db,
    )

    fp.start()
    time.sleep(3)
    fp.stop()
    fp.join(timeout=5)

    # embed_frame returns None by default, so nothing should be stored
    assert db.size == 0


@requires_hardware
def test_frame_processor_with_patched_embed(sensor_server, tmp_path, monkeypatch):
    """FrameProcessor with a patched embed_frame stores entries in FAISS."""
    import core.frame_processor as fp_module
    from core.frame_processor import FrameProcessor
    from data.vector.faiss_db import FaissDB

    def _fake_embed(frame: np.ndarray) -> np.ndarray:
        """Deterministic 512-dim embedding derived from frame content."""
        h = hashlib.sha256(frame.tobytes()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "little"))
        vec = rng.randn(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    monkeypatch.setattr(fp_module, "embed_frame", _fake_embed)

    server, port = sensor_server
    db = FaissDB(dimension=512, index_path=str(tmp_path / "test.index"))
    fp = FrameProcessor(
        sensor_address=f"localhost:{port}",
        faiss_db=db,
    )

    fp.start()
    # Run longer than _BUFFER_SECONDS (10s) so at least one chunk is processed
    time.sleep(12)
    fp.stop()
    fp.join(timeout=5)

    assert db.size >= 1

    # Verify stored metadata
    query = np.random.randn(512).astype(np.float32)
    query /= np.linalg.norm(query)
    results = db.search(query, k=1)
    assert len(results) >= 1

    _id, _score, meta = results[0]
    assert "timestamp" in meta
    assert "chunk_frames" in meta
    assert meta["chunk_frames"] > 0
