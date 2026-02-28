"""Hardware integration tests for VITURE AR glasses.

All tests require VITURE_HARDWARE=1 to run; skipped otherwise.
"""

import grpc
import numpy as np

from generated import sensor_pb2, sensor_pb2_grpc
from services.media.camera_transport import CameraFrameAssembler, assembled_frame_to_rgb

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
    audio = audio_recorder.record(duration_ms=500, sample_rate=48000)

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

    stream = stub.CaptureFrame(sensor_pb2.CaptureRequest())

    assembler = CameraFrameAssembler()
    frame = None
    for chunk in stream:
        frame = assembler.push(chunk)
        if frame is not None:
            break

    assert frame is not None

    width = frame.width
    height = frame.height
    assert width > 0
    assert height > 0

    decoded = assembled_frame_to_rgb(frame)
    assert decoded.shape == (height, width, 3)
    assert decoded.std() > 1.0, "gRPC frame appears to be a solid color"

    channel.close()


@requires_hardware
def test_grpc_record_audio(sensor_server):
    """RecordAudio RPC returns valid audio from real hardware."""
    server, port = sensor_server
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = sensor_pb2_grpc.SensorServiceStub(channel)

    response = stub.RecordAudio(
        sensor_pb2.RecordRequest(duration_ms=500, sample_rate=48000)
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
    assembler = CameraFrameAssembler()

    stream = stub.StreamCamera(sensor_pb2.StreamRequest(fps=30))
    for chunk in stream:
        try:
            frame = assembler.push(chunk)
        except ValueError:
            stream.cancel()
            break

        if frame is not None:
            frames.append(frame)

        if len(frames) >= 3:
            stream.cancel()
            break

    assert len(frames) >= 3

    for frame in frames:
        assert frame.width > 0 and frame.height > 0
        decoded = assembled_frame_to_rgb(frame)
        assert decoded.shape == (frame.height, frame.width, 3)

    timestamps = [frame.timestamp for frame in frames]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1], "Timestamps are not monotonically increasing"

    channel.close()
