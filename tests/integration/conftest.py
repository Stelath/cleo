"""Fixtures for hardware integration tests.

All fixtures here open real hardware and require VITURE_HARDWARE=1.
"""

import os
from concurrent import futures

import grpc
import pytest

from generated import sensor_pb2_grpc

requires_hardware = pytest.mark.skipif(
    os.getenv("VITURE_HARDWARE") != "1",
    reason="Set VITURE_HARDWARE=1 to run device integration tests",
)


@pytest.fixture
def usb_camera():
    """Open a real USBCamera, yield it, and close on teardown."""
    from viture_sensors import USBCamera

    with USBCamera() as cam:
        yield cam


@pytest.fixture
def audio_recorder():
    """Create a real AudioRecorder, yield it, and close on teardown."""
    from viture_sensors import AudioRecorder

    with AudioRecorder() as rec:
        yield rec


@pytest.fixture
def sensor_server():
    """Start a real SensorServiceServicer on an ephemeral gRPC port.

    Yields (server, port) and shuts down on teardown.
    """
    from services.media.sensor_service import SensorServiceServicer

    servicer = SensorServiceServicer()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        ],
    )
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    yield server, port

    servicer.shutdown()
    server.stop(grace=2)
