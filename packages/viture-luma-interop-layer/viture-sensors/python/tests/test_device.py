import os

import numpy as np
import pytest

vs = pytest.importorskip("viture_sensors")


requires_hardware = pytest.mark.skipif(
    os.getenv("VITURE_HARDWARE") != "1",
    reason="Set VITURE_HARDWARE=1 to run device integration tests",
)


@requires_hardware
def test_device_connect_disconnect():
    device = vs.Device()
    device.connect()
    assert device.is_connected()
    device.disconnect()


@requires_hardware
def test_imu_read():
    device = vs.Device()
    device.connect()
    reading = device.imu.read()
    assert len(reading.accel) == 3
    assert len(reading.gyro) == 3
    assert len(reading.quaternion) == 4
    device.disconnect()


@requires_hardware
def test_camera_capture():
    device = vs.Device()
    device.connect()
    frame = device.camera.capture()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    device.disconnect()


@requires_hardware
def test_depth_capture():
    device = vs.Device()
    device.connect()
    frame = device.depth.capture("left")
    assert frame.ndim == 2
    assert frame.dtype == np.uint16
    device.disconnect()


@requires_hardware
def test_mic_record():
    device = vs.Device()
    device.connect()
    audio = device.mic.record(duration_ms=200, sample_rate=16000)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) > 0
    device.disconnect()
