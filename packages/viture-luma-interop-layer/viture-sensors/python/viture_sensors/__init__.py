"""Python API for VITURE sensor access."""

from .audio import AudioRecorder
from .usb_camera import USBCamera

_NATIVE_EXPORTS = {
    "Device",
    "DeviceInfo",
    "IMU",
    "IMUReading",
    "Camera",
    "DepthCamera",
    "Mic",
}


def __getattr__(name: str):
    if name in _NATIVE_EXPORTS:
        from . import _native

        return getattr(_native, name)
    raise AttributeError(f"module 'viture_sensors' has no attribute '{name}'")

__all__ = [
    "Device",
    "DeviceInfo",
    "IMU",
    "IMUReading",
    "Camera",
    "DepthCamera",
    "Mic",
    "USBCamera",
    "AudioRecorder",
]
