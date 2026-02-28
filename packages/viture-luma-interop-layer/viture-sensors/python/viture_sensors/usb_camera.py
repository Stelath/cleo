from __future__ import annotations

import os
import sys
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


class USBCamera:
    """Thin OpenCV wrapper for USB color camera capture."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        max_probe: int = 8,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.device_index = device_index
        self.max_probe = max_probe
        self.width = width
        self.height = height
        self._capture: Optional[cv2.VideoCapture] = None
        self._active_index: Optional[int] = None

    @property
    def active_index(self) -> Optional[int]:
        return self._active_index

    def open(self) -> int:
        if self._capture is not None and self._capture.isOpened():
            return int(self._active_index if self._active_index is not None else -1)

        candidates = [self.device_index] if self.device_index is not None else self._candidate_indices()
        for index in candidates:
            capture = self._open_capture(index)
            if capture is None:
                continue
            if self.width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            if self.height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            self._capture = capture
            self._active_index = index
            return index

        raise RuntimeError("No USB camera device available through OpenCV")

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._active_index = None

    def capture(self) -> NDArray[np.uint8]:
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError("USB camera is not open")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from USB camera")

        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def describe_active_device(self) -> str:
        if self._active_index is None:
            return "none"
        return self.describe_index(self._active_index)

    def __enter__(self) -> "USBCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def available_indices(cls, max_probe: int = 8) -> list[int]:
        found: list[int] = []
        for index in range(max_probe):
            capture = cls._open_capture(index)
            if capture is None:
                continue
            capture.release()
            found.append(index)
        return found

    @classmethod
    def describe_index(cls, index: int) -> str:
        capture = cls._open_capture(index)
        if capture is None:
            return f"index={index} (unavailable)"
        try:
            backend = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown-backend"
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return f"index={index}, backend={backend}, size={width}x{height}"
        finally:
            capture.release()

    def _candidate_indices(self) -> list[int]:
        env_index = os.environ.get("VITURE_USB_CAMERA_INDEX")
        if env_index is not None:
            try:
                return [int(env_index)]
            except ValueError:
                pass
        return list(range(self.max_probe))

    @classmethod
    def _open_capture(cls, index: int) -> Optional[cv2.VideoCapture]:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            return None
        return capture
