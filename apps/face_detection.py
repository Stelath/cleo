"""Face detection: streams camera, detects faces via Rekognition, embeds and stores.

Auto-starts on boot. The user can stop/restart via voice commands.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

import cv2
import grpc
import numpy as np
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import data_pb2, data_pb2_grpc, sensor_pb2, sensor_pb2_grpc
from services.config import (
    DATA_ADDRESS,
    FACE_DETECTION_PORT,
    SENSOR_ADDRESS,
)

log = structlog.get_logger()

_FRAME_INTERVAL_SECONDS = 2.0
_STREAM_FPS = 1.0
_MIN_CONFIDENCE = 90.0
_MAX_GRPC_MESSAGE_BYTES = 32 * 1024 * 1024
_RECENT_FACES_BUFFER_SIZE = 50


def _rgb_to_jpeg(frame_data: bytes, width: int, height: int, quality: int = 80) -> bytes:
    """Convert raw RGB frame bytes to JPEG."""
    frame_np = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)
    bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return encoded.tobytes()


def _crop_face(
    frame_data: bytes,
    width: int,
    height: int,
    bbox: dict,
    padding: float = 0.2,
) -> bytes:
    """Crop a face from raw RGB frame data using Rekognition bounding box ratios.

    Args:
        frame_data: Raw RGB bytes.
        width: Frame width in pixels.
        height: Frame height in pixels.
        bbox: Rekognition BoundingBox dict with Width, Height, Left, Top (ratios 0-1).
        padding: Fraction to expand the crop region.

    Returns:
        JPEG-encoded bytes of the cropped face.
    """
    frame_np = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)

    # Convert ratios to pixel coords
    box_left = bbox["Left"] * width
    box_top = bbox["Top"] * height
    box_w = bbox["Width"] * width
    box_h = bbox["Height"] * height

    # Expand by padding
    pad_w = box_w * padding
    pad_h = box_h * padding

    x1 = int(max(0, box_left - pad_w))
    y1 = int(max(0, box_top - pad_h))
    x2 = int(min(width, box_left + box_w + pad_w))
    y2 = int(min(height, box_top + box_h + pad_h))

    cropped = frame_np[y1:y2, x1:x2]
    bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr)
    if not success:
        raise RuntimeError("Failed to encode cropped face as JPEG")
    return encoded.tobytes()


class FaceDetectionLoop(threading.Thread):
    """Daemon thread that streams camera frames and detects faces."""

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        rekognition_client: Any = None,
        data_address: str = DATA_ADDRESS,
        frame_interval: float = _FRAME_INTERVAL_SECONDS,
        stop_event: threading.Event | None = None,
    ):
        super().__init__(daemon=True, name="FaceDetectionLoop")
        self._sensor_address = sensor_address
        self._data_address = data_address
        self._frame_interval = frame_interval
        self._stop_event = stop_event or threading.Event()
        self._recent_faces: collections.deque[tuple[float, int]] = collections.deque(
            maxlen=_RECENT_FACES_BUFFER_SIZE
        )

        if rekognition_client is not None:
            self._rekognition = rekognition_client
        else:
            import boto3

            self._rekognition = boto3.client("rekognition", region_name="us-east-1")

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def recent_faces(self) -> list[tuple[float, int]]:
        return list(self._recent_faces)

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                log.info("face_detection.connecting", sensor=self._sensor_address)
                self._stream_loop()
            except grpc.RpcError as exc:
                if self._stop_event.is_set():
                    break
                log.warning("face_detection.sensor_error", error=str(exc))
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.warning("face_detection.loop_error", error=str(exc))

            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)

    def _stream_loop(self) -> None:
        channel = grpc.insecure_channel(
            self._sensor_address,
            options=[("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES)],
        )
        stub = sensor_pb2_grpc.SensorServiceStub(channel)

        try:
            stream = stub.StreamCamera(sensor_pb2.StreamRequest(fps=_STREAM_FPS))
            last_process_time = 0.0

            for frame in stream:
                if self._stop_event.is_set():
                    break

                now = time.monotonic()
                if now - last_process_time < self._frame_interval:
                    continue

                last_process_time = now
                self._process_frame(frame)
        finally:
            channel.close()

    def _process_frame(self, frame: Any) -> None:
        try:
            jpeg_bytes = _rgb_to_jpeg(frame.data, frame.width, frame.height)

            response = self._rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )

            face_details = response.get("FaceDetails", [])
            if not face_details:
                return

            # Connect to DataService for storing faces
            data_channel = grpc.insecure_channel(self._data_address)
            data_stub = data_pb2_grpc.DataServiceStub(data_channel)

            try:
                for face in face_details:
                    confidence = face.get("Confidence", 0.0)
                    if confidence < _MIN_CONFIDENCE:
                        continue

                    bbox = face["BoundingBox"]
                    cropped = _crop_face(
                        frame.data, frame.width, frame.height, bbox, padding=0.2
                    )

                    timestamp = frame.timestamp or time.time()
                    resp = data_stub.StoreFaceEmbedding(
                        data_pb2.StoreFaceEmbeddingRequest(
                            image_data=cropped,
                            timestamp=timestamp,
                            confidence=confidence,
                        ),
                        timeout=10,
                    )

                    self._recent_faces.append((timestamp, resp.face_id))
                    log.info(
                        "face_detection.face_stored",
                        face_id=resp.face_id,
                        is_new=resp.is_new,
                        confidence=confidence,
                    )
            finally:
                data_channel.close()

        except Exception as exc:
            log.error("face_detection.process_error", error=str(exc))


class FaceDetectionServicer(ToolServiceBase):
    """Continuously detects and tracks faces from the camera."""

    @property
    def tool_name(self) -> str:
        return "face_detection"

    @property
    def tool_type(self) -> str:
        return "active"

    @property
    def tool_description(self) -> str:
        return (
            "Continuously detects and tracks faces from the camera. "
            "Auto-starts on boot. Use 'stop' to pause and 'start' to resume."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": "Start or stop face detection.",
                },
            },
        }

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        rekognition_client: Any = None,
        data_address: str = DATA_ADDRESS,
    ):
        self._sensor_address = sensor_address
        self._rekognition = rekognition_client
        self._data_address = data_address
        self._lock = threading.Lock()

        # Auto-start on boot
        self._loop = FaceDetectionLoop(
            sensor_address=self._sensor_address,
            rekognition_client=self._rekognition,
            data_address=self._data_address,
        )
        self._loop.start()
        log.info("face_detection.auto_started")

    def execute(self, params: dict) -> tuple[bool, str]:
        action = self._resolve_action(params)
        if action == "start":
            return self._start()
        if action == "stop":
            return self._stop()
        return False, "Face detection action must be 'start' or 'stop'"

    def _resolve_action(self, params: dict) -> str | None:
        action = str(params.get("action", "")).strip().lower()
        if action in {"start", "stop"}:
            return action

        query = str(params.get("query", "")).strip().lower()
        if any(word in query for word in ("stop", "end", "pause", "disable", "cancel")):
            return "stop"
        if any(word in query for word in ("start", "begin", "resume", "enable", "detect")):
            return "start"
        return None

    def _start(self) -> tuple[bool, str]:
        with self._lock:
            if self._loop is not None and self._loop.is_alive():
                return True, "Face detection already active"

            self._loop = FaceDetectionLoop(
                sensor_address=self._sensor_address,
                rekognition_client=self._rekognition,
                data_address=self._data_address,
            )
            self._loop.start()

        log.info("face_detection.started")
        return True, "Face detection started"

    def _stop(self) -> tuple[bool, str]:
        with self._lock:
            loop = self._loop
            self._loop = None

        if loop is None:
            return False, "Face detection is not active"

        loop.stop()
        loop.join(timeout=10.0)

        recent = loop.recent_faces
        log.info("face_detection.stopped", faces_detected=len(recent))
        return True, f"Face detection stopped. {len(recent)} face(s) detected during session."


def serve(port: int = FACE_DETECTION_PORT):
    servicer = FaceDetectionServicer()
    serve_tool(servicer, port)


if __name__ == "__main__":
    serve()
