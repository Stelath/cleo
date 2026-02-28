"""Navigator: continuous visual guide for blind/VI users.

Streams camera frames to Claude Sonnet via Bedrock, generates spoken guidance
text every ~3 seconds, and runs continuously until stopped.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Any

import cv2
import grpc
import numpy as np
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import frontend_pb2, frontend_pb2_grpc, sensor_pb2, sensor_pb2_grpc
from services.config import FRONTEND_ADDRESS, NAVIGATOR_PORT, SENSOR_ADDRESS
from services.media.camera_transport import (
    AssembledCameraFrame,
    CameraFrameAssembler,
    assembled_frame_to_rgb,
)

log = structlog.get_logger()

_MODEL_ID = os.environ.get(
    "CLEO_NAVIGATOR_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
_BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
_FRAME_INTERVAL_SECONDS = 3.0
_STREAM_FPS = 1.0
_GUIDANCE_BUFFER_SIZE = 50

_SYSTEM_PROMPT = (
    "You are a visual guide helping a blind or visually impaired person navigate "
    "their surroundings safely through AR glasses. You receive camera frames from "
    "their perspective.\n\n"
    "Rules:\n"
    "- SAFETY FIRST: Always prioritize obstacles, vehicles, stairs, curbs, "
    "approaching people, and other hazards.\n"
    "- Be concise: 1-3 short sentences. Your text will be spoken aloud.\n"
    "- Use clock directions: 12 o'clock = straight ahead, 3 = right, 9 = left, "
    "6 = behind.\n"
    "- Describe CHANGES from the previous observation. If nothing has changed, "
    'say "all clear" or similar.\n'
    '- Do not guess distances in feet/meters. Use "nearby", "a few steps ahead", '
    '"far ahead".\n'
    "- Do not describe the image format or quality. Focus only on what matters "
    "for safe navigation."
)

_MAX_GRPC_MESSAGE_BYTES = 32 * 1024 * 1024


def _rgb_to_jpeg(frame_rgb: np.ndarray, quality: int = 80) -> bytes:
    """Convert an RGB frame array to JPEG."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return encoded.tobytes()


class NavigatorBedrockClient:
    """Bedrock Converse client for VLM frame analysis."""

    def __init__(
        self,
        client: Any = None,
        model_id: str = _MODEL_ID,
        region: str = _BEDROCK_REGION,
    ):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def analyze_frame(
        self,
        jpeg_bytes: bytes,
        user_context: str,
        previous_guidance: str | None = None,
    ) -> str:
        """Send a camera frame to Claude Sonnet and get navigation guidance."""
        user_text = f"User context: {user_context}"
        if previous_guidance:
            user_text += f"\n\nPrevious guidance: {previous_guidance}"
        user_text += "\n\nDescribe what you see and provide navigation guidance."

        content: list[dict[str, Any]] = [
            {"image": {"format": "jpeg", "source": {"bytes": jpeg_bytes}}},
            {"text": user_text},
        ]

        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": content}],
        )

        text_parts = [
            block["text"]
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if "text" in block
        ]
        return "\n".join(text_parts).strip()


class NavigatorLoop(threading.Thread):
    """Daemon thread that continuously streams camera and calls VLM for guidance."""

    def __init__(
        self,
        user_context: str,
        sensor_address: str = SENSOR_ADDRESS,
        bedrock_client: NavigatorBedrockClient | None = None,
        frame_interval: float = _FRAME_INTERVAL_SECONDS,
        guidance_callback: Any = None,
    ):
        super().__init__(daemon=True, name="NavigatorLoop")
        self._user_context = user_context
        self._sensor_address = sensor_address
        self._bedrock = bedrock_client or NavigatorBedrockClient()
        self._frame_interval = frame_interval
        self._guidance_callback = guidance_callback
        self._stop_event = threading.Event()
        self._guidance_buffer: collections.deque[tuple[float, str]] = collections.deque(
            maxlen=_GUIDANCE_BUFFER_SIZE
        )
        self._previous_guidance: str | None = None

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def recent_guidance(self) -> list[tuple[float, str]]:
        return list(self._guidance_buffer)

    def run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                log.info("navigator.connecting", sensor=self._sensor_address)
                self._stream_loop()
            except grpc.RpcError as exc:
                if self._stop_event.is_set():
                    break
                log.warning("navigator.sensor_error", error=str(exc))
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                log.warning("navigator.loop_error", error=str(exc))

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

            for frame in self._iter_frames(stream):
                if self._stop_event.is_set():
                    break

                now = time.monotonic()
                if now - last_process_time < self._frame_interval:
                    continue

                last_process_time = now
                self._process_frame(frame)
        finally:
            channel.close()

    def _iter_frames(self, stream: Any):
        assembler = CameraFrameAssembler()

        for chunk in stream:
            try:
                frame = assembler.push(chunk)
            except ValueError as exc:
                log.warning(
                    "navigator.frame_chunk_invalid",
                    error=str(exc),
                )
                continue

            if frame is not None:
                yield frame

    def _speak_guidance(self, guidance: str) -> None:
        """Send guidance text to FrontendService for TTS. Errors are logged and swallowed."""
        try:
            channel = grpc.insecure_channel(FRONTEND_ADDRESS)
            try:
                stub = frontend_pb2_grpc.FrontendServiceStub(channel)
                stub.SpeakText(frontend_pb2.SpeakTextRequest(text=guidance))
            finally:
                channel.close()
        except Exception as exc:
            log.warning("navigator.speak_error", error=str(exc))

    def _process_frame(
        self,
        frame: AssembledCameraFrame | Any,
        width: int | None = None,
        height: int | None = None,
        timestamp: float | None = None,
        encoding: int | None = None,
    ) -> None:
        if isinstance(frame, AssembledCameraFrame):
            assembled = frame
        elif isinstance(frame, bytes):
            if width is None or height is None:
                raise ValueError("width and height are required when passing raw frame bytes")
            assembled = AssembledCameraFrame(
                frame_id="legacy-bytes",
                data=frame,
                width=width,
                height=height,
                timestamp=timestamp or 0.0,
                encoding=encoding or sensor_pb2.FRAME_ENCODING_RGB24,
                key_frame=True,
            )
        else:
            assembled = AssembledCameraFrame(
                frame_id=getattr(frame, "frame_id", "legacy-object"),
                data=getattr(frame, "data"),
                width=int(getattr(frame, "width")),
                height=int(getattr(frame, "height")),
                timestamp=float(getattr(frame, "timestamp", timestamp or 0.0)),
                encoding=int(getattr(frame, "encoding", encoding or sensor_pb2.FRAME_ENCODING_RGB24)),
                key_frame=bool(getattr(frame, "key_frame", True)),
            )

        try:
            frame_rgb = assembled_frame_to_rgb(assembled)
            jpeg_bytes = _rgb_to_jpeg(frame_rgb)
            guidance = self._bedrock.analyze_frame(
                jpeg_bytes, self._user_context, self._previous_guidance
            )
            self._previous_guidance = guidance
            frame_ts = assembled.timestamp or time.time()
            self._guidance_buffer.append((frame_ts, guidance))
            log.info("navigator.guidance", guidance=guidance[:100])
            self._speak_guidance(guidance)

            if self._guidance_callback:
                self._guidance_callback(guidance)
        except Exception as exc:
            log.error("navigator.process_error", error=str(exc))


class NavigatorServicer(ToolServiceBase):
    """Continuous visual navigation guide for blind/VI users."""

    @property
    def tool_name(self) -> str:
        return "navigator"

    @property
    def tool_type(self) -> str:
        return "active"

    @property
    def tool_description(self) -> str:
        return (
            "Continuous visual guide for blind and visually impaired users. "
            "Streams camera frames and analyzes the scene for obstacles, hazards, "
            "curbs, stairs, vehicles, and people. Provides real-time spoken guidance "
            "using clock directions. Use action 'start' with a query describing what "
            "to focus on, and 'stop' to end."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": "Start or stop the navigator.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Context for what the user wants help with, "
                        "e.g. 'help me find curbs' or 'tell me when someone comes'."
                    ),
                },
            },
        }

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        bedrock_client: NavigatorBedrockClient | None = None,
        frontend_address: str = FRONTEND_ADDRESS,
    ):
        self._sensor_address = sensor_address
        self._bedrock = bedrock_client
        self._lock = threading.Lock()
        self._loop: NavigatorLoop | None = None

        # Frontend indicator stub
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)

    def _set_indicator(self, active: bool) -> None:
        try:
            self._frontend.SetAppIndicator(
                frontend_pb2.AppIndicatorRequest(app_name="navigator", is_active=active)
            )
        except grpc.RpcError as exc:
            log.warning("navigator.indicator_error", error=str(exc))

    def execute(self, params: dict) -> tuple[bool, str]:
        action = self._resolve_action(params)
        if action == "start":
            query = str(params.get("query", "")).strip() or "general navigation guidance"
            return self._start_navigation(query)
        if action == "stop":
            return self._stop_navigation()
        return False, "Navigator action must be 'start' or 'stop'"

    def _resolve_action(self, params: dict) -> str | None:
        action = str(params.get("action", "")).strip().lower()
        if action in {"start", "stop"}:
            return action

        query = str(params.get("query", "")).strip().lower()
        if any(word in query for word in ("help", "navigate", "guide", "find", "look", "watch")):
            return "start"
        if any(word in query for word in ("stop", "end", "cancel", "quit")):
            return "stop"
        return None

    def _start_navigation(self, context: str) -> tuple[bool, str]:
        with self._lock:
            if self._loop is not None and self._loop.is_alive():
                return True, f"Navigator already active: {context}"

            loop = NavigatorLoop(
                user_context=context,
                sensor_address=self._sensor_address,
                bedrock_client=self._bedrock or NavigatorBedrockClient(),
            )
            loop.start()
            self._loop = loop

        log.info("navigator.started", context=context)
        self._set_indicator(True)
        return True, f"Navigator started: {context}"

    def _stop_navigation(self) -> tuple[bool, str]:
        with self._lock:
            loop = self._loop
            self._loop = None

        if loop is None:
            return False, "Navigator is not active"

        loop.stop()
        loop.join(timeout=10.0)

        recent = loop.recent_guidance
        last_guidance = recent[-1][1] if recent else "No guidance was generated."
        self._set_indicator(False)
        log.info("navigator.stopped", guidance_count=len(recent))
        return True, f"Navigator stopped. Last observation: {last_guidance}"


def serve(port: int = NAVIGATOR_PORT):
    servicer = NavigatorServicer()
    serve_tool(servicer, port)


if __name__ == "__main__":
    serve()
