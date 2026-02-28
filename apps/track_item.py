"""Track item tools: register an item and locate its latest occurrence."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import data_pb2, data_pb2_grpc, frontend_pb2, frontend_pb2_grpc, sensor_pb2, sensor_pb2_grpc
from services.config import (
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    SENSOR_ADDRESS,
    TRACK_ITEM_LOCATE_PORT,
    TRACK_ITEM_REGISTER_PORT,
)
from services.media.camera_transport import (
    AssembledCameraFrame,
    CameraFrameAssembler,
    assembled_frame_to_rgb,
    encode_rgb_to_jpeg,
)

log = structlog.get_logger()

_MODEL_ID = os.environ.get(
    "CLEO_TRACK_ITEM_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
_BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
_LOCATE_TOP_K_DEFAULT = 128
_MAX_INLINE_VIDEO_BYTES = 24 * 1024 * 1024
_LOCATE_WAIT_FOR_FRESH_CLIP_SECONDS = float(
    os.environ.get("CLEO_TRACK_ITEM_LOCATE_WAIT_SECONDS", "20")
)
_LOCATE_POLL_INTERVAL_SECONDS = 2.0


class TrackItemVisionBedrockClient:
    """Bedrock VLM client for registration-time visibility checks."""

    def __init__(self, client: Any = None, model_id: str = _MODEL_ID, region: str = _BEDROCK_REGION):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def verify_trackable_item(
        self,
        image_bytes: bytes,
        *,
        user_query: str = "",
        title_hint: str = "",
    ) -> tuple[bool, str, str | None]:
        prompt = (
            "Inspect this image and decide if there is a clearly visible, trackable physical item. "
            "A trackable item is a specific object like a phone, wallet, keys, bottle, bag, or headphones. "
            "If the scene has no clear object, return has_trackable_item=false. "
            "Reply with JSON only using this schema: "
            '{"has_trackable_item": true|false, "reason": "string", "suggested_title": "string or null"}. '
            "Keep reason brief."
        )
        if title_hint:
            prompt += f" User requested item title: {title_hint}."
        if user_query:
            prompt += f" User request context: {user_query}."

        response = self._client.converse(
            modelId=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": prompt},
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {"bytes": image_bytes},
                            }
                        },
                    ],
                }
            ],
        )

        text = "\n".join(
            block["text"]
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if "text" in block
        ).strip()
        payload = _parse_json_object(text)
        has_item = _coerce_bool(payload.get("has_trackable_item")) if payload else False
        reason = str(payload.get("reason") or "").strip() if payload else ""
        suggested_title_value = payload.get("suggested_title") if payload else None
        suggested_title = (
            str(suggested_title_value).strip()
            if suggested_title_value not in (None, "")
            else None
        )
        return has_item, reason, suggested_title


class TrackItemRegisterServicer(ToolServiceBase):
    """Registers a single user-titled item for later retrieval."""

    @property
    def tool_name(self) -> str:
        return "track_item_register"

    @property
    def tool_description(self) -> str:
        return (
            "Register a visible item to track later. Captures the current camera frame, "
            "verifies there is a real trackable object present, and stores a reference embedding "
            "for the provided title."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short item name to register (for example: phone, wallet, keys).",
                },
                "query": {
                    "type": "string",
                    "description": "Optional natural-language request text.",
                },
            },
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
        sensor_address: str = SENSOR_ADDRESS,
        vision_client: TrackItemVisionBedrockClient | None = None,
    ):
        self._vision = vision_client or TrackItemVisionBedrockClient()
        self._data_channel = grpc.insecure_channel(data_address)
        self._data = data_pb2_grpc.DataServiceStub(self._data_channel)
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)
        self._sensor_channel = grpc.insecure_channel(sensor_address)
        self._sensor = sensor_pb2_grpc.SensorServiceStub(self._sensor_channel)

    def close(self) -> None:
        self._data_channel.close()
        self._frontend_channel.close()
        self._sensor_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        query = str(params.get("query", "")).strip()
        title = _clean_item_title(params.get("title", ""))

        captured_frame = self._capture_frame()
        if captured_frame is None or not captured_frame.data:
            return False, "Failed to capture a camera frame for item registration"

        image_bytes = _captured_frame_to_jpeg(captured_frame)
        has_item, reason, suggested_title = self._vision.verify_trackable_item(
            image_bytes,
            user_query=query,
            title_hint=title,
        )
        if not has_item:
            detail = f" ({reason})" if reason else ""
            return False, f"No visible trackable item in view{detail}."

        resolved_title = title or _clean_item_title(suggested_title or "")
        if not resolved_title:
            return False, "Please provide a short title for the item to track (for example: phone)."

        registered_at = captured_frame.timestamp if captured_frame.timestamp > 0 else time.time()

        response = self._data.StoreTrackedItem(
            data_pb2.StoreTrackedItemRequest(
                title=resolved_title,
                image_data=image_bytes,
                registered_at=registered_at,
            )
        )

        if not response.created:
            message = f"Item '{response.title}' is already registered."
            self._notify(
                title="Item already tracked",
                message=message,
                style="info",
            )
            return True, message

        message = f"Registered '{response.title}' for tracking."
        self._notify(
            title="Item registered",
            message=message,
            style="success",
        )
        return True, message

    def _capture_frame(self) -> AssembledCameraFrame | None:
        stream = self._sensor.CaptureFrame(sensor_pb2.CaptureRequest())
        assembler = CameraFrameAssembler()
        for chunk in stream:
            frame = assembler.push(chunk)
            if frame is not None:
                return frame
        return None

    def _notify(self, *, title: str, message: str, style: str) -> None:
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title=title,
                    message=message,
                    style=style,
                )
            )
        except grpc.RpcError as exc:
            log.warning("track_item.register_notification_failed", error=str(exc))


class TrackItemLocateServicer(ToolServiceBase):
    """Locates the latest video occurrence for a previously tracked item."""

    @property
    def tool_name(self) -> str:
        return "track_item_locate"

    @property
    def tool_description(self) -> str:
        return (
            "Find where a previously tracked item was last seen. "
            "Looks up the registered item embedding and searches recent video memory "
            "for the most recent occurrence."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Tracked item title to locate (for example: phone, wallet, keys).",
                },
                "query": {
                    "type": "string",
                    "description": "Optional natural-language locate request.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional candidate count to scan before picking the latest occurrence.",
                },
                "min_score": {
                    "type": "number",
                    "description": "Optional similarity threshold from 0.0 to 1.0.",
                },
            },
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
        locate_wait_seconds: float = _LOCATE_WAIT_FOR_FRESH_CLIP_SECONDS,
        locate_poll_interval_seconds: float = _LOCATE_POLL_INTERVAL_SECONDS,
    ):
        self._data_channel = grpc.insecure_channel(data_address)
        self._data = data_pb2_grpc.DataServiceStub(self._data_channel)
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)
        self._locate_wait_seconds = max(0.0, float(locate_wait_seconds))
        self._locate_poll_interval_seconds = max(0.1, float(locate_poll_interval_seconds))

    def close(self) -> None:
        self._data_channel.close()
        self._frontend_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        title = _clean_item_title(params.get("title", ""))
        query = str(params.get("query", "")).strip()
        if not title:
            title = _extract_item_title_from_query(query)
        if not title:
            return False, "Tell me which tracked item to find, for example: where did I leave my phone?"

        top_k = _coerce_int(params.get("top_k"), default=_LOCATE_TOP_K_DEFAULT)
        min_score = _coerce_float(params.get("min_score"))
        request = data_pb2.FindLatestTrackedItemOccurrenceRequest(
            title=title,
            top_k=top_k,
        )
        if min_score is not None:
            request.min_score = min_score

        response = self._find_latest_occurrence_with_retry(request)
        if not response.found_item:
            return True, f"I don't have a tracked item named '{title}'. Register it first."

        if not response.found_occurrence:
            log.info(
                "track_item.locate_not_found",
                title=title,
                top_k=top_k,
                min_score=min_score,
                wait_seconds=self._locate_wait_seconds,
            )
            return True, f"I couldn't find '{response.title}' in recent video memory yet."

        result = response.latest_result
        seen_timestamp = result.end_timestamp or result.start_timestamp
        when = _format_relative_time(seen_timestamp)
        message = (
            f"Last saw '{response.title}' {when} "
            f"(clip #{result.clip_id}, similarity {result.score:.2f})."
        )
        self._show_result_card(
            title=response.title,
            clip_id=result.clip_id,
            score=result.score,
            when=when,
        )
        played = self._play_clip(result.clip_id, title=response.title)
        if played:
            message += " Playing the clip now."
        else:
            message += " I found it, but I could not open clip playback."
        return True, message

    def _show_result_card(self, *, title: str, clip_id: int, score: float, when: str) -> None:
        try:
            self._frontend.ShowCard(
                frontend_pb2.CardRequest(
                    cards=[
                        frontend_pb2.Card(
                            title=f"Found: {title}",
                            subtitle=f"Last seen {when}",
                            description="Most recent matching clip in memory",
                            meta=[
                                frontend_pb2.KeyValue(key="Clip", value=f"#{clip_id}"),
                                frontend_pb2.KeyValue(key="Similarity", value=f"{score:.2f}"),
                            ],
                        )
                    ],
                    position="bottom",
                    duration_ms=8000,
                )
            )
        except grpc.RpcError as exc:
            log.warning("track_item.locate_card_failed", error=str(exc))

    def _play_clip(self, clip_id: int, *, title: str) -> bool:
        try:
            clip_bytes = self._load_clip_bytes(clip_id)
        except Exception as exc:
            log.warning("track_item.locate_clip_fetch_failed", clip_id=clip_id, error=str(exc))
            return False

        if not clip_bytes:
            log.warning("track_item.locate_clip_empty", clip_id=clip_id)
            return False
        if len(clip_bytes) > _MAX_INLINE_VIDEO_BYTES:
            log.warning(
                "track_item.locate_clip_too_large",
                clip_id=clip_id,
                bytes=len(clip_bytes),
                max_bytes=_MAX_INLINE_VIDEO_BYTES,
            )
            return False

        clip_b64 = base64.b64encode(clip_bytes).decode("ascii")
        html = (
            "<style>"
            "video::-webkit-media-controls-volume-slider,"
            "video::-webkit-media-controls-mute-button,"
            "video::-webkit-media-controls-volume-control-container,"
            "video::-webkit-media-controls-toggle-closed-captions-button,"
            "video::-webkit-media-controls-fullscreen-button,"
            "video::-webkit-media-controls-overflow-button,"
            "video::-webkit-media-controls-picture-in-picture-button"
            "{display:none !important;}"
            "</style>"
            "<div style='position:fixed;inset:0;display:flex;align-items:center;justify-content:center;"
            "background:rgba(0,0,0,0.65);z-index:9999;'>"
            "<div style='width:92vw;max-width:1100px;'>"
            f"<div style='color:#fff;font-size:18px;margin-bottom:10px;'>Last seen: {title}</div>"
            "<video autoplay playsinline controls style='width:100%;border-radius:14px;"
            "box-shadow:0 10px 28px rgba(0,0,0,0.45);'"
            f" src='data:video/mp4;base64,{clip_b64}'></video>"
            "</div>"
            "</div>"
        )
        self._frontend.RenderHtml(frontend_pb2.RenderHtmlRequest(html=html))
        log.info("track_item.locate_playback_rendered", clip_id=clip_id, bytes=len(clip_bytes))
        return True

    def _find_latest_occurrence_with_retry(self, request):
        response = self._data.FindLatestTrackedItemOccurrence(request)
        if not response.found_item or response.found_occurrence:
            return response

        if self._locate_wait_seconds <= 0.0:
            return response

        deadline = time.monotonic() + self._locate_wait_seconds
        attempts = 1
        while time.monotonic() < deadline:
            time.sleep(self._locate_poll_interval_seconds)
            response = self._data.FindLatestTrackedItemOccurrence(request)
            attempts += 1
            if response.found_occurrence or not response.found_item:
                log.info(
                    "track_item.locate_retry_result",
                    attempts=attempts,
                    found_item=response.found_item,
                    found_occurrence=response.found_occurrence,
                )
                return response
        log.info("track_item.locate_retry_timeout", attempts=attempts)
        return response

    def _load_clip_bytes(self, clip_id: int) -> bytes:
        stream = self._data.GetVideoClip(data_pb2.GetVideoClipRequest(clip_id=clip_id))
        clip_bytes = bytearray()
        expected_chunk_index = 0
        for chunk in stream:
            if chunk.chunk_index != expected_chunk_index:
                raise RuntimeError(
                    f"GetVideoClip chunk out of order for clip {clip_id}: "
                    f"expected {expected_chunk_index}, got {chunk.chunk_index}"
                )
            clip_bytes.extend(chunk.data)
            expected_chunk_index += 1
            if chunk.is_last:
                break
        return bytes(clip_bytes)


def _captured_frame_to_jpeg(frame: AssembledCameraFrame) -> bytes:
    if frame.encoding == sensor_pb2.FRAME_ENCODING_JPEG:
        return frame.data
    frame_rgb = assembled_frame_to_rgb(frame)
    return encode_rgb_to_jpeg(frame_rgb)


def _clean_item_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    if text.startswith("my "):
        text = text[3:].strip()
    return text


def _extract_item_title_from_query(query: str) -> str:
    lowered = _clean_item_title(query)
    if not lowered:
        return ""

    patterns = (
        r"^where did i leave (?:my )?(?P<item>.+)$",
        r"^where did i put (?:my )?(?P<item>.+)$",
        r"^where is (?:my )?(?P<item>.+)$",
        r"^find (?:my )?(?P<item>.+)$",
        r"^locate (?:my )?(?P<item>.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if match:
            return _clean_item_title(match.group("item"))
    return ""


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_relative_time(timestamp: float) -> str:
    if timestamp <= 0:
        return "at an unknown time"
    delta_s = max(0.0, time.time() - timestamp)
    if delta_s < 5:
        return "just now"
    if delta_s < 60:
        return f"{int(delta_s)} seconds ago"
    minutes = int(delta_s // 60)
    if minutes < 60:
        suffix = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {suffix} ago"
    hours = int(minutes // 60)
    suffix = "hour" if hours == 1 else "hours"
    return f"{hours} {suffix} ago"


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]

    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        log.warning("track_item.invalid_vlm_json", text=text[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


def serve_register(port: int = TRACK_ITEM_REGISTER_PORT) -> None:
    servicer = TrackItemRegisterServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


def serve_locate(port: int = TRACK_ITEM_LOCATE_PORT) -> None:
    servicer = TrackItemLocateServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve_register()
