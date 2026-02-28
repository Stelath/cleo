"""Notetaking tool service."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import threading
import time
from typing import Any

import cv2
import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import data_pb2, data_pb2_grpc, frontend_pb2, frontend_pb2_grpc
from services.config import DATA_ADDRESS, FRONTEND_ADDRESS, NOTETAKING_PORT
from services.media.camera_transport import encode_rgb_to_jpeg

log = structlog.get_logger()

_MODEL_ID = os.environ.get(
    "CLEO_NOTETAKING_MODEL", "amazon.nova-pro-v1:0"
)
_BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
_MAX_GRPC_MESSAGE_BYTES = 32 * 1024 * 1024
_MAX_VIDEO_KEYFRAMES = 16


@dataclass
class NoteContext:
    """Captured context for a note session."""

    transcripts: list[data_pb2.TranscriptionLogEntry]
    clips: list[tuple[data_pb2.VideoClipMetadata, bytes]]


class NoteSummaryBedrockClient:
    """Bedrock client used only for multimodal note summaries."""

    def __init__(self, client: Any = None, model_id: str = _MODEL_ID, region: str = _BEDROCK_REGION):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def summarize(
        self,
        *,
        start_timestamp: float,
        end_timestamp: float,
        transcripts: list[data_pb2.TranscriptionLogEntry],
        clips: list[tuple[data_pb2.VideoClipMetadata, bytes]],
    ) -> str:
        """Generate a concise note summary from transcript text and video clips."""
        transcript_lines = [
            f"[{entry.start_time:.3f}-{entry.end_time:.3f}] {entry.text}".strip()
            for entry in transcripts
        ]
        prompt = (
            "Summarize what happened during this notetaking session. "
            "Focus on important actions, decisions, and observations. "
            "If the evidence is sparse, say so clearly.\n"
            f"Session start: {start_timestamp:.3f}\n"
            f"Session end: {end_timestamp:.3f}\n"
            f"Transcript count: {len(transcripts)}\n"
            f"Video clip count: {len(clips)}\n"
            "Transcripts:\n"
            f"{chr(10).join(transcript_lines) if transcript_lines else '(none)'}"
        )
        content = self._build_content(prompt=prompt, clips=clips)
        response = self._converse(content)
        text_parts = [
            block["text"]
            for block in response.get("output", {}).get("message", {}).get("content", [])
            if "text" in block
        ]
        return "\n".join(text_parts).strip()

    def _build_content(
        self,
        *,
        prompt: str,
        clips: list[tuple[data_pb2.VideoClipMetadata, bytes]],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"text": prompt}]
        for metadata, clip_mp4_data in clips:
            content.append(
                {
                    "text": (
                        f"Video clip {metadata.clip_id}: "
                        f"{metadata.start_timestamp:.3f}-{metadata.end_timestamp:.3f}, "
                        f"{metadata.num_frames} frames."
                    )
                }
            )
            for frame_jpeg in self._extract_keyframes(clip_mp4_data):
                content.append(
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": frame_jpeg},
                        }
                    }
                )
        return content

    def _extract_keyframes(
        self,
        clip_mp4_data: bytes,
        max_frames: int = _MAX_VIDEO_KEYFRAMES,
    ) -> list[bytes]:
        if not clip_mp4_data or max_frames <= 0:
            return []

        temp_path: str | None = None
        capture: cv2.VideoCapture | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(clip_mp4_data)
                temp_path = tmp.name

            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                raise RuntimeError("Failed to open temporary video clip for keyframe sampling")

            total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0)
            if total_frames <= 0:
                return []

            target_indices = self._sample_frame_indices(total_frames, max_frames)
            keyframes: list[bytes] = []
            for frame_index in target_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    log.warning(
                        "notetaking.keyframe_read_failed",
                        frame_index=frame_index,
                        total_frames=total_frames,
                    )
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                keyframes.append(encode_rgb_to_jpeg(frame_rgb))
            return keyframes
        except Exception as exc:
            log.warning("notetaking.keyframe_sampling_failed", error=str(exc))
            return []
        finally:
            if capture is not None:
                capture.release()
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _sample_frame_indices(total_frames: int, max_frames: int) -> list[int]:
        if total_frames <= 0 or max_frames <= 0:
            return []
        sample_count = min(total_frames, max_frames)
        if sample_count == 1:
            return [0]
        return sorted(
            {
                round(index * (total_frames - 1) / (sample_count - 1))
                for index in range(sample_count)
            }
        )

    def _converse(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        return self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": content}],
        )


class NotetakingServicer(ToolServiceBase):
    """Tracks a notetaking session and stores a generated summary on stop."""

    @property
    def tool_type(self) -> str:
        return "active"

    @property
    def tool_name(self) -> str:
        return "notetaking"

    @property
    def tool_description(self) -> str:
        return (
            "Start or stop a notetaking session. When stopped, summarize the "
            "captured transcript and video activity into a stored note."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": "Explicit notetaking action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional natural-language request, used when action is not provided."
                    ),
                },
            },
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
        bedrock_client: NoteSummaryBedrockClient | None = None,
    ):
        self._bedrock = bedrock_client or NoteSummaryBedrockClient()
        self._channel = grpc.insecure_channel(
            data_address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self._data = data_pb2_grpc.DataServiceStub(self._channel)
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)
        self._lock = threading.Lock()
        self._session_start: float | None = None

    def close(self) -> None:
        self._channel.close()
        self._frontend_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        action = self._resolve_action(params)
        if action == "start":
            return self._start_session()
        if action == "stop":
            return self._stop_session()
        return False, "Notetaking action must be 'start' or 'stop'"

    def _resolve_action(self, params: dict) -> str | None:
        action = str(params.get("action", "")).strip().lower()
        if action in {"start", "stop"}:
            return action

        query = str(params.get("query", "")).strip().lower()
        if any(word in query for word in ("start", "begin")):
            return "start"
        if any(word in query for word in ("stop", "end", "finish")):
            return "stop"
        return None

    def _start_session(self) -> tuple[bool, str]:
        timestamp = time.time()
        with self._lock:
            if self._session_start is not None:
                return True, f"Notetaking already active since {self._session_start:.3f}"
            self._session_start = timestamp
        log.info("notetaking.started", timestamp=timestamp)
        self._notify(
            title="Notetaking started",
            message="Capturing notes until you stop the session.",
            style="info",
        )
        return True, f"Started notetaking at {timestamp:.3f}"

    def _stop_session(self) -> tuple[bool, str]:
        stop_timestamp = time.time()
        with self._lock:
            start_timestamp = self._session_start
            self._session_start = None

        if start_timestamp is None:
            return False, "Notetaking is not active"

        log.info(
            "notetaking.stopped",
            start_timestamp=start_timestamp,
            stop_timestamp=stop_timestamp,
        )

        context = self._load_note_context(start_timestamp, stop_timestamp)
        summary_text = self._summarize_context(start_timestamp, stop_timestamp, context)
        stored = self._data.StoreNoteSummary(
            data_pb2.StoreNoteSummaryRequest(
                summary_text=summary_text,
                start_timestamp=start_timestamp,
                end_timestamp=stop_timestamp,
            )
        )
        log.info(
            "notetaking.summary_stored",
            note_id=stored.id,
            transcripts=len(context.transcripts),
            clips=len(context.clips),
        )
        self._notify(
            title="Note saved",
            message=f"Saved note summary #{stored.id}",
            style="success",
            note_id=stored.id,
        )
        return True, summary_text

    def _load_note_context(self, start_timestamp: float, stop_timestamp: float) -> NoteContext:
        transcript_response = self._data.GetTranscriptionsInRange(
            data_pb2.TimeRangeRequest(
                start_timestamp=start_timestamp,
                end_timestamp=stop_timestamp,
            )
        )
        clip_metadata_response = self._data.GetVideoClipsInRange(
            data_pb2.TimeRangeRequest(
                start_timestamp=start_timestamp,
                end_timestamp=stop_timestamp,
            )
        )

        clips = []
        for metadata in clip_metadata_response.clips:
            clip_stream = self._data.GetVideoClip(data_pb2.GetVideoClipRequest(clip_id=metadata.clip_id))
            clip_bytes = bytearray()
            expected_chunk_index = 0
            for chunk in clip_stream:
                if chunk.chunk_index != expected_chunk_index:
                    raise RuntimeError(
                        "GetVideoClip returned out-of-order chunks for "
                        f"clip_id={metadata.clip_id}: expected {expected_chunk_index}, got {chunk.chunk_index}"
                    )
                clip_bytes.extend(chunk.data)
                expected_chunk_index += 1
                if chunk.is_last:
                    break
            clips.append((metadata, bytes(clip_bytes)))

        return NoteContext(
            transcripts=list(transcript_response.entries),
            clips=clips,
        )

    def _summarize_context(
        self, start_timestamp: float, stop_timestamp: float, context: NoteContext
    ) -> str:
        if not context.transcripts and not context.clips:
            return "No transcript or video activity was captured during this note session."

        try:
            summary_text = self._bedrock.summarize(
                start_timestamp=start_timestamp,
                end_timestamp=stop_timestamp,
                transcripts=context.transcripts,
                clips=context.clips,
            )
        except Exception as exc:
            log.error("notetaking.summary_failed", error=str(exc))
            summary_text = ""

        if summary_text:
            return summary_text

        return "A note summary could not be generated for this session."

    def _notify(
        self,
        *,
        title: str,
        message: str,
        style: str,
        note_id: int | None = None,
    ) -> None:
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title=title,
                    message=message,
                    style=style,
                )
            )
        except Exception as exc:
            log.warning(
                "notetaking.notification_failed",
                note_id=note_id,
                title=title,
                error=str(exc),
            )


def serve(port: int = NOTETAKING_PORT):
    servicer = NotetakingServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve()
