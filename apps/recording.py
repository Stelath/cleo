"""Recording tool: voice-controlled start/stop video recording sessions.

On "start", the service marks the recording start timestamp and activates the HUD
indicator. On "stop", it composes a new MP4 from stored clips that overlap the
recording window, stores that MP4 as a new clip, and stores recording metadata.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import (
    data_pb2,
    data_pb2_grpc,
    frontend_pb2,
    frontend_pb2_grpc,
)
from services.config import (
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    RECORDING_PORT,
    SENSOR_DEFAULT_FPS,
    VIDEO_EMBED_FPS,
)
from services.media.camera_transport import downsample_mp4_for_embedding
from services.video.service import VideoDataClient

log = structlog.get_logger()

_FFMPEG_TIMEOUT_SECONDS = 30.0
_CLIP_WAIT_TIMEOUT_SECONDS = 20.0
_CLIP_POLL_INTERVAL_SECONDS = 1.0
_STOP_COVERAGE_EPSILON_SECONDS = 0.25
_MAX_EMBED_SOURCE_SECONDS = 29.5


@dataclass(frozen=True)
class RecordingSegment:
    clip_id: int
    start_timestamp: float
    end_timestamp: float
    num_frames: int
    mp4_data: bytes


class RecordingServicer(ToolServiceBase):
    """Start or stop recording video from the camera."""

    @property
    def tool_name(self) -> str:
        return "recording"

    @property
    def tool_type(self) -> str:
        return "active"

    @property
    def tool_description(self) -> str:
        return (
            "Start or stop recording video from the camera. "
            "You MUST pass action='start' to begin recording or action='stop' to "
            "stop recording and save the clip. When the user says 'stop recording', "
            "'end recording', or 'finish recording', always pass action='stop'."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": "Use 'start' to begin recording or 'stop' to end and save.",
                },
            },
            "required": ["action"],
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
    ):
        self._data_address = data_address
        self._lock = threading.Lock()
        self._session_start: float | None = None

        self._data_client = VideoDataClient(address=data_address)

        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)

    def _set_indicator(self, active: bool) -> None:
        try:
            self._frontend.SetAppIndicator(
                frontend_pb2.AppIndicatorRequest(app_name="recording", is_active=active)
            )
        except grpc.RpcError as exc:
            log.warning("recording.indicator_error", error=str(exc))

    def _show_notification(self, title: str, message: str, style: str = "info") -> None:
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title=title,
                    message=message,
                    style=style,
                    duration_ms=4000,
                ),
                timeout=5,
            )
        except grpc.RpcError:
            pass

    def execute(self, params: dict) -> tuple[bool, str]:
        action = self._resolve_action(params)
        if action == "start":
            return self._start()
        if action == "stop":
            return self._stop()
        return False, "Recording action must be 'start' or 'stop'"

    def _resolve_action(self, params: dict) -> str | None:
        action = str(params.get("action", "")).strip().lower()
        if action in {"start", "stop"}:
            return action

        query = str(params.get("query", "")).strip().lower()
        if any(word in query for word in ("stop", "end", "pause", "cancel")):
            return "stop"
        if any(word in query for word in ("start", "begin", "resume", "record")):
            return "start"
        return None

    def _start(self) -> tuple[bool, str]:
        timestamp = time.time()
        with self._lock:
            if self._session_start is not None:
                return True, "Recording is already in progress"
            self._session_start = timestamp

        log.info("recording.started", started_at=timestamp)
        self._set_indicator(True)
        self._show_notification("Recording started", "Recording video from camera...")
        return True, "Recording started"

    def _stop(self) -> tuple[bool, str]:
        stop_timestamp = time.time()
        with self._lock:
            start_timestamp = self._session_start
            self._session_start = None

        if start_timestamp is None:
            return False, "No recording is in progress"

        self._set_indicator(False)

        threading.Thread(
            target=self._finalize_recording,
            args=(start_timestamp, stop_timestamp),
            daemon=True,
            name="RecordingFinalize",
        ).start()

        duration = max(0.0, stop_timestamp - start_timestamp)
        return True, f"Recording stopped ({duration:.1f}s). Saving..."

    def _finalize_recording(
        self, start_timestamp: float, stop_timestamp: float
    ) -> None:
        try:
            self._finalize_recording_impl(start_timestamp, stop_timestamp)
        except Exception as exc:
            log.error("recording.finalize_failed", error=str(exc))
            self._show_notification(
                "Recording save failed",
                f"Could not save recording: {exc}",
                style="error",
            )

    def _finalize_recording_impl(
        self, start_timestamp: float, stop_timestamp: float
    ) -> None:
        segments = self._load_segments_in_range(start_timestamp, stop_timestamp)
        if not segments:
            raise RuntimeError(
                "No video clips available for the requested recording window"
            )

        mp4_data = self._compose_recording_mp4(
            segments, start_timestamp, stop_timestamp
        )
        if not mp4_data:
            raise RuntimeError("Failed to compose recording MP4")

        duration = max(0.0, stop_timestamp - start_timestamp)
        embed_source = self._build_embed_source(mp4_data, duration)

        try:
            embed_data = downsample_mp4_for_embedding(
                embed_source, target_fps=VIDEO_EMBED_FPS
            )
        except Exception as exc:
            log.warning("recording.embed_downsample_failed", error=str(exc))
            embed_data = embed_source

        num_frames = max(1, int(round(duration * SENSOR_DEFAULT_FPS)))

        resp = self._data_client.store_clip(
            mp4_data=mp4_data,
            embed_data=embed_data,
            start_timestamp=start_timestamp,
            end_timestamp=stop_timestamp,
            num_frames=num_frames,
        )
        if resp is None:
            raise RuntimeError("Failed to store recording clip in DataService")

        self._store_recording_metadata(resp.clip_id, start_timestamp, stop_timestamp)
        log.info(
            "recording.stored",
            clip_id=resp.clip_id,
            faiss_id=resp.faiss_id,
            num_frames=num_frames,
            duration_s=duration,
            source_clip_count=len(segments),
        )

        self._show_notification(
            "Recording saved",
            (
                f"Saved {duration:.1f}s recording from {len(segments)} clip"
                f"{'s' if len(segments) != 1 else ''} as clip #{resp.clip_id}"
            ),
            style="success",
        )

    def _build_embed_source(self, mp4_data: bytes, duration_seconds: float) -> bytes:
        if duration_seconds <= _MAX_EMBED_SOURCE_SECONDS:
            return mp4_data

        log.info(
            "recording.embed_source_trimmed",
            original_duration_s=duration_seconds,
            embed_duration_s=_MAX_EMBED_SOURCE_SECONDS,
        )

        return self._trim_mp4_window(
            mp4_data,
            start_offset_seconds=0.0,
            duration_seconds=_MAX_EMBED_SOURCE_SECONDS,
        )

    def _store_recording_metadata(
        self,
        clip_id: int,
        started_at: float,
        ended_at: float,
    ) -> None:
        try:
            data_channel = grpc.insecure_channel(self._data_address)
            data_stub = data_pb2_grpc.DataServiceStub(data_channel)
            try:
                data_stub.StoreRecording(
                    data_pb2.StoreRecordingRequest(
                        clip_id=clip_id,
                        started_at=started_at,
                        ended_at=ended_at,
                    ),
                    timeout=10,
                )
            finally:
                data_channel.close()
        except grpc.RpcError as exc:
            log.warning("recording.store_metadata_failed", error=str(exc))

    def _load_segments_in_range(
        self,
        start_timestamp: float,
        stop_timestamp: float,
    ) -> list[RecordingSegment]:
        channel = grpc.insecure_channel(self._data_address)
        stub = data_pb2_grpc.DataServiceStub(channel)
        try:
            clip_metadatas = self._wait_for_overlapping_clips(
                stub,
                start_timestamp=start_timestamp,
                stop_timestamp=stop_timestamp,
            )
            segments: list[RecordingSegment] = []
            for clip in clip_metadatas:
                mp4_data = self._read_clip_bytes(stub, clip.clip_id)
                if not mp4_data:
                    continue
                segments.append(
                    RecordingSegment(
                        clip_id=clip.clip_id,
                        start_timestamp=clip.start_timestamp,
                        end_timestamp=clip.end_timestamp,
                        num_frames=clip.num_frames,
                        mp4_data=mp4_data,
                    )
                )
            return segments
        finally:
            channel.close()

    def _wait_for_overlapping_clips(
        self,
        stub: data_pb2_grpc.DataServiceStub,
        *,
        start_timestamp: float,
        stop_timestamp: float,
    ) -> list:
        deadline = time.time() + _CLIP_WAIT_TIMEOUT_SECONDS
        latest_clips = []

        while True:
            range_response = stub.GetVideoClipsInRange(
                data_pb2.TimeRangeRequest(
                    start_timestamp=start_timestamp,
                    end_timestamp=stop_timestamp,
                ),
                timeout=10,
            )
            latest_clips = list(range_response.clips)

            if latest_clips:
                latest_end = max(float(clip.end_timestamp) for clip in latest_clips)
                if latest_end >= (stop_timestamp - _STOP_COVERAGE_EPSILON_SECONDS):
                    return latest_clips

            if time.time() >= deadline:
                return latest_clips

            time.sleep(_CLIP_POLL_INTERVAL_SECONDS)

    @staticmethod
    def _read_clip_bytes(stub: data_pb2_grpc.DataServiceStub, clip_id: int) -> bytes:
        stream = stub.GetVideoClip(
            data_pb2.GetVideoClipRequest(clip_id=clip_id), timeout=30
        )
        clip_bytes = bytearray()
        expected_chunk_index = 0
        for chunk in stream:
            if chunk.chunk_index != expected_chunk_index:
                raise RuntimeError(
                    "GetVideoClip returned out-of-order chunks for "
                    f"clip_id={clip_id}: expected {expected_chunk_index}, got {chunk.chunk_index}"
                )
            clip_bytes.extend(chunk.data)
            expected_chunk_index += 1
            if chunk.is_last:
                break
        return bytes(clip_bytes)

    def _compose_recording_mp4(
        self,
        segments: list[RecordingSegment],
        start_timestamp: float,
        stop_timestamp: float,
    ) -> bytes:
        if not segments:
            return b""

        segment_bytes = [segment.mp4_data for segment in segments]
        combined_mp4 = self._concat_mp4_segments(segment_bytes)

        if not combined_mp4:
            raise RuntimeError("Combined recording MP4 was empty")

        combined_start = segments[0].start_timestamp
        combined_end = segments[-1].end_timestamp

        start_offset = max(0.0, start_timestamp - combined_start)
        desired_duration = max(0.0, stop_timestamp - start_timestamp)
        available_duration = max(0.0, combined_end - combined_start)
        trim_duration = min(
            desired_duration, max(0.0, available_duration - start_offset)
        )

        if trim_duration <= 0.05:
            raise RuntimeError("Recording duration is too short to save")

        needs_trim = start_offset > 0.02 or trim_duration < available_duration - 0.05
        if not needs_trim:
            return combined_mp4

        return self._trim_mp4_window(
            combined_mp4,
            start_offset_seconds=start_offset,
            duration_seconds=trim_duration,
        )

    def _concat_mp4_segments(self, segment_bytes: list[bytes]) -> bytes:
        if not segment_bytes:
            return b""
        if len(segment_bytes) == 1:
            return segment_bytes[0]

        with tempfile.TemporaryDirectory(prefix="cleo_recording_concat_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            segment_paths: list[Path] = []
            for index, data in enumerate(segment_bytes):
                segment_path = tmp_path / f"segment_{index:04d}.mp4"
                segment_path.write_bytes(data)
                segment_paths.append(segment_path)

            concat_list = tmp_path / "concat.txt"
            lines = []
            for path in segment_paths:
                escaped_path = str(path).replace("'", "'\\''")
                lines.append(f"file '{escaped_path}'")
            concat_list.write_text("\n".join(lines), encoding="utf-8")

            output_path = tmp_path / "combined.mp4"

            copy_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-an",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            copy_result = subprocess.run(
                copy_cmd,
                capture_output=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )

            if (
                copy_result.returncode == 0
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                return output_path.read_bytes()

            reencode_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            reencode_result = subprocess.run(
                reencode_cmd,
                capture_output=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
            if (
                reencode_result.returncode != 0
                or not output_path.exists()
                or output_path.stat().st_size == 0
            ):
                stderr = reencode_result.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    f"ffmpeg MP4 concat failed: {stderr or reencode_result.returncode}"
                )

            return output_path.read_bytes()

    def _trim_mp4_window(
        self,
        mp4_data: bytes,
        *,
        start_offset_seconds: float,
        duration_seconds: float,
    ) -> bytes:
        if not mp4_data:
            return b""

        with tempfile.TemporaryDirectory(prefix="cleo_recording_trim_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src = tmp_path / "input.mp4"
            dst = tmp_path / "trimmed.mp4"
            src.write_bytes(mp4_data)

            trim_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-ss",
                f"{max(0.0, float(start_offset_seconds)):.3f}",
                "-t",
                f"{max(0.05, float(duration_seconds)):.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ]
            result = subprocess.run(
                trim_cmd,
                capture_output=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
            if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
                stderr = result.stderr.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"ffmpeg trim failed: {stderr or result.returncode}")
            return dst.read_bytes()

    def close(self) -> None:
        self._data_client.close()
        self._frontend_channel.close()


def serve(port: int = RECORDING_PORT) -> None:
    servicer = RecordingServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve()
