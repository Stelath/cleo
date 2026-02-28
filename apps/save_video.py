"""Save Video tool service — clips the camera ring buffer and stores via DataService."""

from __future__ import annotations

import time
from typing import Any

import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import frontend_pb2, frontend_pb2_grpc, sensor_pb2, sensor_pb2_grpc
from services.config import (
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    SAVE_VIDEO_PORT,
    SENSOR_ADDRESS,
    SENSOR_CAMERA_BUFFER_SECONDS,
    SENSOR_DEFAULT_FPS,
)
from services.media.camera_transport import (
    CameraFrameAssembler,
    assembled_frame_to_rgb,
    encode_rgb_to_h264_annexb,
    h264_frames_to_mp4,
    downsample_mp4_for_embedding,
)
from services.video.service import VideoDataClient

log = structlog.get_logger()

_MAX_GRPC_MESSAGE_BYTES = 64 * 1024 * 1024


class SaveVideoServicer(ToolServiceBase):
    """Clips the last N seconds from the sensor camera ring buffer and stores the video."""

    @property
    def tool_name(self) -> str:
        return "save_video"

    @property
    def tool_description(self) -> str:
        return (
            "Save a video clip of what just happened. Captures the last ~30 seconds "
            "of camera footage from the ring buffer, encodes it as an MP4, generates "
            "a searchable embedding, and stores it persistently. Use when the user "
            'says things like "clip that", "save the video", or "what just happened".'
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
    ):
        self._sensor_channel = grpc.insecure_channel(
            sensor_address,
            options=[("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES)],
        )
        self._sensor = sensor_pb2_grpc.SensorServiceStub(self._sensor_channel)

        self._data_client = VideoDataClient(address=data_address)

        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)

    def close(self):
        self._data_client.close()
        self._sensor_channel.close()
        self._frontend_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        past_s = float(SENSOR_CAMERA_BUFFER_SECONDS)
        future_s = 60.0
        log.info("save_video.execute", past_seconds=past_s, future_seconds=future_s)

        # Notify user instantly
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title="Clipping Video...",
                    message=f"Capturing past {past_s:.0f}s and next {future_s:.0f}s...",
                    style="info",
                    duration_ms=4000,
                ),
                timeout=5,
            )
        except grpc.RpcError:
            pass

        import threading
        threading.Thread(
            target=self._capture_and_save,
            args=(past_s, future_s),
            daemon=True,
        ).start()

        return True, f"Started saving clip (past {past_s:.0f}s, future {future_s:.0f}s). Video will be ready when capture completes."

    def _capture_and_save(self, past_s: float, future_s: float) -> None:
        try:
            self._capture_and_save_impl(past_s, future_s)
        except Exception as exc:
            log.error("save_video.capture_failed", error=str(exc))
            try:
                self._frontend.ShowNotification(
                    frontend_pb2.NotificationRequest(
                        title="Video Save Failed",
                        message=f"Failed to save clip: {exc}",
                        style="error",
                        duration_ms=5000,
                    ),
                    timeout=5,
                )
            except grpc.RpcError:
                pass

    def _capture_and_save_impl(self, past_s: float, future_s: float) -> None:
        assembler = CameraFrameAssembler()
        h264_payloads = []
        timestamps = []

        # ── 1. Drain buffered frames from the sensor service ──
        if past_s > 0:
            try:
                request = sensor_pb2.GetBufferedFramesRequest(max_duration_seconds=past_s)
                stream = self._sensor.GetBufferedFrames(request, timeout=30)
                for chunk in stream:
                    frame = assembler.push(chunk)
                    if frame is not None:
                        if frame.encoding == sensor_pb2.FRAME_ENCODING_H264:
                            h264_payloads.append(frame.data)
                        else:
                            h264_payloads.append(
                                encode_rgb_to_h264_annexb(assembled_frame_to_rgb(frame), fps=SENSOR_DEFAULT_FPS)
                            )
                        timestamps.append(frame.timestamp)
            except grpc.RpcError as exc:
                log.warning("save_video.past_stream_error", error=str(exc))

        # ── 2. Capture future frames ──
        if future_s > 0:
            try:
                stream_req = sensor_pb2.StreamRequest(fps=SENSOR_DEFAULT_FPS)
                future_stream = self._sensor.StreamCamera(stream_req)
                start_future = time.time()

                for chunk in future_stream:
                    frame = assembler.push(chunk)
                    if frame is not None:
                        # Avoid duplicates exactly at the buffer boundary
                        if timestamps and frame.timestamp <= timestamps[-1]:
                            continue
                        
                        if frame.encoding == sensor_pb2.FRAME_ENCODING_H264:
                            h264_payloads.append(frame.data)
                        else:
                            h264_payloads.append(
                                encode_rgb_to_h264_annexb(assembled_frame_to_rgb(frame), fps=SENSOR_DEFAULT_FPS)
                            )
                        timestamps.append(frame.timestamp)

                        if time.time() - start_future >= future_s:
                            future_stream.cancel()
                            break
            except grpc.RpcError as exc:
                log.warning("save_video.future_stream_error", error=str(exc))

        if not h264_payloads:
            raise RuntimeError("No historical or future video frames available to save.")

        log.info("save_video.frames_received", count=len(h264_payloads))

        # ── 3. Combine encoded frames into an MP4 ──
        mp4_data = h264_frames_to_mp4(h264_payloads, fps=SENSOR_DEFAULT_FPS)
        if not mp4_data:
            raise RuntimeError("Failed to encode video clip to MP4.")

        # ── 4. Generate the embedding-friendly downsampled clip ──
        embed_data = downsample_mp4_for_embedding(mp4_data, target_fps=5.0)
        if not embed_data:
            embed_data = mp4_data  # fallback: use full clip

        # ── 5. Store via DataService (embedding + FAISS + SQLite + disk) ──
        start_ts = timestamps[0]
        end_ts = timestamps[-1]

        resp = self._data_client.store_clip(
            mp4_data=mp4_data,
            embed_data=embed_data,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            num_frames=len(h264_payloads),
        )

        if resp is None:
            raise RuntimeError("Failed to store video clip in DataService.")

        log.info(
            "save_video.stored",
            clip_id=resp.clip_id,
            faiss_id=resp.faiss_id,
            num_frames=len(h264_payloads),
        )

        # ── 6. Notify the user via the HUD ──
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title="Video saved",
                    message=f"Saved {len(h264_payloads)} frames ({end_ts - start_ts:.1f}s) as clip #{resp.clip_id}",
                    style="success",
                    duration_ms=4000,
                ),
                timeout=5,
            )
        except grpc.RpcError:
            pass


def serve(port: int = SAVE_VIDEO_PORT) -> None:
    serve_tool(SaveVideoServicer(), port=port)
