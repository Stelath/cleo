"""Save Video tool service — bookmarks a time window for user-initiated clip saves.

Instead of capturing and encoding video directly, this tool records a
(start_ts, end_ts) bookmark in the DataService's ``saved_clips`` table.
The continuous VideoClipPipeline already stores all footage as 15-second
clips — the bookmark timestamps let downstream consumers pull the right
clips via ``GetVideoClipsInRange``.
"""

from __future__ import annotations

import time
from typing import Any

import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import data_pb2, data_pb2_grpc, frontend_pb2, frontend_pb2_grpc
from services.config import (
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    SAVE_VIDEO_PORT,
    SENSOR_CAMERA_BUFFER_SECONDS,
)

log = structlog.get_logger()

_FORWARD_SECONDS = 60.0


class SaveVideoServicer(ToolServiceBase):
    """Bookmarks a time window so continuous pipeline clips can be recalled later."""

    @property
    def tool_name(self) -> str:
        return "save_video"

    @property
    def tool_description(self) -> str:
        return (
            "Save a video clip of what just happened and what is currently happening. "
            "Bookmarks the past ~30 seconds and next ~60 seconds of camera footage "
            "so the clips can be retrieved later. Use when the user says things like "
            '"clip that", "save the video", "record what just happened", or similar.'
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    def __init__(
        self,
        data_address: str = DATA_ADDRESS,
        frontend_address: str = FRONTEND_ADDRESS,
    ):
        self._data_channel = grpc.insecure_channel(data_address)
        self._data = data_pb2_grpc.DataServiceStub(self._data_channel)

        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)

    def close(self) -> None:
        self._data_channel.close()
        self._frontend_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        now = time.time()
        start_ts = now - float(SENSOR_CAMERA_BUFFER_SECONDS)
        end_ts = now + _FORWARD_SECONDS

        log.info(
            "save_video.execute",
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )

        try:
            resp = self._data.StoreSavedClip(
                data_pb2.StoreSavedClipRequest(
                    label="user clip",
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                ),
                timeout=5.0,
            )
        except grpc.RpcError as exc:
            log.error("save_video.store_failed", error=str(exc))
            return False, f"Failed to save clip bookmark: {exc}"

        duration = end_ts - start_ts
        log.info("save_video.stored", bookmark_id=resp.id, duration=duration)

        self._notify(
            title="Video saved",
            message=(
                f"Bookmarked {duration:.0f}s of footage "
                f"(clip #{resp.id}). Past footage is available now; "
                f"future footage will be ready in ~{_FORWARD_SECONDS:.0f}s."
            ),
            style="success",
        )

        return True, (
            f"Saved clip bookmark #{resp.id} covering {duration:.0f}s "
            f"({SENSOR_CAMERA_BUFFER_SECONDS}s past + {_FORWARD_SECONDS:.0f}s future)."
        )

    def _notify(self, *, title: str, message: str, style: str) -> None:
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
                "save_video.notification_failed",
                title=title,
                error=str(exc),
            )


def serve(port: int = SAVE_VIDEO_PORT) -> None:
    servicer = SaveVideoServicer()
    try:
        serve_tool(servicer, port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve()
