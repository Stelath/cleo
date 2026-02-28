"""Video capture pipeline: sensor camera stream -> MP4 clips -> DataService."""

from __future__ import annotations

import signal
import tempfile
import threading
import time
from pathlib import Path

import cv2
import grpc
import numpy as np
import structlog

from generated import data_pb2, data_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc
from services.config import (
    DATA_ADDRESS,
    SENSOR_ADDRESS,
    VIDEO_CLIP_DURATION_SECONDS,
    VIDEO_CLIP_FPS,
    VIDEO_EMBED_FPS,
)

log = structlog.get_logger()

_MAX_GRPC_MESSAGE_BYTES = 32 * 1024 * 1024


class VideoDataClient:
    """gRPC client for DataService.StoreVideoClip."""

    def __init__(self, address: str = DATA_ADDRESS, timeout: float = 30.0):
        self._timeout = timeout
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self._stub = data_pb2_grpc.DataServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def store_clip(
        self,
        mp4_data: bytes,
        embed_data: bytes,
        start_timestamp: float,
        end_timestamp: float,
        num_frames: int,
    ) -> data_pb2.StoreVideoClipResponse | None:
        try:
            return self._stub.StoreVideoClip(
                data_pb2.StoreVideoClipRequest(
                    mp4_data=mp4_data,
                    embed_data=embed_data,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    num_frames=num_frames,
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("video.store_clip_failed", error=str(exc))
            return None


def _encode_frames_to_mp4(frames: list[np.ndarray], fps: float) -> bytes:
    """Encode a list of RGB uint8 frames to MP4 bytes using OpenCV."""
    if not frames:
        return b""

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            log.error("video.writer_open_failed", path=tmp_path)
            return b""

        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

        return Path(tmp_path).read_bytes()
    except Exception as exc:
        log.error("video.encode_failed", error=str(exc))
        return b""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class VideoClipPipeline(threading.Thread):
    """Daemon thread: subscribes to sensor camera, produces 15s MP4 clips."""

    def __init__(
        self,
        *,
        sensor_address: str = SENSOR_ADDRESS,
        data_address: str = DATA_ADDRESS,
        clip_fps: float = VIDEO_CLIP_FPS,
        embed_fps: float = VIDEO_EMBED_FPS,
        clip_duration: float = VIDEO_CLIP_DURATION_SECONDS,
    ):
        super().__init__(daemon=True, name="VideoClipPipeline")
        self._sensor_address = sensor_address
        self._data_address = data_address
        self._clip_fps = clip_fps
        self._embed_fps = embed_fps
        self._clip_duration = clip_duration
        self._target_frames = int(clip_fps * clip_duration)
        # Keep every Nth frame for embed stream (30/5 = every 6th)
        self._embed_every_n = max(1, int(clip_fps / embed_fps))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        data_client = VideoDataClient(address=self._data_address)
        sensor_channel = grpc.insecure_channel(
            self._sensor_address,
            options=[("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES)],
        )
        sensor_stub = sensor_pb2_grpc.SensorServiceStub(sensor_channel)

        try:
            while not self._stop_event.is_set():
                try:
                    log.info("video.pipeline_connecting")
                    self._stream_loop(sensor_stub, data_client)
                except grpc.RpcError as exc:
                    if self._stop_event.is_set():
                        break
                    log.warning("video.sensor_stream_error", error=str(exc))
                    time.sleep(2)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    log.warning("video.pipeline_error", error=str(exc))
                    time.sleep(2)
        finally:
            data_client.close()
            sensor_channel.close()

    def _stream_loop(
        self,
        sensor_stub: sensor_pb2_grpc.SensorServiceStub,
        data_client: VideoDataClient,
    ) -> None:
        stream_request = sensor_pb2.StreamRequest(fps=self._clip_fps)
        stream = sensor_stub.StreamCamera(stream_request)

        # State for current clip window
        writer = None
        tmp_path = None
        embed_frames: list[np.ndarray] = []
        frame_count = 0
        start_timestamp = 0.0
        end_timestamp = 0.0
        width = 0
        height = 0

        try:
            for camera_frame in stream:
                if self._stop_event.is_set():
                    break

                frame_np = np.frombuffer(camera_frame.data, dtype=np.uint8).reshape(
                    camera_frame.height, camera_frame.width, 3
                )

                # Start a new clip window
                if writer is None:
                    width = camera_frame.width
                    height = camera_frame.height
                    start_timestamp = camera_frame.timestamp
                    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(tmp_path, fourcc, self._clip_fps, (width, height))
                    if not writer.isOpened():
                        log.error("video.writer_open_failed", path=tmp_path)
                        Path(tmp_path).unlink(missing_ok=True)
                        writer = None
                        tmp_path = None
                        continue
                    embed_frames = []
                    frame_count = 0

                # Write frame to full-rate MP4 (RGB -> BGR)
                bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                end_timestamp = camera_frame.timestamp
                frame_count += 1

                # Keep every Nth frame for embed stream
                if (frame_count - 1) % self._embed_every_n == 0:
                    embed_frames.append(frame_np.copy())

                # Clip window complete
                if frame_count >= self._target_frames:
                    self._finalize_clip(
                        writer, tmp_path, embed_frames,
                        start_timestamp, end_timestamp, frame_count,
                        data_client,
                    )
                    writer = None
                    tmp_path = None
                    embed_frames = []
                    frame_count = 0
        finally:
            # Flush partial window if we have >= 2 seconds of footage
            min_frames = int(self._clip_fps * 2)
            if writer is not None and frame_count >= min_frames:
                self._finalize_clip(
                    writer, tmp_path, embed_frames,
                    start_timestamp, end_timestamp, frame_count,
                    data_client,
                )
            elif writer is not None:
                writer.release()
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

    def _finalize_clip(
        self,
        writer: cv2.VideoWriter,
        tmp_path: str | None,
        embed_frames: list[np.ndarray],
        start_timestamp: float,
        end_timestamp: float,
        num_frames: int,
        data_client: VideoDataClient,
    ) -> None:
        writer.release()

        mp4_data = b""
        if tmp_path:
            try:
                mp4_data = Path(tmp_path).read_bytes()
            except Exception as exc:
                log.error("video.read_clip_failed", error=str(exc))
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        if not mp4_data:
            log.warning("video.empty_clip", num_frames=num_frames)
            return

        embed_data = _encode_frames_to_mp4(embed_frames, self._embed_fps)

        resp = data_client.store_clip(
            mp4_data=mp4_data,
            embed_data=embed_data,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            num_frames=num_frames,
        )
        if resp is not None:
            log.info(
                "video.clip_stored",
                clip_id=resp.clip_id,
                faiss_id=resp.faiss_id,
                num_frames=num_frames,
            )


def serve() -> None:
    """Start the video capture pipeline (no gRPC server)."""
    log.info("video.starting")
    pipeline = VideoClipPipeline()
    pipeline.start()

    shutdown_event = threading.Event()

    def _shutdown(signum, frame) -> None:
        del frame
        log.info("video.stopping", signal=signum)
        pipeline.stop()
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    shutdown_event.wait()

    pipeline.join(timeout=10)
    log.info("video.stopped")


if __name__ == "__main__":
    serve()
