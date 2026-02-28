"""Video capture pipeline: sensor camera stream -> MP4 clips -> DataService."""

from __future__ import annotations

import signal
import tempfile
import threading
import time
import uuid
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
from services.media.camera_transport import (
    CameraFrameAssembler,
    assembled_frame_to_rgb,
    downsample_mp4_for_embedding,
    encode_rgb_to_h264_annexb,
    h264_frames_to_mp4,
)

log = structlog.get_logger()

_MAX_GRPC_MESSAGE_BYTES = 32 * 1024 * 1024
_MEDIA_CHUNK_BYTES = 256 * 1024
_MP4_CODEC = "avc1"
_MIN_PARTIAL_CLIP_SECONDS = 2.0


def _open_mp4_writer(
    path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter | None:
    """Open an MP4 writer using the avc1 (H.264) codec."""
    fourcc = cv2.VideoWriter_fourcc(*_MP4_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer
    writer.release()
    return None


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
        upload_id = uuid.uuid4().hex

        def _request_iter():
            yield data_pb2.StoreVideoClipChunk(
                metadata=data_pb2.StoreVideoClipMetadata(
                    upload_id=upload_id,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    num_frames=num_frames,
                )
            )

            for chunk in _iter_media_chunks(mp4_data, media_id=f"{upload_id}:mp4"):
                yield data_pb2.StoreVideoClipChunk(mp4_chunk=chunk)

            if embed_data:
                for chunk in _iter_media_chunks(embed_data, media_id=f"{upload_id}:embed"):
                    yield data_pb2.StoreVideoClipChunk(embed_chunk=chunk)

        try:
            return self._stub.StoreVideoClip(_request_iter(), timeout=self._timeout)
        except grpc.RpcError as exc:
            log.warning("video.store_clip_failed", error=str(exc))
            return None


def _iter_media_chunks(data: bytes, media_id: str, chunk_bytes: int = _MEDIA_CHUNK_BYTES):
    if not data:
        yield data_pb2.MediaChunk(
            data=b"",
            media_id=media_id,
            chunk_index=0,
            is_last=True,
        )
        return

    total = len(data)
    chunk_index = 0
    for start in range(0, total, chunk_bytes):
        end = min(start + chunk_bytes, total)
        yield data_pb2.MediaChunk(
            data=data[start:end],
            media_id=media_id,
            chunk_index=chunk_index,
            is_last=end >= total,
        )
        chunk_index += 1


def _encode_frames_to_mp4(frames: list[np.ndarray], fps: float) -> bytes:
    """Encode a list of RGB uint8 frames to MP4 bytes using OpenCV."""
    if not frames:
        return b""

    height, width = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = _open_mp4_writer(tmp_path, fps, width, height)
        if writer is None:
            log.error("video.writer_open_failed", path=tmp_path, codec=_MP4_CODEC)
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
        self._stop_event = threading.Event()

    def _is_clip_window_complete(self, start_timestamp: float, end_timestamp: float) -> bool:
        if end_timestamp <= start_timestamp:
            return False

        # A clip with N frames spans roughly (N-1)/fps seconds, so allow one frame
        # period of slack before rolling to the next window.
        frame_period = 1.0 / max(1.0, float(self._clip_fps))
        min_duration = max(0.0, float(self._clip_duration) - frame_period)
        return (end_timestamp - start_timestamp) >= min_duration

    def _should_flush_partial_window(
        self,
        start_timestamp: float,
        end_timestamp: float,
        num_frames: int,
    ) -> bool:
        if num_frames < 2:
            return False

        elapsed = end_timestamp - start_timestamp
        if elapsed >= _MIN_PARTIAL_CLIP_SECONDS:
            return True

        # If timestamps are unavailable or non-monotonic, fall back to frame count.
        fallback_frames = int(max(2.0, float(self._clip_fps) * _MIN_PARTIAL_CLIP_SECONDS))
        return elapsed <= 0.0 and num_frames >= fallback_frames

    def _effective_clip_fps(
        self,
        start_timestamp: float,
        end_timestamp: float,
        num_frames: int,
    ) -> float:
        if num_frames <= 1:
            return float(self._clip_fps)

        elapsed = end_timestamp - start_timestamp
        if elapsed <= 0.0:
            return float(self._clip_fps)

        estimated = float(num_frames - 1) / elapsed
        return max(1.0, min(float(self._clip_fps), estimated))

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
        h264_frames: list[bytes] = []
        frame_count = 0
        start_timestamp = 0.0
        end_timestamp = 0.0

        try:
            for frame in self._iter_camera_frames(stream):
                if self._stop_event.is_set():
                    break

                # Start a new clip window
                if not h264_frames:
                    start_timestamp = frame.timestamp
                    frame_count = 0

                if frame.encoding == sensor_pb2.FRAME_ENCODING_H264:
                    h264_payload = frame.data
                else:
                    frame_rgb = assembled_frame_to_rgb(frame)
                    h264_payload = encode_rgb_to_h264_annexb(
                        frame_rgb,
                        fps=self._clip_fps,
                    )

                h264_frames.append(h264_payload)
                end_timestamp = frame.timestamp
                frame_count += 1

                # Clip window complete
                if self._is_clip_window_complete(start_timestamp, end_timestamp):
                    self._finalize_clip(
                        h264_frames,
                        start_timestamp, end_timestamp, frame_count,
                        data_client,
                    )
                    h264_frames = []
                    frame_count = 0
        finally:
            # Flush partial window if we have enough footage.
            if h264_frames and self._should_flush_partial_window(
                start_timestamp,
                end_timestamp,
                frame_count,
            ):
                self._finalize_clip(
                    h264_frames,
                    start_timestamp, end_timestamp, frame_count,
                    data_client,
                )

    @staticmethod
    def _iter_camera_frames(stream):
        assembler = CameraFrameAssembler()

        for chunk in stream:
            try:
                frame = assembler.push(chunk)
            except ValueError as exc:
                log.warning(
                    "video.frame_chunk_invalid",
                    error=str(exc),
                )
                continue

            if frame is not None:
                yield frame

    def _finalize_clip(
        self,
        h264_frames: list[bytes],
        start_timestamp: float,
        end_timestamp: float,
        num_frames: int,
        data_client: VideoDataClient,
    ) -> None:
        clip_fps = self._effective_clip_fps(start_timestamp, end_timestamp, num_frames)
        duration_s = max(0.0, end_timestamp - start_timestamp)

        try:
            mp4_data = h264_frames_to_mp4(h264_frames, clip_fps)
        except Exception as exc:
            log.error("video.h264_to_mp4_failed", error=str(exc))
            return

        if not mp4_data:
            log.warning("video.empty_clip", num_frames=num_frames)
            return

        try:
            embed_data = downsample_mp4_for_embedding(mp4_data, self._embed_fps)
        except Exception as exc:
            log.warning("video.embed_downsample_failed", error=str(exc))
            embed_data = b""

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
                duration_s=duration_s,
                clip_fps=clip_fps,
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
