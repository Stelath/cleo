"""Background thread that streams camera frames, buffers 10s chunks, and pushes MP4 clips to DataService."""

import tempfile
import threading
import time
from pathlib import Path

import cv2
import grpc
import numpy as np
import structlog

from core.config import DATA_ADDRESS, SENSOR_ADDRESS
from generated import data_pb2, data_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc

log = structlog.get_logger()

_BUFFER_SECONDS = 10
_TARGET_FPS = 2


class FrameProcessor(threading.Thread):
    """Daemon thread that streams camera frames, encodes 10s MP4 clips, and pushes them to DataService.

    Connects to SensorService.StreamCamera at 2 FPS, buffers frames for 10 seconds,
    encodes them into an MP4 clip, and sends the clip to DataService.StoreVideoClip
    for embedding + persistence.
    """

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        data_address: str = DATA_ADDRESS,
    ):
        super().__init__(daemon=True, name="FrameProcessor")
        self._sensor_address = sensor_address
        self._data_address = data_address
        self._fps = _TARGET_FPS
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        log.info("frame_processor.started", sensor=self._sensor_address, data=self._data_address)

        sensor_channel = grpc.insecure_channel(
            self._sensor_address,
            options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
        )
        data_channel = grpc.insecure_channel(
            self._data_address,
            options=[
                ("grpc.max_send_message_length", 32 * 1024 * 1024),
                ("grpc.max_receive_message_length", 32 * 1024 * 1024),
            ],
        )
        sensor_stub = sensor_pb2_grpc.SensorServiceStub(sensor_channel)
        self._data_stub = data_pb2_grpc.DataServiceStub(data_channel)
        request = sensor_pb2.StreamRequest(fps=_TARGET_FPS)

        while not self._stop_event.is_set():
            try:
                self._process_stream(sensor_stub, request)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    break
                log.error("frame_processor.rpc_error", error=str(e))
                time.sleep(2)

        sensor_channel.close()
        data_channel.close()
        log.info("frame_processor.stopped")

    def _process_stream(self, stub, request):
        """Stream frames, buffer 10s chunks, encode and push to DataService."""
        frame_buffer: list[tuple[np.ndarray, float]] = []
        chunk_start = time.monotonic()

        for frame_msg in stub.StreamCamera(request):
            if self._stop_event.is_set():
                return

            frame = np.frombuffer(frame_msg.data, dtype=np.uint8).reshape(
                frame_msg.height, frame_msg.width, 3
            )
            frame_buffer.append((frame, frame_msg.timestamp))

            elapsed = time.monotonic() - chunk_start
            if elapsed >= _BUFFER_SECONDS:
                self._process_chunk(frame_buffer)
                frame_buffer = []
                chunk_start = time.monotonic()

    def _process_chunk(self, frames: list[tuple[np.ndarray, float]]):
        """Encode buffered frames into MP4 and push to DataService."""
        if not frames:
            return

        start_ts = frames[0][1]
        end_ts = frames[-1][1]

        log.info(
            "frame_processor.chunk",
            num_frames=len(frames),
            start_ts=f"{start_ts:.2f}",
            end_ts=f"{end_ts:.2f}",
        )

        mp4_bytes = self._encode_clip(frames)
        if mp4_bytes is None:
            return

        try:
            req = data_pb2.StoreVideoClipRequest(
                mp4_data=mp4_bytes,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                num_frames=len(frames),
            )
            resp = self._data_stub.StoreVideoClip(req)
            log.info(
                "frame_processor.clip_stored",
                clip_id=resp.clip_id,
                faiss_id=resp.faiss_id,
            )
        except grpc.RpcError as e:
            log.error("frame_processor.store_error", error=str(e))

    def _encode_clip(self, frames: list[tuple[np.ndarray, float]]) -> bytes | None:
        """Encode buffered frames into an MP4 clip."""
        try:
            h, w = frames[0][0].shape[:2]
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp.name, fourcc, self._fps, (w, h))
                for frame, _ in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                return Path(tmp.name).read_bytes()
        except Exception as e:
            log.error("frame_processor.encode_error", error=str(e))
            return None
