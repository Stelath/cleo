"""Sensor service with in-process fan-out and ring buffers."""

from collections import deque
from dataclasses import dataclass
import queue
import signal
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import structlog

from generated import sensor_pb2, sensor_pb2_grpc
from services.media.broadcast import BroadcastHub
from services.config import (
    SENSOR_CAMERA_CHUNK_BYTES,
    SENSOR_AUDIO_BUFFER_SECONDS,
    SENSOR_DEFAULT_CHUNK_MS,
    SENSOR_DEFAULT_FPS,
    SENSOR_H264_CRF,
    SENSOR_H264_PRESET,
    SENSOR_JPEG_QUALITY,
    SENSOR_DEFAULT_SAMPLE_RATE,
    SENSOR_PORT,
)
from services.media.camera_transport import (
    encode_rgb_to_jpeg,
    iter_camera_frame_chunks,
    PersistentH264Encoder,
)
from viture_sensors import AudioRecorder, USBCamera

log = structlog.get_logger()

_AUDIO_BUFFER_CHUNKS = max(
    4, int(SENSOR_AUDIO_BUFFER_SECONDS * 1000 / SENSOR_DEFAULT_CHUNK_MS)
)

@dataclass(frozen=True)
class _CameraPacket:
    frame_rgb: np.ndarray
    h264_data: bytes
    width: int
    height: int
    timestamp: float
    frame_id: str


class SensorServiceServicer(sensor_pb2_grpc.SensorServiceServicer):
    """gRPC servicer exposing buffered camera and audio streams."""

    def __init__(self):
        log.info("sensor_service.init", msg="Initializing hardware...")
        self._camera = USBCamera()
        self._camera.open()
        log.info(
            "sensor_service.camera_ready",
            device=self._camera.describe_active_device(),
        )

        self._recorder = AudioRecorder()
        log.info("sensor_service.audio_ready")

        self._camera_hub = BroadcastHub(maxsize=64)
        self._audio_hub = BroadcastHub(maxsize=256)
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._encoder_lock = threading.Lock()
        self._latest_frame: _CameraPacket | None = None
        self._frame_seq = 0
        self._h264_encoder: PersistentH264Encoder | None = None
        self._h264_encoder_shape: tuple[int, int] | None = None
        self._audio_buffer: deque[sensor_pb2.AudioChunk] = deque(
            maxlen=_AUDIO_BUFFER_CHUNKS
        )

        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop,
            daemon=True,
            name="CameraCaptureLoop",
        )
        self._audio_thread = threading.Thread(
            target=self._audio_capture_loop,
            daemon=True,
            name="AudioCaptureLoop",
        )
        self._camera_thread.start()
        self._audio_thread.start()

    def _camera_capture_loop(self) -> None:
        interval = 1.0 / SENSOR_DEFAULT_FPS
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                frame = self._camera.capture()
                h, w = frame.shape[:2]
                ts = time.time()
                h264_data = self._encode_stream_h264_frame(frame)
                with self._state_lock:
                    self._frame_seq += 1
                    frame_id = f"{int(ts * 1_000_000)}-{self._frame_seq}"
                msg = _CameraPacket(
                    frame_rgb=frame,
                    h264_data=h264_data,
                    width=w,
                    height=h,
                    timestamp=ts,
                    frame_id=frame_id,
                )
                with self._state_lock:
                    self._latest_frame = msg
                self._camera_hub.publish(msg)
            except RuntimeError as exc:
                log.error("sensor_service.camera_loop_error", error=str(exc))
                time.sleep(1)
                continue

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _audio_capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio = self._recorder.record(
                    duration_ms=SENSOR_DEFAULT_CHUNK_MS,
                    sample_rate=SENSOR_DEFAULT_SAMPLE_RATE,
                )
                msg = sensor_pb2.AudioChunk(
                    data=audio.tobytes(),
                    sample_rate=SENSOR_DEFAULT_SAMPLE_RATE,
                    num_samples=len(audio),
                    timestamp=time.time(),
                )
                with self._state_lock:
                    self._audio_buffer.append(msg)
                self._audio_hub.publish(msg)
            except RuntimeError as exc:
                log.error("sensor_service.audio_loop_error", error=str(exc))
                time.sleep(1)

    def shutdown(self) -> None:
        log.info("sensor_service.shutdown")
        self._stop_event.set()
        self._camera_thread.join(timeout=3)
        self._audio_thread.join(timeout=3)
        self._close_h264_encoder()
        self._camera.close()
        self._recorder.close()

    def _close_h264_encoder(self) -> None:
        with self._encoder_lock:
            encoder = self._h264_encoder
            self._h264_encoder = None
            self._h264_encoder_shape = None
        if encoder is not None:
            try:
                encoder.close()
            except Exception as exc:
                log.warning("sensor_service.h264_encoder_close_failed", error=str(exc))

    def _ensure_h264_encoder(self, width: int, height: int) -> PersistentH264Encoder:
        with self._encoder_lock:
            if (
                self._h264_encoder is not None
                and self._h264_encoder_shape == (width, height)
            ):
                return self._h264_encoder

            old_encoder = self._h264_encoder
            self._h264_encoder = None
            self._h264_encoder_shape = None
            if old_encoder is not None:
                old_encoder.close()

            self._h264_encoder = PersistentH264Encoder(
                width=width,
                height=height,
                fps=SENSOR_DEFAULT_FPS,
                crf=SENSOR_H264_CRF,
                preset=SENSOR_H264_PRESET,
            )
            self._h264_encoder_shape = (width, height)
            return self._h264_encoder

    def _encode_stream_h264_frame(self, frame_rgb: np.ndarray) -> bytes:
        height, width = frame_rgb.shape[:2]
        encoder = self._ensure_h264_encoder(width, height)
        try:
            return encoder.encode_frame(frame_rgb)
        except Exception as exc:
            log.warning("sensor_service.h264_encoder_retry", error=str(exc))
            self._close_h264_encoder()
            encoder = self._ensure_h264_encoder(width, height)
            return encoder.encode_frame(frame_rgb)

    def StreamCamera(self, request, context):
        fps = request.fps if request.fps > 0 else SENSOR_DEFAULT_FPS
        log.info("sensor_service.stream_camera", fps=fps)

        sid, q = self._camera_hub.subscribe()
        min_interval = 1.0 / max(0.1, float(fps))
        last_sent_ts = 0.0
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    frame_msg = q.get(timeout=1.0)
                    if frame_msg.timestamp - last_sent_ts < min_interval:
                        continue
                    last_sent_ts = frame_msg.timestamp
                    yield from self._iter_frame_chunks(
                        frame_id=frame_msg.frame_id,
                        data=frame_msg.h264_data,
                        width=frame_msg.width,
                        height=frame_msg.height,
                        timestamp=frame_msg.timestamp,
                        encoding=sensor_pb2.FRAME_ENCODING_H264,
                        key_frame=True,
                    )
                except queue.Empty:
                    continue
        finally:
            self._camera_hub.unsubscribe(sid)

    def StreamAudio(self, request, context):
        chunk_ms = (
            request.chunk_ms if request.chunk_ms > 0 else SENSOR_DEFAULT_CHUNK_MS
        )
        sample_rate = (
            request.sample_rate
            if request.sample_rate > 0
            else SENSOR_DEFAULT_SAMPLE_RATE
        )
        log.info(
            "sensor_service.stream_audio",
            chunk_ms=chunk_ms,
            sample_rate=sample_rate,
        )

        sid, q = self._audio_hub.subscribe()
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    yield q.get(timeout=1.0)
                except queue.Empty:
                    continue
        finally:
            self._audio_hub.unsubscribe(sid)

    def StreamIMU(self, request, context):
        del request
        log.warning("sensor_service.stream_imu", msg="IMU streaming not yet implemented")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("IMU streaming requires Device connection (not yet wired)")
        return
        yield

    def CaptureFrame(self, request, context):
        del request
        with self._state_lock:
            latest = self._latest_frame
        if latest is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("No camera frame available yet")
            return

        jpeg_data = encode_rgb_to_jpeg(latest.frame_rgb, quality=SENSOR_JPEG_QUALITY)
        yield from self._iter_frame_chunks(
            frame_id=latest.frame_id,
            data=jpeg_data,
            width=latest.width,
            height=latest.height,
            timestamp=latest.timestamp,
            encoding=sensor_pb2.FRAME_ENCODING_JPEG,
            key_frame=True,
        )

    @staticmethod
    def _iter_frame_chunks(
        *,
        frame_id: str,
        data: bytes,
        width: int,
        height: int,
        timestamp: float,
        encoding: int,
        key_frame: bool,
    ):
        yield from iter_camera_frame_chunks(
            data=data,
            frame_id=frame_id,
            width=width,
            height=height,
            timestamp=timestamp,
            encoding=encoding,
            key_frame=key_frame,
            chunk_bytes=SENSOR_CAMERA_CHUNK_BYTES,
        )

    @staticmethod
    def _chunk_to_array(chunk: sensor_pb2.AudioChunk) -> np.ndarray:
        return np.frombuffer(chunk.data, dtype=np.float32)

    def _recent_audio(self, duration_ms: int, sample_rate: int) -> np.ndarray:
        target_samples = max(1, int(sample_rate * duration_ms / 1000))

        with self._state_lock:
            chunks = list(self._audio_buffer)

        if not chunks:
            return np.zeros((0,), dtype=np.float32)

        pieces: list[np.ndarray] = []
        remaining = target_samples
        for chunk in reversed(chunks):
            audio = self._chunk_to_array(chunk)
            if len(audio) == 0:
                continue
            if len(audio) >= remaining:
                pieces.append(audio[-remaining:])
                remaining = 0
                break
            pieces.append(audio)
            remaining -= len(audio)

        if not pieces:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(list(reversed(pieces)))

    def RecordAudio(self, request, context):
        duration_ms = request.duration_ms if request.duration_ms > 0 else 1000
        sample_rate = (
            request.sample_rate
            if request.sample_rate > 0
            else SENSOR_DEFAULT_SAMPLE_RATE
        )

        if sample_rate != SENSOR_DEFAULT_SAMPLE_RATE:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Unsupported sample_rate={sample_rate}; sensor buffer is {SENSOR_DEFAULT_SAMPLE_RATE}"
            )
            return sensor_pb2.AudioChunk()

        audio = self._recent_audio(duration_ms=duration_ms, sample_rate=sample_rate)
        if len(audio) == 0:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("No audio available yet")
            return sensor_pb2.AudioChunk()

        return sensor_pb2.AudioChunk(
            data=audio.tobytes(),
            sample_rate=sample_rate,
            num_samples=len(audio),
            timestamp=time.time(),
        )


def serve(port: int = SENSOR_PORT) -> None:
    """Start the sensor gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        ],
    )
    servicer = SensorServiceServicer()
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("sensor_service.started", port=port)

    def _shutdown(signum, frame) -> None:
        del frame
        log.info("sensor_service.stopping", signal=signum)
        servicer.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
