"""Sensor service with in-process fan-out and ring buffers."""

from collections import deque
import queue
import signal
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import structlog

from generated import sensor_pb2, sensor_pb2_grpc
from services.broadcast import BroadcastHub
from services.config import (
    SENSOR_AUDIO_BUFFER_SECONDS,
    SENSOR_DEFAULT_CHUNK_MS,
    SENSOR_DEFAULT_FPS,
    SENSOR_DEFAULT_SAMPLE_RATE,
    SENSOR_PORT,
)
from viture_sensors import AudioRecorder, USBCamera

log = structlog.get_logger()

_AUDIO_BUFFER_CHUNKS = max(
    4, int(SENSOR_AUDIO_BUFFER_SECONDS * 1000 / SENSOR_DEFAULT_CHUNK_MS)
)


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
        self._latest_frame: sensor_pb2.CameraFrame | None = None
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
                msg = sensor_pb2.CameraFrame(
                    data=frame.tobytes(),
                    width=w,
                    height=h,
                    timestamp=time.time(),
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
        self._camera.close()
        self._recorder.close()

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
                    yield frame_msg
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
            return sensor_pb2.CameraFrame()

        return sensor_pb2.CameraFrame(
            data=latest.data,
            width=latest.width,
            height=latest.height,
            timestamp=latest.timestamp,
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
