"""gRPC Sensor Service wrapping VITURE Luma Ultra camera and microphone hardware."""

import queue
import signal
import sys
import threading
import time
from concurrent import futures

import grpc
import numpy as np
import structlog

from core.broadcast import BroadcastHub
from generated import sensor_pb2, sensor_pb2_grpc
from viture_sensors import AudioRecorder, USBCamera

log = structlog.get_logger()

_DEFAULT_FPS = 2.0
_DEFAULT_CHUNK_MS = 500
_DEFAULT_SAMPLE_RATE = 16000


class SensorServiceServicer(sensor_pb2_grpc.SensorServiceServicer):
    """gRPC servicer that exposes VITURE hardware sensors over the network."""

    def __init__(self):
        log.info("sensor_service.init", msg="Initializing hardware...")
        self._camera = USBCamera()
        self._camera.open()
        log.info("sensor_service.camera_ready", device=self._camera.describe_active_device())

        self._recorder = AudioRecorder()
        log.info("sensor_service.audio_ready")

        # Broadcast hubs for fan-out streaming
        self._camera_hub = BroadcastHub(maxsize=64)
        self._audio_hub = BroadcastHub(maxsize=128)
        self._stop_event = threading.Event()

        # Start background capture loops
        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop, daemon=True, name="CameraCaptureLoop"
        )
        self._audio_thread = threading.Thread(
            target=self._audio_capture_loop, daemon=True, name="AudioCaptureLoop"
        )
        self._camera_thread.start()
        self._audio_thread.start()

    def _camera_capture_loop(self):
        """Continuously capture frames and publish to the camera hub."""
        interval = 1.0 / _DEFAULT_FPS
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
                self._camera_hub.publish(msg)
            except RuntimeError as e:
                log.error("sensor_service.camera_loop_error", error=str(e))
                time.sleep(1)
                continue

            elapsed = time.monotonic() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _audio_capture_loop(self):
        """Continuously record audio chunks and publish to the audio hub."""
        while not self._stop_event.is_set():
            try:
                audio = self._recorder.record(
                    duration_ms=_DEFAULT_CHUNK_MS, sample_rate=_DEFAULT_SAMPLE_RATE
                )
                msg = sensor_pb2.AudioChunk(
                    data=audio.tobytes(),
                    sample_rate=_DEFAULT_SAMPLE_RATE,
                    num_samples=len(audio),
                    timestamp=time.time(),
                )
                self._audio_hub.publish(msg)
            except RuntimeError as e:
                log.error("sensor_service.audio_loop_error", error=str(e))
                time.sleep(1)

    def shutdown(self):
        """Gracefully release hardware resources."""
        log.info("sensor_service.shutdown")
        self._stop_event.set()
        self._camera_thread.join(timeout=3)
        self._audio_thread.join(timeout=3)
        self._camera.close()
        self._recorder.close()

    def StreamCamera(self, request, context):
        """Stream camera frames at the requested FPS via BroadcastHub."""
        fps = request.fps if request.fps > 0 else _DEFAULT_FPS
        log.info("sensor_service.stream_camera", fps=fps)

        sid, q = self._camera_hub.subscribe()
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    frame_msg = q.get(timeout=1.0)
                    yield frame_msg
                except queue.Empty:
                    continue
        finally:
            self._camera_hub.unsubscribe(sid)

    def StreamAudio(self, request, context):
        """Stream audio chunks via BroadcastHub."""
        chunk_ms = request.chunk_ms if request.chunk_ms > 0 else _DEFAULT_CHUNK_MS
        sample_rate = request.sample_rate if request.sample_rate > 0 else _DEFAULT_SAMPLE_RATE
        log.info("sensor_service.stream_audio", chunk_ms=chunk_ms, sample_rate=sample_rate)

        sid, q = self._audio_hub.subscribe()
        try:
            while context.is_active() and not self._stop_event.is_set():
                try:
                    audio_msg = q.get(timeout=1.0)
                    yield audio_msg
                except queue.Empty:
                    continue
        finally:
            self._audio_hub.unsubscribe(sid)

    def StreamIMU(self, request, context):
        """Stream IMU readings. Requires a connected VITURE Device."""
        log.warning("sensor_service.stream_imu", msg="IMU streaming not yet implemented")
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("IMU streaming requires Device connection (not yet wired)")
        return
        yield  # make this a generator

    def CaptureFrame(self, request, context):
        """Capture and return a single camera frame."""
        try:
            frame = self._camera.capture()
            h, w = frame.shape[:2]
            return sensor_pb2.CameraFrame(
                data=frame.tobytes(),
                width=w,
                height=h,
                timestamp=time.time(),
            )
        except RuntimeError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sensor_pb2.CameraFrame()

    def RecordAudio(self, request, context):
        """Record a fixed-duration audio clip."""
        duration_ms = request.duration_ms if request.duration_ms > 0 else 1000
        sample_rate = request.sample_rate if request.sample_rate > 0 else _DEFAULT_SAMPLE_RATE

        try:
            audio = self._recorder.record(
                duration_ms=duration_ms, sample_rate=sample_rate
            )
            return sensor_pb2.AudioChunk(
                data=audio.tobytes(),
                sample_rate=sample_rate,
                num_samples=len(audio),
                timestamp=time.time(),
            )
        except RuntimeError as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sensor_pb2.AudioChunk()


def serve(port: int = 50051):
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

    def _shutdown(signum, frame):
        log.info("sensor_service.stopping", signal=signum)
        servicer.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
