"""Core orchestrator: starts services, connects gRPC clients, runs processing threads."""

import multiprocessing
import signal
import sys
import threading
import time

import grpc
import numpy as np
import structlog

from core.frame_processor import FrameProcessor
from generated import sensor_pb2, sensor_pb2_grpc
from generated import transcription_pb2, transcription_pb2_grpc

log = structlog.get_logger()

_SENSOR_PORT = 50051
_TRANSCRIPTION_PORT = 50052
_SENSOR_ADDR = f"localhost:{_SENSOR_PORT}"
_TRANSCRIPTION_ADDR = f"localhost:{_TRANSCRIPTION_PORT}"
_AUDIO_CHUNK_MS = 500
_AUDIO_SAMPLE_RATE = 16000


def _run_sensor_service():
    """Entry point for the sensor service subprocess."""
    from services.sensor_service import serve
    serve(port=_SENSOR_PORT)


def _run_transcription_service():
    """Entry point for the transcription service subprocess."""
    from transcription.parakeet import serve
    serve(port=_TRANSCRIPTION_PORT)


def _wait_for_grpc(address: str, timeout: float = 30.0):
    """Block until a gRPC server is accepting connections."""
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        log.info("orchestrator.grpc_ready", address=address)
    except grpc.FutureTimeoutError:
        log.error("orchestrator.grpc_timeout", address=address, timeout=timeout)
        raise RuntimeError(f"gRPC server at {address} did not become ready in {timeout}s")
    finally:
        channel.close()


class AudioTranscriptionBridge(threading.Thread):
    """Daemon thread that streams audio from SensorService to TranscriptionService.

    Reads audio chunks from SensorService.StreamAudio and forwards them to
    TranscriptionService.TranscribeStream, logging transcription results.
    """

    def __init__(
        self,
        sensor_address: str = _SENSOR_ADDR,
        transcription_address: str = _TRANSCRIPTION_ADDR,
    ):
        super().__init__(daemon=True, name="AudioTranscriptionBridge")
        self._sensor_address = sensor_address
        self._transcription_address = transcription_address
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        log.info("audio_bridge.started")

        sensor_channel = grpc.insecure_channel(
            self._sensor_address,
            options=[("grpc.max_receive_message_length", 8 * 1024 * 1024)],
        )
        transcription_channel = grpc.insecure_channel(
            self._transcription_address,
            options=[
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ],
        )
        sensor_stub = sensor_pb2_grpc.SensorServiceStub(sensor_channel)
        transcription_stub = transcription_pb2_grpc.TranscriptionServiceStub(
            transcription_channel
        )

        while not self._stop_event.is_set():
            try:
                self._bridge(sensor_stub, transcription_stub)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    break
                log.error("audio_bridge.rpc_error", error=str(e))
                time.sleep(2)

        sensor_channel.close()
        transcription_channel.close()
        log.info("audio_bridge.stopped")

    def _audio_generator(self, sensor_stub):
        """Yield AudioInput messages from the sensor audio stream."""
        request = sensor_pb2.StreamRequest(
            chunk_ms=_AUDIO_CHUNK_MS, sample_rate=_AUDIO_SAMPLE_RATE
        )
        for chunk in sensor_stub.StreamAudio(request):
            if self._stop_event.is_set():
                return
            yield transcription_pb2.AudioInput(
                audio_data=chunk.data,
                sample_rate=chunk.sample_rate,
                is_final=False,
            )

    def _bridge(self, sensor_stub, transcription_stub):
        """Connect sensor audio stream to transcription stream."""
        audio_stream = self._audio_generator(sensor_stub)
        results = transcription_stub.TranscribeStream(audio_stream)

        for result in results:
            if self._stop_event.is_set():
                return
            if result.text.strip():
                log.info(
                    "transcription.result",
                    text=result.text,
                    start=f"{result.start_time:.2f}",
                    end=f"{result.end_time:.2f}",
                    partial=result.is_partial,
                )


def main():
    """Start all services and processing threads, then wait for shutdown."""
    log.info("orchestrator.starting")

    # Start services as separate processes
    sensor_proc = multiprocessing.Process(target=_run_sensor_service, daemon=True)
    transcription_proc = multiprocessing.Process(target=_run_transcription_service, daemon=True)

    sensor_proc.start()
    transcription_proc.start()
    log.info("orchestrator.processes_started",
             sensor_pid=sensor_proc.pid,
             transcription_pid=transcription_proc.pid)

    # Wait for gRPC servers to be ready
    _wait_for_grpc(_SENSOR_ADDR)
    _wait_for_grpc(_TRANSCRIPTION_ADDR)

    # Start processing threads
    frame_processor = FrameProcessor(sensor_address=_SENSOR_ADDR)
    audio_bridge = AudioTranscriptionBridge()

    frame_processor.start()
    audio_bridge.start()
    log.info("orchestrator.threads_started")

    # Graceful shutdown handler
    shutdown_event = threading.Event()

    def _shutdown(signum, frame):
        log.info("orchestrator.shutdown", signal=signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for shutdown signal
    shutdown_event.wait()

    # Stop threads
    log.info("orchestrator.stopping_threads")
    frame_processor.stop()
    audio_bridge.stop()
    frame_processor.join(timeout=5)
    audio_bridge.join(timeout=5)

    # Save FAISS index
    try:
        frame_processor.db.save()
        log.info("orchestrator.faiss_saved")
    except Exception as e:
        log.error("orchestrator.faiss_save_error", error=str(e))

    # Terminate service processes
    log.info("orchestrator.stopping_processes")
    sensor_proc.terminate()
    transcription_proc.terminate()
    sensor_proc.join(timeout=5)
    transcription_proc.join(timeout=5)

    log.info("orchestrator.stopped")


if __name__ == "__main__":
    main()
