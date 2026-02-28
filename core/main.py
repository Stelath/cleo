"""Core orchestrator: starts services, connects gRPC clients, runs processing threads."""

import multiprocessing
import signal
import threading
import time

import grpc
import structlog

from core.config import (
    ASSISTANT_ADDRESS,
    ASSISTANT_PORT,
    COLOR_BLIND_ADDRESS,
    COLOR_BLIND_PORT,
    DATA_ADDRESS,
    DATA_PORT,
    NAVIGATION_ASSIST_ADDRESS,
    NAVIGATION_ASSIST_PORT,
    OBJECT_RECOGNITION_ADDRESS,
    OBJECT_RECOGNITION_PORT,
    SENSOR_ADDRESS,
    SENSOR_PORT,
    TRANSCRIPTION_ADDRESS,
    TRANSCRIPTION_PORT,
)
from core.frame_processor import FrameProcessor
from generated import data_pb2, data_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc
from generated import transcription_pb2, transcription_pb2_grpc

log = structlog.get_logger()

_AUDIO_CHUNK_MS = 500
_AUDIO_SAMPLE_RATE = 16000


def _run_sensor_service():
    """Entry point for the sensor service subprocess."""
    from apps.sensor_service import serve
    serve(port=SENSOR_PORT)


def _run_transcription_service():
    """Entry point for the transcription service subprocess."""
    from transcription.parakeet import serve
    serve(port=TRANSCRIPTION_PORT)


def _run_data_service():
    """Entry point for the data service subprocess."""
    from data.service import serve
    serve(port=DATA_PORT)


def _run_assistant_service():
    """Entry point for the assistant service subprocess."""
    from assistant.service import serve
    serve(port=ASSISTANT_PORT)


def _run_color_blind_service():
    """Entry point for the color blindness tool subprocess."""
    from apps.color_blind import serve
    serve(port=COLOR_BLIND_PORT)


def _run_object_recognition_service():
    """Entry point for the object recognition tool subprocess."""
    from apps.object_recognition import serve
    serve(port=OBJECT_RECOGNITION_PORT)


def _run_navigation_assist_service():
    """Entry point for the navigation assist tool subprocess."""
    from apps.navigation_assist import serve
    serve(port=NAVIGATION_ASSIST_PORT)


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
    TranscriptionService.TranscribeStream. Transcription results are pushed to
    DataService.StoreTranscription for persistence.
    """

    def __init__(
        self,
        sensor_address: str = SENSOR_ADDRESS,
        transcription_address: str = TRANSCRIPTION_ADDRESS,
        data_address: str = DATA_ADDRESS,
    ):
        super().__init__(daemon=True, name="AudioTranscriptionBridge")
        self._sensor_address = sensor_address
        self._transcription_address = transcription_address
        self._data_address = data_address
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
        data_channel = grpc.insecure_channel(
            self._data_address,
            options=[("grpc.max_send_message_length", 8 * 1024 * 1024)],
        )
        sensor_stub = sensor_pb2_grpc.SensorServiceStub(sensor_channel)
        transcription_stub = transcription_pb2_grpc.TranscriptionServiceStub(
            transcription_channel
        )
        data_stub = data_pb2_grpc.DataServiceStub(data_channel)

        while not self._stop_event.is_set():
            try:
                self._bridge(sensor_stub, transcription_stub, data_stub)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    break
                log.error("audio_bridge.rpc_error", error=str(e))
                time.sleep(2)

        sensor_channel.close()
        transcription_channel.close()
        data_channel.close()
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

    def _bridge(self, sensor_stub, transcription_stub, data_stub):
        """Connect sensor audio stream to transcription stream, push results to DataService."""
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
                # Persist to DataService
                if not result.is_partial:
                    try:
                        data_stub.StoreTranscription(
                            data_pb2.StoreTranscriptionRequest(
                                text=result.text,
                                confidence=result.confidence,
                                start_time=result.start_time,
                                end_time=result.end_time,
                            )
                        )
                    except grpc.RpcError as e:
                        log.error("audio_bridge.store_error", error=str(e))


def main():
    """Start all services and processing threads, then wait for shutdown."""
    log.info("orchestrator.starting")

    # Start services as separate processes
    sensor_proc = multiprocessing.Process(target=_run_sensor_service, daemon=True)
    transcription_proc = multiprocessing.Process(target=_run_transcription_service, daemon=True)
    data_proc = multiprocessing.Process(target=_run_data_service, daemon=True)
    assistant_proc = multiprocessing.Process(target=_run_assistant_service, daemon=True)
    color_blind_proc = multiprocessing.Process(target=_run_color_blind_service, daemon=True)
    object_recognition_proc = multiprocessing.Process(target=_run_object_recognition_service, daemon=True)
    navigation_assist_proc = multiprocessing.Process(target=_run_navigation_assist_service, daemon=True)

    all_procs = [
        sensor_proc,
        transcription_proc,
        data_proc,
        assistant_proc,
        color_blind_proc,
        object_recognition_proc,
        navigation_assist_proc,
    ]

    for proc in all_procs:
        proc.start()
    log.info(
        "orchestrator.processes_started",
        sensor_pid=sensor_proc.pid,
        transcription_pid=transcription_proc.pid,
        data_pid=data_proc.pid,
        assistant_pid=assistant_proc.pid,
    )

    # Wait for gRPC servers to be ready
    _wait_for_grpc(SENSOR_ADDRESS)
    _wait_for_grpc(TRANSCRIPTION_ADDRESS)
    _wait_for_grpc(DATA_ADDRESS)
    _wait_for_grpc(ASSISTANT_ADDRESS)
    _wait_for_grpc(COLOR_BLIND_ADDRESS)
    _wait_for_grpc(OBJECT_RECOGNITION_ADDRESS)
    _wait_for_grpc(NAVIGATION_ASSIST_ADDRESS)

    # Start processing threads
    frame_processor = FrameProcessor(
        sensor_address=SENSOR_ADDRESS,
        data_address=DATA_ADDRESS,
    )
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

    # Terminate service processes (DataService handles its own FAISS persistence)
    log.info("orchestrator.stopping_processes")
    for proc in all_procs:
        proc.terminate()
    for proc in all_procs:
        proc.join(timeout=5)

    log.info("orchestrator.stopped")


if __name__ == "__main__":
    main()
