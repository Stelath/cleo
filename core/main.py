"""Core orchestrator: starts services, connects gRPC clients, runs processing threads."""

from collections import deque
import math
import multiprocessing
import signal
import struct
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

_AUDIO_CHUNK_MS = 160
_AUDIO_SAMPLE_RATE = 16000
_SILENCE_THRESHOLD_RMS = 0.01
_FINAL_SILENCE_MS = 1000
_PREROLL_CHUNKS = 1


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
        self._partial_results: dict[str, transcription_pb2.TranscriptionResult] = {}

    def stop(self):
        self._stop_event.set()

    def _chunk_rms(self, chunk_data: bytes) -> float:
        """Estimate RMS for a float32 PCM chunk."""
        if not chunk_data:
            return 0.0

        sample_count = len(chunk_data) // 4
        if sample_count == 0:
            return 0.0

        total = 0.0
        for index in range(sample_count):
            sample = struct.unpack_from("<f", chunk_data, index * 4)[0]
            total += sample * sample
        return math.sqrt(total / sample_count)

    def _result_text(self, result) -> str:
        if result.text.strip():
            return result.text.strip()
        parts = [result.committed_text.strip(), result.unstable_text.strip()]
        return " ".join(part for part in parts if part).strip()

    def _store_completed_transcript(self, data_stub, result):
        if result.is_partial:
            if result.utterance_id:
                self._partial_results[result.utterance_id] = result
            return

        if result.utterance_id:
            self._partial_results.pop(result.utterance_id, None)

        text = result.committed_text.strip() or result.text.strip()
        if not text:
            return

        try:
            data_stub.StoreTranscription(
                data_pb2.StoreTranscriptionRequest(
                    text=text,
                    confidence=result.confidence,
                    start_time=result.start_time,
                    end_time=result.end_time,
                )
            )
        except grpc.RpcError as e:
            log.error("audio_bridge.store_error", error=str(e))

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
        silence_ms = 0.0
        speech_active = False
        pre_roll: deque[tuple[bytes, int]] = deque(maxlen=_PREROLL_CHUNKS)
        for chunk in sensor_stub.StreamAudio(request):
            if self._stop_event.is_set():
                if speech_active:
                    yield transcription_pb2.AudioInput(
                        audio_data=b"",
                        sample_rate=chunk.sample_rate,
                        is_final=True,
                        stream_id=self.name,
                    )
                return

            rms = self._chunk_rms(chunk.data)
            chunk_ms = (len(chunk.data) / 4) / chunk.sample_rate * 1000 if chunk.sample_rate > 0 else 0.0

            if rms >= _SILENCE_THRESHOLD_RMS:
                if not speech_active:
                    speech_active = True
                    for buffered_chunk, buffered_rate in pre_roll:
                        yield transcription_pb2.AudioInput(
                            audio_data=buffered_chunk,
                            sample_rate=buffered_rate,
                            is_final=False,
                            stream_id=self.name,
                        )
                    pre_roll.clear()
                speech_active = True
                silence_ms = 0.0
            elif speech_active:
                silence_ms += chunk_ms
            else:
                pre_roll.append((chunk.data, chunk.sample_rate))
                continue

            is_final = speech_active and silence_ms >= _FINAL_SILENCE_MS
            yield transcription_pb2.AudioInput(
                audio_data=chunk.data,
                sample_rate=chunk.sample_rate,
                is_final=is_final,
                stream_id=self.name,
            )
            if is_final:
                speech_active = False
                silence_ms = 0.0
                pre_roll.clear()

    def _bridge(self, sensor_stub, transcription_stub, data_stub):
        """Connect sensor audio stream to transcription stream, push results to DataService."""
        self._partial_results.clear()
        audio_stream = self._audio_generator(sensor_stub)
        results = transcription_stub.TranscribeStream(audio_stream)

        for result in results:
            text = self._result_text(result)
            if text:
                log.info(
                    "transcription.result",
                    utterance_id=result.utterance_id,
                    revision=result.revision,
                    text=text,
                    start=f"{result.start_time:.2f}",
                    end=f"{result.end_time:.2f}",
                    partial=result.is_partial,
                    stability=f"{result.stability:.2f}",
                )
            self._store_completed_transcript(data_stub, result)


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
