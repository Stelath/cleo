"""Streaming transcription service with persistent sensor ingestion."""

from __future__ import annotations

from collections import deque
import math
import signal
import tempfile
import threading
import time
from concurrent import futures
from dataclasses import dataclass

import grpc
import numpy as np
import soundfile as sf
import structlog

from generated import assistant_pb2, assistant_pb2_grpc
from generated import data_pb2, data_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc
from generated import transcription_pb2, transcription_pb2_grpc
from services.config import (
    ASSISTANT_ADDRESS,
    DATA_ADDRESS,
    SENSOR_ADDRESS,
    SENSOR_DEFAULT_CHUNK_MS,
    SENSOR_DEFAULT_SAMPLE_RATE,
    TRANSCRIPTION_ACCUMULATION_SECONDS,
    TRANSCRIPTION_FINAL_SILENCE_MS,
    TRANSCRIPTION_PORT,
    TRANSCRIPTION_PREROLL_CHUNKS,
    TRANSCRIPTION_SILENCE_THRESHOLD_RMS,
)

log = structlog.get_logger()

_DEFAULT_SAMPLE_RATE = 48000
_TRIGGER_PHRASE = "hey cleo"
_TRIGGER_WINDOW_SECONDS = 2.0
_MAX_GRPC_MESSAGE_BYTES = 16 * 1024 * 1024


def _load_parakeet_model():
    """Load the NVIDIA Parakeet ASR model once at startup."""
    import nemo.collections.asr as nemo_asr

    log.info("transcription.loading_model", model="nvidia/parakeet-tdt-0.6b-v3")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    log.info("transcription.model_loaded")
    return model


@dataclass(slots=True)
class TranscriptChunk:
    text: str
    start_time: float
    end_time: float


@dataclass(slots=True)
class PendingTrigger:
    start_time: float
    deadline: float


class TranscriptBuffer:
    """In-memory transcript state for one stream/session."""

    def __init__(self):
        self._final_chunks: list[TranscriptChunk] = []
        self._partial_chunk: TranscriptChunk | None = None
        self._lock = threading.Lock()

    def update(
        self,
        text: str,
        start_time: float,
        end_time: float,
        *,
        is_partial: bool,
    ) -> None:
        clean_text = text.strip()
        with self._lock:
            if is_partial:
                self._partial_chunk = (
                    TranscriptChunk(clean_text, start_time, end_time) if clean_text else None
                )
                return

            if clean_text:
                self._final_chunks.append(TranscriptChunk(clean_text, start_time, end_time))
            self._partial_chunk = None

    def full_text(self, include_partial: bool = True) -> str:
        with self._lock:
            parts = [chunk.text for chunk in self._final_chunks if chunk.text]
            if include_partial and self._partial_chunk and self._partial_chunk.text:
                parts.append(self._partial_chunk.text)
            return " ".join(parts).strip()

    def text_between(self, start_time: float, end_time: float) -> str:
        with self._lock:
            parts = [
                chunk.text
                for chunk in self._final_chunks
                if chunk.end_time > start_time and chunk.start_time < end_time and chunk.text
            ]
            if (
                self._partial_chunk
                and self._partial_chunk.end_time > start_time
                and self._partial_chunk.start_time < end_time
                and self._partial_chunk.text
            ):
                parts.append(self._partial_chunk.text)
            return " ".join(parts).strip()

    def clear(self) -> None:
        with self._lock:
            self._final_chunks.clear()
            self._partial_chunk = None


class AssistantCommandClient:
    """Client for wake-word handoff to AssistantService."""

    def __init__(self, address: str = ASSISTANT_ADDRESS, timeout: float = 10.0):
        self._timeout = timeout
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self._stub = assistant_pb2_grpc.AssistantServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def send_command(self, text: str) -> None:
        command = text.strip()
        if not command:
            return

        try:
            response = self._stub.ProcessCommand(
                assistant_pb2.CommandRequest(text=command),
                timeout=self._timeout,
            )
            log.info(
                "transcription.assistant_invoked",
                success=response.success,
                tool_name=response.tool_name,
                response=response.response_text,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.assistant_call_failed", error=str(exc))


class DataClient:
    """Client for storing finalized transcriptions."""

    def __init__(self, address: str = DATA_ADDRESS, timeout: float = 5.0):
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

    def store_transcription(self, result: transcription_pb2.TranscriptionResult) -> None:
        text = result.text.strip()
        if not text:
            return
        try:
            self._stub.StoreTranscription(
                data_pb2.StoreTranscriptionRequest(
                    text=text,
                    confidence=result.confidence,
                    start_time=result.start_time,
                    end_time=result.end_time,
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.store_failed", error=str(exc))


class StreamSession:
    """Owns transcript state and trigger handling for one audio stream."""

    def __init__(
        self,
        model,
        *,
        command_client: AssistantCommandClient,
        trigger_phrase: str = _TRIGGER_PHRASE,
        trigger_window_seconds: float = _TRIGGER_WINDOW_SECONDS,
    ):
        self._model = model
        self._command_client = command_client
        self._trigger_phrase = trigger_phrase.lower()
        self._trigger_window_seconds = trigger_window_seconds
        self._transcript = TranscriptBuffer()
        self._inference_lock = threading.Lock()
        self._pending_triggers: list[PendingTrigger] = []
        self._processed_phrase_count = 0

    def close(self) -> None:
        self._transcript.clear()

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with self._inference_lock:
                results = self._model.transcribe([tmp.name])

        if not results:
            return ""
        first = results[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "text"):
            return first.text
        return str(first)

    def process_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        start_time: float,
        end_time: float,
        is_partial: bool,
    ) -> transcription_pb2.TranscriptionResult:
        text = self.transcribe_audio(audio, sample_rate)
        self._transcript.update(text, start_time, end_time, is_partial=is_partial)
        self._detect_new_triggers(at_time=end_time)
        self._flush_ready_triggers(current_time=end_time, force=False)
        return transcription_pb2.TranscriptionResult(
            text=text,
            confidence=1.0 if text.strip() else 0.0,
            start_time=start_time,
            end_time=end_time,
            is_partial=is_partial,
        )

    def finalize(self, current_time: float) -> None:
        self._flush_ready_triggers(current_time=current_time, force=True)

    def _detect_new_triggers(self, *, at_time: float) -> None:
        phrase_count = self._transcript.full_text().lower().count(self._trigger_phrase)
        while phrase_count > self._processed_phrase_count:
            self._pending_triggers.append(
                PendingTrigger(
                    start_time=at_time,
                    deadline=at_time + self._trigger_window_seconds,
                )
            )
            self._processed_phrase_count += 1
            log.info(
                "transcription.trigger_detected",
                phrase=self._trigger_phrase,
                window_start=at_time,
                window_end=at_time + self._trigger_window_seconds,
            )

    def _flush_ready_triggers(self, *, current_time: float, force: bool) -> None:
        if not self._pending_triggers:
            return

        ready: list[PendingTrigger] = []
        waiting: list[PendingTrigger] = []
        for trigger in self._pending_triggers:
            if force or current_time >= trigger.deadline:
                ready.append(trigger)
            else:
                waiting.append(trigger)
        self._pending_triggers = waiting

        for trigger in ready:
            snippet = self._transcript.text_between(trigger.start_time, trigger.deadline)
            if not snippet:
                continue
            log.info(
                "transcription.trigger_window_ready",
                start=trigger.start_time,
                end=trigger.deadline,
                transcript=snippet,
            )
            self._command_client.send_command(snippet)


class TranscriptionServiceServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC transcription service backed by NVIDIA Parakeet."""

    def __init__(self, command_client: AssistantCommandClient | None = None):
        self._model = _load_parakeet_model()
        self._command_client = command_client or AssistantCommandClient()

    def _new_session(self) -> StreamSession:
        return StreamSession(self._model, command_client=self._command_client)

    @staticmethod
    def _sample_rate(request: transcription_pb2.AudioInput) -> int:
        return request.sample_rate if request.sample_rate > 0 else _DEFAULT_SAMPLE_RATE

    @staticmethod
    def _audio_from_request(request: transcription_pb2.AudioInput) -> np.ndarray:
        return np.frombuffer(request.audio_data, dtype=np.float32)

    def _iter_results(self, request_iterator, *, context=None):
        session = self._new_session()
        sample_rate = _DEFAULT_SAMPLE_RATE
        stream_time = 0.0
        buffer: list[np.ndarray] = []
        buffer_samples = 0

        try:
            for request in request_iterator:
                if context is not None and not context.is_active():
                    break

                sample_rate = self._sample_rate(request)
                chunk = self._audio_from_request(request)
                if len(chunk) == 0:
                    continue

                buffer.append(chunk)
                buffer_samples += len(chunk)
                threshold_samples = int(TRANSCRIPTION_ACCUMULATION_SECONDS * sample_rate)
                should_flush = buffer_samples >= threshold_samples or request.is_final
                if not should_flush:
                    continue

                audio = np.concatenate(buffer)
                duration = len(audio) / sample_rate
                start_time = stream_time
                end_time = stream_time + duration
                is_partial = not request.is_final

                buffer.clear()
                buffer_samples = 0
                stream_time = end_time

                yield session.process_chunk(
                    audio,
                    sample_rate,
                    start_time=start_time,
                    end_time=end_time,
                    is_partial=is_partial,
                )

                if request.is_final:
                    session.finalize(current_time=stream_time)

            if buffer_samples > 0:
                audio = np.concatenate(buffer)
                duration = len(audio) / sample_rate
                end_time = stream_time + duration
                yield session.process_chunk(
                    audio,
                    sample_rate,
                    start_time=stream_time,
                    end_time=end_time,
                    is_partial=False,
                )
                stream_time = end_time

            session.finalize(current_time=stream_time)
        finally:
            session.close()

    def Transcribe(self, request, context):
        sample_rate = self._sample_rate(request)
        audio = self._audio_from_request(request)
        if len(audio) == 0:
            return transcription_pb2.TranscriptionResult(text="", confidence=0.0)

        session = self._new_session()
        duration = len(audio) / sample_rate
        try:
            result = session.process_chunk(
                audio,
                sample_rate,
                start_time=0.0,
                end_time=duration,
                is_partial=False,
            )
            session.finalize(current_time=duration)
            return result
        finally:
            session.close()

    def TranscribeStream(self, request_iterator, context):
        for result in self._iter_results(request_iterator, context=context):
            yield result


class SensorTranscriptionPipeline(threading.Thread):
    """Persistent pipeline: sensor stream -> transcription -> data store."""

    def __init__(
        self,
        servicer: TranscriptionServiceServicer,
        *,
        sensor_address: str = SENSOR_ADDRESS,
        data_address: str = DATA_ADDRESS,
    ):
        super().__init__(daemon=True, name="SensorTranscriptionPipeline")
        self._servicer = servicer
        self._sensor_address = sensor_address
        self._data_address = data_address
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _chunk_rms(chunk_data: bytes) -> float:
        if not chunk_data:
            return 0.0
        audio = np.frombuffer(chunk_data, dtype=np.float32)
        if len(audio) == 0:
            return 0.0
        return math.sqrt(float(np.mean(audio * audio)))

    def _sensor_audio_requests(self, sensor_stub):
        request = sensor_pb2.StreamRequest(
            chunk_ms=SENSOR_DEFAULT_CHUNK_MS,
            sample_rate=SENSOR_DEFAULT_SAMPLE_RATE,
        )
        silence_ms = 0.0
        speech_active = False
        pre_roll: deque[tuple[bytes, int]] = deque(maxlen=TRANSCRIPTION_PREROLL_CHUNKS)

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
            chunk_ms = (
                (len(chunk.data) / 4) / chunk.sample_rate * 1000
                if chunk.sample_rate > 0
                else 0.0
            )

            if rms >= TRANSCRIPTION_SILENCE_THRESHOLD_RMS:
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
                silence_ms = 0.0
            elif speech_active:
                silence_ms += chunk_ms
            else:
                pre_roll.append((chunk.data, chunk.sample_rate))
                continue

            is_final = speech_active and silence_ms >= TRANSCRIPTION_FINAL_SILENCE_MS
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

    def run(self) -> None:
        sensor_channel = grpc.insecure_channel(
            self._sensor_address,
            options=[("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES)],
        )
        sensor_stub = sensor_pb2_grpc.SensorServiceStub(sensor_channel)
        data_client = DataClient(address=self._data_address)

        try:
            while not self._stop_event.is_set():
                try:
                    requests = self._sensor_audio_requests(sensor_stub)
                    for result in self._servicer._iter_results(requests):
                        if result.is_partial:
                            continue
                        data_client.store_transcription(result)
                except grpc.RpcError as exc:
                    if self._stop_event.is_set():
                        break
                    log.warning("transcription.sensor_stream_error", error=str(exc))
                    time.sleep(2)
        finally:
            data_client.close()
            sensor_channel.close()


def serve(port: int = TRANSCRIPTION_PORT) -> None:
    """Start transcription service and persistent sensor ingestion pipeline."""
    command_client = AssistantCommandClient(address=ASSISTANT_ADDRESS)
    servicer = TranscriptionServiceServicer(command_client=command_client)
    pipeline = SensorTranscriptionPipeline(servicer)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ],
    )
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    pipeline.start()
    log.info("transcription.started", port=port)

    def _shutdown(signum, frame) -> None:
        del frame
        log.info("transcription.stopping", signal=signum)
        pipeline.stop()
        pipeline.join(timeout=5)
        command_client.close()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
