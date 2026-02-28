"""Streaming transcription service with trigger-based tool invocation."""

from __future__ import annotations

import os
import signal
import tempfile
import threading
from concurrent import futures
from dataclasses import dataclass

import grpc
import numpy as np
import soundfile as sf
import structlog
from google.protobuf import wrappers_pb2

from generated import transcription_pb2, transcription_pb2_grpc

log = structlog.get_logger()

_DEFAULT_SAMPLE_RATE = 48000
_ACCUMULATION_SECONDS = 1.5
_TRIGGER_PHRASE = "hey cleo"
_TRIGGER_WINDOW_SECONDS = 2.0
_DEFAULT_TOOL_METHOD = "/cleo.tooling.ToolCallingService/AnalyzeTranscript"
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
    """A finalized transcript chunk with stream-relative timing."""

    text: str
    start_time: float
    end_time: float


@dataclass(slots=True)
class PendingTrigger:
    """Tracks a post-trigger capture window."""

    start_time: float
    deadline: float


class TranscriptBuffer:
    """In-memory transcript state for a stream."""

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


class ToolCallingClient:
    """Optional gRPC client for forwarding trigger windows to a tool service."""

    def __init__(
        self,
        address: str | None = None,
        *,
        method: str = _DEFAULT_TOOL_METHOD,
        timeout: float = 5.0,
    ):
        self._address = address or os.getenv("CLEO_TOOL_CALLER_ADDR", "").strip()
        self._method = os.getenv("CLEO_TOOL_CALLER_METHOD", method).strip() or method
        self._timeout = timeout
        self._lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._rpc = None

        if self._address:
            self._connect()
        else:
            log.info("tool_calling.disabled")

    @property
    def enabled(self) -> bool:
        return self._rpc is not None

    def _connect(self) -> None:
        self._channel = grpc.insecure_channel(
            self._address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self._rpc = self._channel.unary_unary(
            self._method,
            request_serializer=wrappers_pb2.StringValue.SerializeToString,
            response_deserializer=wrappers_pb2.StringValue.FromString,
        )
        log.info("tool_calling.connected", address=self._address, method=self._method)

    def close(self) -> None:
        with self._lock:
            if self._channel is not None:
                self._channel.close()
                self._channel = None
                self._rpc = None

    def send_transcript(self, text: str) -> None:
        if not self._rpc or not text.strip():
            return

        request = wrappers_pb2.StringValue(value=text)
        try:
            with self._lock:
                response = self._rpc(request, timeout=self._timeout)
            log.info("tool_calling.invoked", transcript=text, response=response.value)
        except grpc.RpcError as exc:
            log.warning(
                "tool_calling.failed",
                address=self._address,
                method=self._method,
                error=str(exc),
            )


class StreamSession:
    """Owns transcript state and trigger handling for one audio stream."""

    def __init__(
        self,
        model,
        *,
        tool_client: ToolCallingClient | None = None,
        trigger_phrase: str = _TRIGGER_PHRASE,
        trigger_window_seconds: float = _TRIGGER_WINDOW_SECONDS,
    ):
        self._model = model
        self._tool_client = tool_client or ToolCallingClient()
        self._trigger_phrase = trigger_phrase.lower()
        self._trigger_window_seconds = trigger_window_seconds
        self._transcript = TranscriptBuffer()
        self._inference_lock = threading.Lock()
        self._pending_triggers: list[PendingTrigger] = []
        self._processed_phrase_count = 0

    def close(self) -> None:
        self._transcript.clear()

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Run ASR against a PCM float32 buffer."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            with self._inference_lock:
                results = self._model.transcribe([tmp.name])

        if not results:
            return ""
        first_result = results[0]
        if isinstance(first_result, str):
            return first_result
        if hasattr(first_result, "text"):
            return first_result.text
        return str(first_result)

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
        self._transcript.update(
            text,
            start_time,
            end_time,
            is_partial=is_partial,
        )
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
            if snippet:
                log.info(
                    "transcription.trigger_window_ready",
                    start=trigger.start_time,
                    end=trigger.deadline,
                    transcript=snippet,
                )
                self._tool_client.send_transcript(snippet)


class TranscriptionServiceServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC transcription service backed by NVIDIA Parakeet."""

    def __init__(self, tool_client: ToolCallingClient | None = None):
        self._model = _load_parakeet_model()
        self._tool_client = tool_client or ToolCallingClient()

    def _new_session(self) -> StreamSession:
        return StreamSession(self._model, tool_client=self._tool_client)

    @staticmethod
    def _sample_rate(request: transcription_pb2.AudioInput) -> int:
        return request.sample_rate if request.sample_rate > 0 else _DEFAULT_SAMPLE_RATE

    @staticmethod
    def _audio_from_request(request: transcription_pb2.AudioInput) -> np.ndarray:
        return np.frombuffer(request.audio_data, dtype=np.float32)

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
        session = self._new_session()
        sample_rate = _DEFAULT_SAMPLE_RATE
        stream_time = 0.0
        buffer: list[np.ndarray] = []
        buffer_samples = 0

        try:
            for request in request_iterator:
                if not context.is_active():
                    break

                sample_rate = self._sample_rate(request)
                chunk = self._audio_from_request(request)
                if len(chunk) == 0:
                    continue

                buffer.append(chunk)
                buffer_samples += len(chunk)
                threshold_samples = int(_ACCUMULATION_SECONDS * sample_rate)
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


def serve(port: int = 50052) -> None:
    """Start the transcription gRPC server."""
    tool_client = ToolCallingClient()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ],
    )
    servicer = TranscriptionServiceServicer(tool_client=tool_client)
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("transcription.started", port=port)

    def _shutdown(signum, frame):
        del frame
        log.info("transcription.stopping", signal=signum)
        tool_client.close()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
