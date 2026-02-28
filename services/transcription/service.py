"""Amazon Transcribe-backed transcription service and sensor ingestion pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent import futures
from dataclasses import dataclass
import os
import queue
import re
import signal
import threading
import time
from typing import Callable, Iterable, Iterator

import grpc
import numpy as np
import structlog

from generated import assistant_pb2, assistant_pb2_grpc
from generated import data_pb2, data_pb2_grpc
from generated import frontend_pb2, frontend_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc
from generated import transcription_pb2, transcription_pb2_grpc
from services.config import (
    ASSISTANT_ADDRESS,
    DATA_ADDRESS,
    FRONTEND_ADDRESS,
    SENSOR_ADDRESS,
    SENSOR_DEFAULT_CHUNK_MS,
    SENSOR_DEFAULT_SAMPLE_RATE,
    TRANSCRIPTION_PORT,
)

log = structlog.get_logger()

_MAX_GRPC_MESSAGE_BYTES = 16 * 1024 * 1024
_TRANSCRIBE_REGION = (
    os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
)
_DEBUG_TRANSCRIPTION_HUD_ENV = "CLEO_DEBUG_TRANSCRIPTION_HUD"
_DEBUG_TRANSCRIPT_MAX_CHARS = 180
_ASSISTANT_RESPONSE_LOG_MAX_CHARS = 240
_TRIGGER_PHRASES = ("hey cleo", "hey clio", "hi cleo", "hi clio")
_TRIGGER_CAPTURE_SECONDS = 3.0
_QUEUE_SENTINEL = object()


def normalize_for_trigger_match(text: str) -> str:
    """Normalize transcript text for wake-word matching only."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(normalized.split())


def _float32_to_pcm16(audio: bytes) -> bytes:
    """Convert mono float32 PCM bytes to little-endian signed 16-bit PCM."""
    if not audio:
        return b""

    samples = np.frombuffer(audio, dtype=np.float32)
    if len(samples) == 0:
        return b""

    clipped = np.clip(samples, -1.0, 1.0)
    scaled = (clipped * 32767.0).astype(np.int16)
    return scaled.tobytes()


def _audio_duration_seconds(audio: bytes, sample_rate: int) -> float:
    """Estimate the duration of a mono float32 PCM buffer in seconds."""
    if not audio or sample_rate <= 0:
        return 0.0

    bytes_per_sample = np.dtype(np.float32).itemsize
    sample_count = len(audio) // bytes_per_sample
    if sample_count <= 0:
        return 0.0
    return sample_count / float(sample_rate)


@dataclass(slots=True)
class TranscriptSpan:
    text: str
    start_time: float
    end_time: float
    is_partial: bool


@dataclass(slots=True)
class PendingTrigger:
    start_time: float
    end_time: float


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
            response_text = response.response_text.strip()
            if len(response_text) > _ASSISTANT_RESPONSE_LOG_MAX_CHARS:
                response_text = (
                    f"{response_text[:_ASSISTANT_RESPONSE_LOG_MAX_CHARS - 3]}..."
                )
            log.info(
                "transcription.assistant_invoked",
                success=response.success,
                tool_name=response.tool_name,
                response_text=response_text,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.assistant_call_failed", error=str(exc))


class DataClient:
    """Client for storing finalized transcription results."""

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


class FrontendTranscriptDebugClient:
    """Optional HUD publisher for final transcript text in debug mode."""

    def __init__(self, address: str = FRONTEND_ADDRESS, timeout: float = 2.0):
        enabled = os.getenv(_DEBUG_TRANSCRIPTION_HUD_ENV, "").strip().lower()
        self._enabled = enabled in {"1", "true", "yes", "on"}
        self._timeout = timeout
        self._channel = None
        self._stub = None

        if self._enabled:
            self._channel = grpc.insecure_channel(
                address,
                options=[
                    ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                    ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ],
            )
            self._stub = frontend_pb2_grpc.FrontendServiceStub(self._channel)
            log.info(
                "transcription.debug_hud_enabled",
                env_var=_DEBUG_TRANSCRIPTION_HUD_ENV,
                address=address,
            )

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()

    def show_final_transcript(self, result: transcription_pb2.TranscriptionResult) -> None:
        if not self._enabled or self._stub is None or result.is_partial:
            return

        text = result.text.strip()
        if not text:
            return

        if len(text) > _DEBUG_TRANSCRIPT_MAX_CHARS:
            text = f"{text[:_DEBUG_TRANSCRIPT_MAX_CHARS - 1]}..."

        try:
            self._stub.ShowText(
                frontend_pb2.TextRequest(
                    text=f"ASR: {text}",
                    position="top-right",
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.debug_hud_publish_failed", error=str(exc))


class TriggerRouter:
    """Tracks wake phrases and sends the next three seconds of transcript to the assistant."""

    def __init__(
        self,
        command_client: AssistantCommandClient,
        *,
        trigger_phrases: tuple[str, ...] = _TRIGGER_PHRASES,
        capture_seconds: float = _TRIGGER_CAPTURE_SECONDS,
        on_trigger_detected: Callable[[transcription_pb2.TranscriptionResult], None] | None = None,
    ):
        self._command_client = command_client
        self._trigger_phrases = tuple(
            normalize_for_trigger_match(phrase) for phrase in trigger_phrases
        )
        self._capture_seconds = capture_seconds
        self._on_trigger_detected = on_trigger_detected
        self._history: deque[TranscriptSpan] = deque()
        self._pending: list[PendingTrigger] = []
        self._last_trigger_start: float | None = None
        self._lock = threading.Lock()

    def observe(self, result: transcription_pb2.TranscriptionResult) -> None:
        text = result.text.strip()
        if not text:
            self._flush_ready(result.end_time, force=result.is_partial is False)
            return

        span = TranscriptSpan(
            text=text,
            start_time=result.start_time,
            end_time=result.end_time,
            is_partial=result.is_partial,
        )

        with self._lock:
            if not result.is_partial:
                self._history.append(span)

            normalized = normalize_for_trigger_match(text)
            trigger_seen = any(phrase in normalized for phrase in self._trigger_phrases)
            is_duplicate = (
                self._last_trigger_start is not None
                and abs(self._last_trigger_start - result.start_time) < 0.25
            )
            if trigger_seen and not is_duplicate:
                self._pending.append(
                    PendingTrigger(
                        start_time=result.start_time,
                        end_time=result.start_time + self._capture_seconds,
                    )
                )
                self._last_trigger_start = result.start_time
                log.info(
                    "transcription.trigger_detected",
                    start_time=result.start_time,
                    end_time=result.start_time + self._capture_seconds,
                    text=text,
                )
                if self._on_trigger_detected is not None:
                    self._on_trigger_detected(result)

            self._prune_history(current_time=result.end_time)
            self._flush_ready(result.end_time, force=False, partial_span=span)

    def finalize(self) -> None:
        with self._lock:
            cutoff = self._history[-1].end_time if self._history else 0.0
            self._flush_ready(cutoff, force=True, partial_span=None)

    def _prune_history(self, *, current_time: float) -> None:
        threshold = current_time - max(10.0, self._capture_seconds + 2.0)
        while self._history and self._history[0].end_time < threshold:
            self._history.popleft()

    def _flush_ready(
        self,
        current_time: float,
        *,
        force: bool,
        partial_span: TranscriptSpan | None = None,
    ) -> None:
        if not self._pending:
            return

        ready: list[PendingTrigger] = []
        waiting: list[PendingTrigger] = []
        for trigger in self._pending:
            if force or current_time >= trigger.end_time:
                ready.append(trigger)
            else:
                waiting.append(trigger)
        self._pending = waiting

        for trigger in ready:
            snippet = self._snippet_for_trigger(trigger, partial_span=partial_span)
            if not snippet:
                continue
            log.info(
                "transcription.trigger_window_ready",
                start_time=trigger.start_time,
                end_time=trigger.end_time,
                text=snippet,
            )
            self._command_client.send_command(snippet)

    def _snippet_for_trigger(
        self,
        trigger: PendingTrigger,
        *,
        partial_span: TranscriptSpan | None,
    ) -> str:
        parts = [
            span.text
            for span in self._history
            if span.end_time > trigger.start_time and span.start_time < trigger.end_time
        ]

        if (
            partial_span is not None
            and partial_span.end_time > trigger.start_time
            and partial_span.start_time < trigger.end_time
        ):
            parts.append(partial_span.text)

        return " ".join(part.strip() for part in parts if part.strip()).strip()


class AmazonTranscribeBackend:
    """Wraps the async Amazon Transcribe streaming client behind sync iterators."""

    def __init__(self, region: str = _TRANSCRIBE_REGION):
        self._region = region

    @staticmethod
    def _stream_start_epoch(first_request: transcription_pb2.AudioInput) -> float:
        """Anchor stream-relative offsets to a UTC epoch when available."""
        duration = _audio_duration_seconds(first_request.audio_data, first_request.sample_rate)
        chunk_end = (
            float(first_request.timestamp)
            if float(first_request.timestamp) > 0.0
            else time.time()
        )
        return max(0.0, chunk_end - duration)

    @staticmethod
    def _sdk_imports():
        try:
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "amazon-transcribe is not installed. Add the package and run `uv sync`."
            ) from exc

        return TranscribeStreamingClient, TranscriptResultStreamHandler

    def transcribe_stream(
        self, requests: Iterable[transcription_pb2.AudioInput]
    ) -> Iterator[transcription_pb2.TranscriptionResult]:
        input_queue: queue.Queue[object] = queue.Queue()
        output_queue: queue.Queue[object] = queue.Queue()

        def _enqueue_requests() -> None:
            try:
                for request in requests:
                    input_queue.put(request)
            except Exception as exc:  # pragma: no cover - defensive passthrough
                output_queue.put(exc)
            finally:
                input_queue.put(_QUEUE_SENTINEL)

        producer = threading.Thread(
            target=_enqueue_requests,
            daemon=True,
            name="TranscribeRequestProducer",
        )
        producer.start()

        worker = threading.Thread(
            target=self._run_transcribe_worker,
            args=(input_queue, output_queue),
            daemon=True,
            name="AmazonTranscribeWorker",
        )
        worker.start()

        while True:
            item = output_queue.get()
            if item is _QUEUE_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        worker.join(timeout=1)
        producer.join(timeout=1)

    def transcribe_once(
        self, request: transcription_pb2.AudioInput
    ) -> transcription_pb2.TranscriptionResult:
        final_segments: list[transcription_pb2.TranscriptionResult] = []
        partial_segment: transcription_pb2.TranscriptionResult | None = None

        for result in self.transcribe_stream([request]):
            if result.is_partial:
                partial_segment = result
            else:
                final_segments.append(result)

        if final_segments:
            return self._merge_segments(final_segments)
        if partial_segment is not None:
            return partial_segment

        return transcription_pb2.TranscriptionResult(
            text="",
            confidence=0.0,
            start_time=0.0,
            end_time=0.0,
            is_partial=False,
            utterance_id="",
        )

    def _run_transcribe_worker(
        self,
        input_queue: queue.Queue[object],
        output_queue: queue.Queue[object],
    ) -> None:
        try:
            asyncio.run(self._run_transcribe_async(input_queue, output_queue))
        except Exception as exc:
            output_queue.put(exc)
            output_queue.put(_QUEUE_SENTINEL)

    async def _run_transcribe_async(
        self,
        input_queue: queue.Queue[object],
        output_queue: queue.Queue[object],
    ) -> None:
        TranscribeStreamingClient, TranscriptResultStreamHandler = self._sdk_imports()
        client = TranscribeStreamingClient(region=self._region)

        first_request = await asyncio.to_thread(input_queue.get)
        if first_request is _QUEUE_SENTINEL:
            output_queue.put(_QUEUE_SENTINEL)
            return

        if not isinstance(first_request, transcription_pb2.AudioInput):
            raise RuntimeError("Unexpected request type passed to transcription backend")

        sample_rate = (
            first_request.sample_rate
            if first_request.sample_rate > 0
            else SENSOR_DEFAULT_SAMPLE_RATE
        )
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=sample_rate,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="medium",
        )
        stream_start_epoch = self._stream_start_epoch(first_request)

        class ResultHandler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event) -> None:
                results = getattr(transcript_event.transcript, "results", [])
                for result in results:
                    alternatives = getattr(result, "alternatives", [])
                    if not alternatives:
                        continue
                    text = getattr(alternatives[0], "transcript", "").strip()
                    if not text:
                        continue

                    output_queue.put(
                        transcription_pb2.TranscriptionResult(
                            text=text,
                            confidence=1.0,
                            start_time=stream_start_epoch
                            + float(getattr(result, "start_time", 0.0) or 0.0),
                            end_time=stream_start_epoch
                            + float(getattr(result, "end_time", 0.0) or 0.0),
                            is_partial=bool(getattr(result, "is_partial", False)),
                            utterance_id=str(getattr(result, "result_id", "") or ""),
                        )
                    )

        async def _send_audio() -> None:
            current = first_request
            while True:
                if current.reset_context:
                    log.info(
                        "transcription.reset_context_ignored",
                        reason="Amazon Transcribe streams cannot be reset mid-session",
                    )

                pcm_chunk = _float32_to_pcm16(current.audio_data)
                if pcm_chunk:
                    await stream.input_stream.send_audio_event(audio_chunk=pcm_chunk)

                if current.is_final:
                    break

                next_item = await asyncio.to_thread(input_queue.get)
                if next_item is _QUEUE_SENTINEL:
                    break
                if not isinstance(next_item, transcription_pb2.AudioInput):
                    raise RuntimeError(
                        "Unexpected request type passed to transcription backend"
                    )
                current = next_item

            await stream.input_stream.end_stream()

        handler = ResultHandler(stream.output_stream)
        await asyncio.gather(_send_audio(), handler.handle_events())
        output_queue.put(_QUEUE_SENTINEL)

    @staticmethod
    def _merge_segments(
        segments: list[transcription_pb2.TranscriptionResult],
    ) -> transcription_pb2.TranscriptionResult:
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        if not text:
            return transcription_pb2.TranscriptionResult(
                text="",
                confidence=0.0,
                start_time=0.0,
                end_time=0.0,
                is_partial=False,
                utterance_id="",
            )

        start_time = min(segment.start_time for segment in segments)
        end_time = max(segment.end_time for segment in segments)
        confidence = sum(segment.confidence for segment in segments) / len(segments)

        return transcription_pb2.TranscriptionResult(
            text=text,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            is_partial=False,
            utterance_id=segments[-1].utterance_id,
        )


class TranscriptionServiceServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC transcription service backed by Amazon Transcribe Streaming."""

    def __init__(
        self,
        *,
        backend: AmazonTranscribeBackend | None = None,
        command_client: AssistantCommandClient | None = None,
        on_trigger_detected: Callable[[transcription_pb2.TranscriptionResult], None] | None = None,
    ):
        self._backend = backend or AmazonTranscribeBackend()
        self._command_client = command_client or AssistantCommandClient()
        self._on_trigger_detected = on_trigger_detected

    def Transcribe(self, request, context):
        if not request.audio_data:
            return transcription_pb2.TranscriptionResult(
                text="",
                confidence=0.0,
                start_time=0.0,
                end_time=0.0,
                is_partial=False,
                utterance_id="",
            )

        try:
            result = self._backend.transcribe_once(request)
            trigger_router = TriggerRouter(
                self._command_client,
                on_trigger_detected=self._on_trigger_detected,
            )
            trigger_router.observe(result)
            trigger_router.finalize()
            return result
        except RuntimeError as exc:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(str(exc))
            return transcription_pb2.TranscriptionResult()
        except Exception as exc:
            log.exception("transcription.unary_failed", error=str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Transcription failed: {exc}")
            return transcription_pb2.TranscriptionResult()

    def TranscribeStream(self, request_iterator, context):
        trigger_router = TriggerRouter(
            self._command_client,
            on_trigger_detected=self._on_trigger_detected,
        )
        try:
            for result in self._backend.transcribe_stream(request_iterator):
                if context is not None and not context.is_active():
                    break
                trigger_router.observe(result)
                yield result
            trigger_router.finalize()
        except grpc.RpcError as exc:
            if context is not None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(str(exc))
                return
            raise
        except RuntimeError as exc:
            if context is not None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(str(exc))
                return
            raise
        except Exception as exc:
            if context is not None:
                log.exception("transcription.stream_failed", error=str(exc))
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Transcription failed: {exc}")
                return
            raise
        finally:
            trigger_router.finalize()


class SensorTranscriptionPipeline(threading.Thread):
    """Persistent pipeline: sensor stream -> transcription -> data store -> assistant."""

    def __init__(
        self,
        servicer: TranscriptionServiceServicer,
        *,
        sensor_address: str = SENSOR_ADDRESS,
        data_address: str = DATA_ADDRESS,
        debug_client: FrontendTranscriptDebugClient | None = None,
    ):
        super().__init__(daemon=True, name="SensorTranscriptionPipeline")
        self._servicer = servicer
        self._sensor_address = sensor_address
        self._data_address = data_address
        self._debug_client = debug_client
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _sensor_audio_requests(self, sensor_stub):
        stream_request = sensor_pb2.StreamRequest(
            chunk_ms=SENSOR_DEFAULT_CHUNK_MS,
            sample_rate=SENSOR_DEFAULT_SAMPLE_RATE,
        )
        chunk_index = 0

        for chunk in sensor_stub.StreamAudio(stream_request):
            if self._stop_event.is_set():
                return

            chunk_index += 1
            yield transcription_pb2.AudioInput(
                audio_data=chunk.data,
                sample_rate=chunk.sample_rate,
                is_final=False,
                stream_id=f"sensor-{chunk_index}",
                timestamp=chunk.timestamp,
            )

        if not self._stop_event.is_set():
            yield transcription_pb2.AudioInput(
                audio_data=b"",
                sample_rate=SENSOR_DEFAULT_SAMPLE_RATE,
                is_final=True,
                stream_id="sensor-final",
                timestamp=time.time(),
            )

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
                    for result in self._servicer.TranscribeStream(requests, context=None):
                        if not result.is_partial:
                            if self._debug_client is not None:
                                self._debug_client.show_final_transcript(result)
                            data_client.store_transcription(result)
                except grpc.RpcError as exc:
                    if self._stop_event.is_set():
                        break
                    log.warning("transcription.sensor_stream_error", error=str(exc))
                    time.sleep(2)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    log.warning("transcription.pipeline_error", error=str(exc))
                    time.sleep(2)
        finally:
            data_client.close()
            sensor_channel.close()


def serve(port: int = TRANSCRIPTION_PORT) -> None:
    """Start transcription service and persistent sensor ingestion pipeline."""
    command_client = AssistantCommandClient(address=ASSISTANT_ADDRESS)
    debug_client = FrontendTranscriptDebugClient(address=FRONTEND_ADDRESS)
    servicer = TranscriptionServiceServicer(command_client=command_client)
    pipeline = SensorTranscriptionPipeline(servicer, debug_client=debug_client)

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
        debug_client.close()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
