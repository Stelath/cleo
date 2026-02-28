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
from typing import Callable, Iterable, Iterator, Protocol

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
_DEBUG_TRANSCRIPT_MAX_LINES = 5
_ASSISTANT_RESPONSE_LOG_MAX_CHARS = 240
_SPEAKER_ALIAS_RESET_SECONDS = 90.0
_TRIGGER_PHRASES = ("hey cleo", "hey clio", "hi cleo", "hi clio")
_TRIGGER_CAPTURE_SECONDS = 3.0
_TRIGGER_PREROLL_SECONDS = 0.75
_TRIGGER_FINAL_FLUSH_GRACE_SECONDS = 0.2
_TRIGGER_EARLY_FINAL_SECONDS = 1.25
_TRIGGER_SPEAKER_HANDOFF_SECONDS = 0.45
_FOLLOW_UP_BASE_WINDOW_SECONDS = 6.0
_FOLLOW_UP_MAX_WINDOW_SECONDS = 14.0
_FOLLOW_UP_BUFFER_SECONDS = 1.25
_FOLLOW_UP_WORDS_PER_SECOND = 2.8
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
    speaker_label: str


@dataclass(slots=True)
class PendingTrigger:
    trigger_time: float
    capture_start_time: float
    capture_end_time: float
    detected_monotonic: float
    speaker_label: str


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

    def send_command(
        self,
        text: str,
        *,
        is_follow_up: bool = False,
    ) -> assistant_pb2.CommandResponse | None:
        command = text.strip()
        if not command:
            return None

        try:
            response = self._stub.ProcessCommand(
                assistant_pb2.CommandRequest(
                    text=command,
                    is_follow_up=is_follow_up,
                ),
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
            return response
        except grpc.RpcError as exc:
            log.warning("transcription.assistant_call_failed", error=str(exc))
            return None


class CommandClient(Protocol):
    def send_command(
        self,
        text: str,
        *,
        is_follow_up: bool = False,
    ) -> assistant_pb2.CommandResponse | None: ...


class DataClient:
    """Client for storing finalized transcription results."""

    def __init__(self, address: str = DATA_ADDRESS, timeout: float = 5.0):
        self._timeout = timeout
        self._speaker_aliases: dict[str, str] = {}
        self._next_speaker_alias = 1
        self._last_alias_activity_at = 0.0
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

    def _speaker_turn_rows(
        self,
        result: transcription_pb2.TranscriptionResult,
    ) -> list[tuple[str, float, float]]:
        rows: list[tuple[str, float, float]] = []
        reference_time = result.end_time if result.end_time > 0 else time.time()
        self._maybe_reset_speaker_aliases(reference_time)

        for turn in result.speaker_turns:
            turn_text = turn.text.strip()
            if not turn_text:
                continue

            label_key = turn.speaker_label.strip() or "__unknown__"
            speaker_alias = self._speaker_aliases.get(label_key)
            if speaker_alias is None:
                speaker_alias = f"Speaker {self._next_speaker_alias}"
                self._next_speaker_alias += 1
                self._speaker_aliases[label_key] = speaker_alias

            start_time = turn.start_time if turn.start_time > 0 else result.start_time
            end_time = turn.end_time if turn.end_time > 0 else result.end_time
            rows.append((f"{speaker_alias}: {turn_text}", start_time, end_time))

        return rows

    def _maybe_reset_speaker_aliases(self, reference_time: float) -> None:
        if (
            self._last_alias_activity_at > 0.0
            and reference_time - self._last_alias_activity_at > _SPEAKER_ALIAS_RESET_SECONDS
        ):
            self._speaker_aliases.clear()
            self._next_speaker_alias = 1
        self._last_alias_activity_at = reference_time

    def store_transcription(self, result: transcription_pb2.TranscriptionResult) -> None:
        try:
            speaker_rows = self._speaker_turn_rows(result)
            if speaker_rows:
                for text, start_time, end_time in speaker_rows:
                    self._stub.StoreTranscription(
                        data_pb2.StoreTranscriptionRequest(
                            text=text,
                            confidence=result.confidence,
                            start_time=start_time,
                            end_time=end_time,
                        ),
                        timeout=self._timeout,
                    )
                return

            text = result.text.strip()
            if not text:
                return

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
        self._speaker_aliases: dict[str, str] = {}
        self._speaker_index = 1
        self._recent_lines: deque[str] = deque(maxlen=_DEBUG_TRANSCRIPT_MAX_LINES)
        self._lock = threading.Lock()

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

        lines = self._format_debug_lines(result)
        if not lines:
            return

        clipped_lines = []
        for line in lines:
            if len(line) <= _DEBUG_TRANSCRIPT_MAX_CHARS:
                clipped_lines.append(line)
            else:
                clipped_lines.append(f"{line[:_DEBUG_TRANSCRIPT_MAX_CHARS - 3]}...")

        with self._lock:
            self._recent_lines.extend(clipped_lines)
            while self._recent_lines and len("\n".join(self._recent_lines)) > _DEBUG_TRANSCRIPT_MAX_CHARS:
                self._recent_lines.popleft()
            text = "\n".join(self._recent_lines)

        try:
            self._stub.ShowText(
                frontend_pb2.TextRequest(
                    text=f"ASR:\n{text}",
                    position="upper-right",
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.debug_hud_publish_failed", error=str(exc))

    def _format_debug_lines(
        self,
        result: transcription_pb2.TranscriptionResult,
    ) -> list[str]:
        if result.speaker_turns:
            lines: list[str] = []
            for turn in result.speaker_turns:
                text = turn.text.strip()
                if not text:
                    continue
                lines.append(f"{self._speaker_tag(turn.speaker_label)}: {text}")
            if lines:
                return lines

        text = result.text.strip()
        if not text:
            return []
        return [f"{self._speaker_tag(result.speaker_label)}: {text}"]

    def _speaker_tag(self, speaker_label: str) -> str:
        label = speaker_label.strip() or "unknown"
        with self._lock:
            existing = self._speaker_aliases.get(label)
            if existing is not None:
                return existing
            tag = f"S{self._speaker_index}"
            self._speaker_index += 1
            self._speaker_aliases[label] = tag
            return tag


class FrontendThrobberClient:
    """Best-effort client to show/hide the activation throbber on the HUD."""

    def __init__(self, address: str = FRONTEND_ADDRESS, timeout: float = 2.0):
        self._timeout = timeout
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self._stub = frontend_pb2_grpc.FrontendServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def show(self) -> None:
        try:
            self._stub.ShowThrobber(
                frontend_pb2.ThrobberRequest(
                    visible=True,
                    position="top-right",
                    color="#d7ebff",
                    hz=0.6,
                    size_px=28,
                ),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.throbber_show_failed", error=str(exc))

    def hide(self) -> None:
        try:
            self._stub.ShowThrobber(
                frontend_pb2.ThrobberRequest(visible=False),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            log.warning("transcription.throbber_hide_failed", error=str(exc))


class TriggerRouter:
    """Tracks wake phrases and sends the next three seconds of transcript to the assistant."""

    def __init__(
        self,
        command_client: CommandClient,
        *,
        trigger_phrases: tuple[str, ...] = _TRIGGER_PHRASES,
        capture_seconds: float = _TRIGGER_CAPTURE_SECONDS,
        preroll_seconds: float = _TRIGGER_PREROLL_SECONDS,
        final_flush_grace_seconds: float = _TRIGGER_FINAL_FLUSH_GRACE_SECONDS,
        early_final_seconds: float = _TRIGGER_EARLY_FINAL_SECONDS,
        speaker_handoff_seconds: float = _TRIGGER_SPEAKER_HANDOFF_SECONDS,
        on_trigger_detected: Callable[[transcription_pb2.TranscriptionResult], None] | None = None,
        throbber_client: FrontendThrobberClient | None = None,
    ):
        self._command_client = command_client
        self._trigger_phrases = tuple(
            normalize_for_trigger_match(phrase) for phrase in trigger_phrases
        )
        self._capture_seconds = capture_seconds
        self._preroll_seconds = max(0.0, preroll_seconds)
        self._final_flush_grace_seconds = max(0.0, final_flush_grace_seconds)
        self._early_final_seconds = max(0.0, early_final_seconds)
        self._speaker_handoff_seconds = max(0.0, speaker_handoff_seconds)
        self._on_trigger_detected = on_trigger_detected
        self._throbber_client = throbber_client
        self._history: deque[TranscriptSpan] = deque()
        self._pending: list[PendingTrigger] = []
        self._last_trigger_start: float | None = None
        self._last_trigger_utterance_id: str | None = None
        self._follow_up_until: float | None = None
        self._follow_up_speakers: set[str] = set()
        self._last_follow_up_utterance_id: str | None = None
        self._lock = threading.Lock()

    def observe(self, result: transcription_pb2.TranscriptionResult) -> None:
        text = result.text.strip()
        if not text:
            self._flush_ready(
                result.end_time,
                force=result.is_partial is False,
                current_is_final=result.is_partial is False,
                current_speaker_label=result.speaker_label,
            )
            return

        span = TranscriptSpan(
            text=text,
            start_time=result.start_time,
            end_time=result.end_time,
            is_partial=result.is_partial,
            speaker_label=result.speaker_label.strip(),
        )

        with self._lock:
            if not result.is_partial:
                self._history.append(span)

            normalized = normalize_for_trigger_match(text)
            trigger_seen = any(phrase in normalized for phrase in self._trigger_phrases)
            utterance_id = result.utterance_id.strip()
            is_duplicate = (
                self._last_trigger_start is not None
                and abs(self._last_trigger_start - result.start_time) < 0.25
            )
            if (
                not is_duplicate
                and utterance_id
                and self._last_trigger_utterance_id == utterance_id
            ):
                is_duplicate = True
            if trigger_seen and not is_duplicate:
                trigger_speaker = self._speaker_for_last_trigger_phrase(result)
                if trigger_speaker:
                    self._follow_up_speakers.add(trigger_speaker)
                self._deactivate_follow_up(clear_speakers=False)
                trigger_time = result.end_time if result.end_time > 0 else result.start_time
                capture_start_time = max(0.0, trigger_time - self._preroll_seconds)
                capture_end_time = trigger_time + self._capture_seconds
                self._pending.append(
                    PendingTrigger(
                        trigger_time=trigger_time,
                        capture_start_time=capture_start_time,
                        capture_end_time=capture_end_time,
                        detected_monotonic=time.monotonic(),
                        speaker_label=trigger_speaker,
                    )
                )
                self._last_trigger_start = result.start_time
                if utterance_id:
                    self._last_trigger_utterance_id = utterance_id
                log.info(
                    "transcription.trigger_detected",
                    trigger_time=trigger_time,
                    capture_start_time=capture_start_time,
                    capture_end_time=capture_end_time,
                    speaker_label=trigger_speaker,
                    text=text,
                )
                if self._throbber_client is not None:
                    self._throbber_client.show()
                if self._on_trigger_detected is not None:
                    self._on_trigger_detected(result)

            if not result.is_partial and not trigger_seen:
                self._dispatch_follow_up_if_active(result)

            self._prune_history(current_time=result.end_time)
            self._flush_ready(
                result.end_time,
                force=False,
                partial_span=span if result.is_partial else None,
                current_is_final=result.is_partial is False,
                current_speaker_label=result.speaker_label,
            )

    def finalize(self) -> None:
        with self._lock:
            cutoff = self._history[-1].end_time if self._history else 0.0
            self._flush_ready(
                cutoff,
                force=True,
                partial_span=None,
                current_is_final=True,
                current_speaker_label="",
            )

    def _prune_history(self, *, current_time: float) -> None:
        threshold = current_time - max(
            10.0, self._capture_seconds + self._preroll_seconds + 2.0
        )
        while self._history and self._history[0].end_time < threshold:
            self._history.popleft()

    def _flush_ready(
        self,
        current_time: float,
        *,
        force: bool,
        current_is_final: bool,
        current_speaker_label: str,
        partial_span: TranscriptSpan | None = None,
    ) -> None:
        if not self._pending:
            return

        ready: list[PendingTrigger] = []
        waiting: list[PendingTrigger] = []
        for trigger in self._pending:
            window_elapsed = current_time >= trigger.capture_end_time
            early_final_ready = current_is_final and (
                current_time >= trigger.trigger_time + self._early_final_seconds
            )
            speaker_handoff_ready = self._speaker_handoff_ready(
                trigger=trigger,
                current_time=current_time,
                current_speaker_label=current_speaker_label,
            )
            timed_out = (
                time.monotonic() - trigger.detected_monotonic
                >= self._capture_seconds + self._final_flush_grace_seconds
            )
            if force or early_final_ready or speaker_handoff_ready or (
                window_elapsed and (current_is_final or timed_out)
            ):
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
                trigger_time=trigger.trigger_time,
                start_time=trigger.capture_start_time,
                end_time=trigger.capture_end_time,
                text=snippet,
            )
            response = self._command_client.send_command(snippet)
            if self._throbber_client is not None:
                self._throbber_client.hide()
            if response is not None and response.success and response.continue_follow_up:
                self._activate_follow_up(current_time=current_time, response_text=response.response_text)
            else:
                self._deactivate_follow_up()

    def _dispatch_follow_up_if_active(
        self,
        result: transcription_pb2.TranscriptionResult,
    ) -> None:
        if not self._is_follow_up_active(current_time=result.end_time):
            return

        if not self._speaker_allowed_for_follow_up(result.speaker_label):
            log.info(
                "transcription.follow_up_ignored_speaker",
                speaker_label=result.speaker_label,
            )
            return

        text = result.text.strip()
        if not text:
            return

        utterance_id = result.utterance_id.strip()
        if utterance_id and utterance_id == self._last_follow_up_utterance_id:
            return

        response = self._command_client.send_command(text, is_follow_up=True)
        if self._throbber_client is not None:
            self._throbber_client.hide()
        if utterance_id:
            self._last_follow_up_utterance_id = utterance_id

        if response is not None and response.success and response.continue_follow_up:
            self._activate_follow_up(
                current_time=result.end_time,
                response_text=response.response_text,
            )
            return

        self._deactivate_follow_up()

    def _activate_follow_up(self, *, current_time: float, response_text: str) -> None:
        duration = self._estimate_follow_up_window_seconds(response_text)
        follow_up_until = current_time + duration
        if self._follow_up_until is None:
            self._follow_up_until = follow_up_until
        else:
            self._follow_up_until = max(self._follow_up_until, follow_up_until)
        log.info("transcription.follow_up_active", until=self._follow_up_until)

    def _deactivate_follow_up(self, *, clear_speakers: bool = True) -> None:
        if self._follow_up_until is not None:
            log.info("transcription.follow_up_ended")
        self._follow_up_until = None
        self._last_follow_up_utterance_id = None
        if clear_speakers:
            self._follow_up_speakers.clear()

    def _is_follow_up_active(self, *, current_time: float) -> bool:
        if self._follow_up_until is None:
            return False
        if current_time <= self._follow_up_until:
            return True
        self._deactivate_follow_up()
        return False

    @staticmethod
    def _estimate_follow_up_window_seconds(response_text: str) -> float:
        words = re.findall(r"\w+", response_text)
        speech_seconds = len(words) / _FOLLOW_UP_WORDS_PER_SECOND if words else 0.0
        window = speech_seconds + _FOLLOW_UP_BUFFER_SECONDS
        return min(
            _FOLLOW_UP_MAX_WINDOW_SECONDS,
            max(_FOLLOW_UP_BASE_WINDOW_SECONDS, window),
        )

    def _snippet_for_trigger(
        self,
        trigger: PendingTrigger,
        *,
        partial_span: TranscriptSpan | None,
    ) -> str:
        parts = [
            span.text
            for span in self._history
            if span.end_time > trigger.capture_start_time
            and span.start_time < trigger.capture_end_time
            and self._speaker_matches_trigger(
                span_speaker=span.speaker_label,
                trigger_speaker=trigger.speaker_label,
            )
        ]

        if (
            partial_span is not None
            and partial_span.end_time > trigger.capture_start_time
            and partial_span.start_time < trigger.capture_end_time
            and self._speaker_matches_trigger(
                span_speaker=partial_span.speaker_label,
                trigger_speaker=trigger.speaker_label,
            )
        ):
            parts.append(partial_span.text)

        text = " ".join(part.strip() for part in parts if part.strip()).strip()
        return self._trim_to_last_trigger_phrase(text)

    @staticmethod
    def _trim_to_last_trigger_phrase(text: str) -> str:
        if not text:
            return ""

        matches = list(
            re.finditer(r"\b(?:hey|hi)\W+cl(?:eo|io)\b", text, flags=re.IGNORECASE)
        )
        if not matches:
            return text.strip()

        trimmed = text[matches[-1].start() :].strip()
        return trimmed or text.strip()

    def _speaker_allowed_for_follow_up(self, speaker_label: str) -> bool:
        if not self._follow_up_speakers:
            return True

        label = speaker_label.strip()
        if not label:
            return True
        return label in self._follow_up_speakers

    def _speaker_for_last_trigger_phrase(
        self,
        result: transcription_pb2.TranscriptionResult,
    ) -> str:
        trigger_speaker = ""
        for turn in result.speaker_turns:
            normalized = normalize_for_trigger_match(turn.text)
            if any(phrase in normalized for phrase in self._trigger_phrases):
                trigger_speaker = turn.speaker_label.strip()
        if trigger_speaker:
            return trigger_speaker
        return result.speaker_label.strip()

    @staticmethod
    def _speaker_matches_trigger(*, span_speaker: str, trigger_speaker: str) -> bool:
        if not trigger_speaker:
            return True
        if not span_speaker:
            return True
        return span_speaker == trigger_speaker

    def _speaker_handoff_ready(
        self,
        *,
        trigger: PendingTrigger,
        current_time: float,
        current_speaker_label: str,
    ) -> bool:
        trigger_speaker = trigger.speaker_label.strip()
        if not trigger_speaker:
            return False
        if current_time < trigger.trigger_time + self._speaker_handoff_seconds:
            return False

        current_speaker = current_speaker_label.strip()
        if not current_speaker or current_speaker == trigger_speaker:
            return False

        return any(
            span.end_time > trigger.capture_start_time
            and span.start_time < trigger.capture_end_time
            and span.speaker_label == trigger_speaker
            and bool(span.text.strip())
            for span in self._history
        )


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

    @staticmethod
    def _speaker_turns_from_alternative(
        alternative,
        *,
        stream_start_epoch: float,
        result_start_offset: float,
        result_end_offset: float,
    ) -> list[transcription_pb2.SpeakerTurn]:
        turns: list[transcription_pb2.SpeakerTurn] = []
        items = getattr(alternative, "items", []) or []

        fallback_start_time = stream_start_epoch + max(0.0, result_start_offset)
        fallback_end_time = stream_start_epoch + max(result_start_offset, result_end_offset)

        for item in items:
            item_type = str(getattr(item, "item_type", "") or "")
            content = str(getattr(item, "content", "") or "").strip()
            if not content:
                continue

            if item_type == "punctuation":
                if turns:
                    turns[-1].text = f"{turns[-1].text}{content}"
                continue
            if item_type and item_type != "pronunciation":
                continue

            speaker_label = str(getattr(item, "speaker", "") or "").strip()
            start_offset = float(getattr(item, "start_time", 0.0) or 0.0)
            end_offset = float(getattr(item, "end_time", 0.0) or start_offset)
            start_time = (
                stream_start_epoch + start_offset
                if start_offset > 0.0
                else fallback_start_time
            )
            end_time = (
                stream_start_epoch + end_offset
                if end_offset > 0.0
                else max(start_time, fallback_end_time)
            )

            if turns and turns[-1].speaker_label == speaker_label:
                turns[-1].text = f"{turns[-1].text} {content}".strip()
                turns[-1].end_time = max(turns[-1].end_time, end_time)
                continue

            turns.append(
                transcription_pb2.SpeakerTurn(
                    speaker_label=speaker_label,
                    text=content,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

        return turns

    @staticmethod
    def _dominant_speaker_label(
        speaker_turns: list[transcription_pb2.SpeakerTurn],
    ) -> str:
        if not speaker_turns:
            return ""

        speaker_weights: dict[str, float] = {}
        for turn in speaker_turns:
            label = turn.speaker_label.strip()
            if not label:
                continue
            duration = max(0.0, turn.end_time - turn.start_time)
            if duration == 0.0:
                duration = max(1, len(re.findall(r"\w+", turn.text)))
            speaker_weights[label] = speaker_weights.get(label, 0.0) + float(duration)

        if not speaker_weights:
            return ""
        return max(speaker_weights.items(), key=lambda item: item[1])[0]

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
            show_speaker_label=True,
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
                    alternative = alternatives[0]
                    text = getattr(alternative, "transcript", "").strip()
                    if not text:
                        continue

                    result_start_offset = float(getattr(result, "start_time", 0.0) or 0.0)
                    result_end_offset = float(getattr(result, "end_time", 0.0) or 0.0)
                    speaker_turns = AmazonTranscribeBackend._speaker_turns_from_alternative(
                        alternative,
                        stream_start_epoch=stream_start_epoch,
                        result_start_offset=result_start_offset,
                        result_end_offset=result_end_offset,
                    )
                    dominant_speaker = AmazonTranscribeBackend._dominant_speaker_label(
                        speaker_turns
                    )

                    output_queue.put(
                        transcription_pb2.TranscriptionResult(
                            text=text,
                            confidence=1.0,
                            start_time=stream_start_epoch
                            + result_start_offset,
                            end_time=stream_start_epoch
                            + result_end_offset,
                            is_partial=bool(getattr(result, "is_partial", False)),
                            utterance_id=str(getattr(result, "result_id", "") or ""),
                            speaker_label=dominant_speaker,
                            speaker_turns=speaker_turns,
                        )
                    )

        async def _send_audio() -> None:
            current = first_request
            while True:
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

        speaker_turns: list[transcription_pb2.SpeakerTurn] = []
        for segment in segments:
            for turn in segment.speaker_turns:
                turn_text = turn.text.strip()
                if not turn_text:
                    continue
                if (
                    speaker_turns
                    and speaker_turns[-1].speaker_label == turn.speaker_label
                    and turn.start_time <= speaker_turns[-1].end_time + 0.35
                ):
                    speaker_turns[-1].text = (
                        f"{speaker_turns[-1].text} {turn_text}".strip()
                    )
                    speaker_turns[-1].end_time = max(
                        speaker_turns[-1].end_time,
                        turn.end_time,
                    )
                    continue
                speaker_turns.append(
                    transcription_pb2.SpeakerTurn(
                        speaker_label=turn.speaker_label,
                        text=turn_text,
                        start_time=turn.start_time,
                        end_time=turn.end_time,
                    )
                )

        start_time = min(segment.start_time for segment in segments)
        end_time = max(segment.end_time for segment in segments)
        confidence = sum(segment.confidence for segment in segments) / len(segments)
        speaker_label = AmazonTranscribeBackend._dominant_speaker_label(speaker_turns)

        return transcription_pb2.TranscriptionResult(
            text=text,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            is_partial=False,
            utterance_id=segments[-1].utterance_id,
            speaker_label=speaker_label,
            speaker_turns=speaker_turns,
        )


class TranscriptionServiceServicer(transcription_pb2_grpc.TranscriptionServiceServicer):
    """gRPC transcription service backed by Amazon Transcribe Streaming."""

    def __init__(
        self,
        *,
        backend: AmazonTranscribeBackend | None = None,
        command_client: AssistantCommandClient | None = None,
        on_trigger_detected: Callable[[transcription_pb2.TranscriptionResult], None] | None = None,
        throbber_client: FrontendThrobberClient | None = None,
    ):
        self._backend = backend or AmazonTranscribeBackend()
        self._command_client = command_client or AssistantCommandClient()
        self._on_trigger_detected = on_trigger_detected
        self._throbber_client = throbber_client

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
                throbber_client=self._throbber_client,
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
            throbber_client=self._throbber_client,
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
    throbber_client = FrontendThrobberClient(address=FRONTEND_ADDRESS)
    servicer = TranscriptionServiceServicer(
        command_client=command_client,
        throbber_client=throbber_client,
    )
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
        throbber_client.close()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
