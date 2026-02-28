"""Unit tests for transcription service timestamp handling."""

from unittest.mock import MagicMock, patch

import numpy as np

from generated import assistant_pb2
from generated import sensor_pb2
from generated import transcription_pb2
from services.transcription.service import (
    _ASSISTANT_RESPONSE_LOG_MAX_CHARS,
    AmazonTranscribeBackend,
    AssistantCommandClient,
    SensorTranscriptionPipeline,
    TriggerRouter,
)


def test_stream_start_epoch_uses_chunk_timestamp():
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()
    request = transcription_pb2.AudioInput(
        audio_data=audio,
        sample_rate=4,
        timestamp=100.0,
    )

    stream_start = AmazonTranscribeBackend._stream_start_epoch(request)

    assert stream_start == 99.5


def test_stream_start_epoch_falls_back_to_current_time(monkeypatch):
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()
    request = transcription_pb2.AudioInput(
        audio_data=audio,
        sample_rate=4,
    )
    monkeypatch.setattr("services.transcription.service.time.time", lambda: 500.0)

    stream_start = AmazonTranscribeBackend._stream_start_epoch(request)

    assert stream_start == 499.5


def test_sensor_audio_requests_forward_chunk_timestamps():
    servicer = object()
    pipeline = SensorTranscriptionPipeline(servicer)
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()

    class Stub:
        @staticmethod
        def StreamAudio(_request):
            yield sensor_pb2.AudioChunk(
                data=audio,
                sample_rate=4,
                num_samples=2,
                timestamp=123.4,
            )

    requests = list(pipeline._sensor_audio_requests(Stub()))

    assert requests[0].timestamp == 123.4
    assert requests[1].timestamp > 0.0
    assert requests[1].is_final is True


@patch("services.transcription.service.log")
@patch("services.transcription.service.assistant_pb2_grpc.AssistantServiceStub")
@patch("services.transcription.service.grpc.insecure_channel")
def test_assistant_command_logs_response_text(
    mock_channel_cls,
    mock_stub_cls,
    mock_log,
):
    mock_channel_cls.return_value = MagicMock()
    stub = MagicMock()
    stub.ProcessCommand.return_value = assistant_pb2.CommandResponse(
        success=False,
        response_text="Assistant error: Bedrock unavailable",
        tool_name="",
    )
    mock_stub_cls.return_value = stub

    client = AssistantCommandClient(address="localhost:50054")
    client.send_command("Hey, Cleo, start taking notes")

    mock_log.info.assert_any_call(
        "transcription.assistant_invoked",
        success=False,
        tool_name="",
        response_text="Assistant error: Bedrock unavailable",
    )


@patch("services.transcription.service.log")
@patch("services.transcription.service.assistant_pb2_grpc.AssistantServiceStub")
@patch("services.transcription.service.grpc.insecure_channel")
def test_assistant_command_truncates_long_response_text(
    mock_channel_cls,
    mock_stub_cls,
    mock_log,
):
    mock_channel_cls.return_value = MagicMock()
    stub = MagicMock()
    long_response = "x" * (_ASSISTANT_RESPONSE_LOG_MAX_CHARS + 20)
    stub.ProcessCommand.return_value = assistant_pb2.CommandResponse(
        success=False,
        response_text=long_response,
        tool_name="",
    )
    mock_stub_cls.return_value = stub

    client = AssistantCommandClient(address="localhost:50054")
    client.send_command("Hey, Cleo")

    expected = f"{'x' * (_ASSISTANT_RESPONSE_LOG_MAX_CHARS - 3)}..."
    mock_log.info.assert_any_call(
        "transcription.assistant_invoked",
        success=False,
        tool_name="",
        response_text=expected,
    )


class _RecordingCommandClient:
    def __init__(self):
        self.calls: list[str] = []

    def send_command(self, text: str) -> None:
        self.calls.append(text)


def _result(
    *,
    text: str,
    start_time: float,
    end_time: float,
    is_partial: bool,
    utterance_id: str = "",
) -> transcription_pb2.TranscriptionResult:
    return transcription_pb2.TranscriptionResult(
        text=text,
        confidence=1.0,
        start_time=start_time,
        end_time=end_time,
        is_partial=is_partial,
        utterance_id=utterance_id,
    )


def test_trigger_router_anchors_capture_to_detection_time():
    client = _RecordingCommandClient()
    router = TriggerRouter(
        client,
        capture_seconds=3.0,
        preroll_seconds=0.75,
        final_flush_grace_seconds=999.0,
    )

    router.observe(
        _result(
            text="lots of chatter before wake phrase hey cleo",
            start_time=100.0,
            end_time=105.0,
            is_partial=False,
            utterance_id="u1",
        )
    )
    assert client.calls == []

    router.observe(
        _result(
            text="how many calories is this",
            start_time=105.1,
            end_time=106.1,
            is_partial=False,
            utterance_id="u2",
        )
    )
    assert client.calls == []

    router.observe(
        _result(
            text="thanks",
            start_time=108.2,
            end_time=108.4,
            is_partial=False,
            utterance_id="u3",
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0].lower().startswith("hey cleo")
    assert "how many calories is this" in client.calls[0].lower()


def test_trigger_router_does_not_duplicate_final_span_text():
    client = _RecordingCommandClient()
    router = TriggerRouter(
        client,
        capture_seconds=0.1,
        preroll_seconds=0.1,
        final_flush_grace_seconds=0.0,
    )

    router.observe(
        _result(
            text="Hey, Cleo, test request",
            start_time=10.0,
            end_time=11.0,
            is_partial=False,
            utterance_id="u1",
        )
    )

    router.observe(
        _result(
            text="trailing",
            start_time=11.2,
            end_time=11.3,
            is_partial=False,
            utterance_id="u2",
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0].lower().count("hey") == 1
    assert client.calls[0].lower().count("test request") == 1


def test_trigger_router_waits_for_final_result_before_dispatch():
    client = _RecordingCommandClient()
    router = TriggerRouter(
        client,
        capture_seconds=1.0,
        preroll_seconds=0.0,
        final_flush_grace_seconds=999.0,
    )

    router.observe(
        _result(
            text="Hey Cleo",
            start_time=200.0,
            end_time=200.4,
            is_partial=True,
            utterance_id="u1",
        )
    )
    router.observe(
        _result(
            text="Hey Cleo how many calories is this",
            start_time=200.0,
            end_time=201.6,
            is_partial=True,
            utterance_id="u1",
        )
    )
    assert client.calls == []

    router.observe(
        _result(
            text="Hey Cleo how many calories is this",
            start_time=200.0,
            end_time=201.8,
            is_partial=False,
            utterance_id="u1",
        )
    )

    assert len(client.calls) == 1
    assert "how many calories" in client.calls[0].lower()


def test_trigger_router_trims_to_last_wake_phrase():
    client = _RecordingCommandClient()
    router = TriggerRouter(
        client,
        capture_seconds=0.1,
        preroll_seconds=0.1,
        final_flush_grace_seconds=0.0,
    )

    router.observe(
        _result(
            text="Hey Cleo first request. random chatter. Hey Cleo second request",
            start_time=1.0,
            end_time=2.0,
            is_partial=False,
            utterance_id="u1",
        )
    )

    router.observe(
        _result(
            text="done",
            start_time=2.2,
            end_time=2.3,
            is_partial=False,
            utterance_id="u2",
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0].lower().startswith("hey cleo second request")
