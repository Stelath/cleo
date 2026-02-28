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
