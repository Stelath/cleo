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
    DataClient,
    FrontendTranscriptDebugClient,
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


@patch("services.transcription.service.data_pb2_grpc.DataServiceStub")
@patch("services.transcription.service.grpc.insecure_channel")
def test_data_client_stores_diarized_turns_with_generic_speaker_tags(
    mock_channel_cls,
    mock_stub_cls,
):
    mock_channel_cls.return_value = MagicMock()
    stub = MagicMock()
    mock_stub_cls.return_value = stub

    client = DataClient(address="localhost:50053")
    result = transcription_pb2.TranscriptionResult(
        text="ignored aggregate",
        confidence=0.91,
        start_time=10.0,
        end_time=12.0,
        speaker_turns=[
            transcription_pb2.SpeakerTurn(
                speaker_label="spk_9",
                text="first",
                start_time=10.1,
                end_time=10.3,
            ),
            transcription_pb2.SpeakerTurn(
                speaker_label="spk_2",
                text="second",
                start_time=10.4,
                end_time=10.6,
            ),
            transcription_pb2.SpeakerTurn(
                speaker_label="spk_9",
                text="third",
                start_time=10.7,
                end_time=11.0,
            ),
        ],
    )

    client.store_transcription(result)

    assert stub.StoreTranscription.call_count == 3
    requests = [call.args[0] for call in stub.StoreTranscription.call_args_list]
    assert [req.text for req in requests] == [
        "Speaker 1: first",
        "Speaker 2: second",
        "Speaker 1: third",
    ]
    assert [req.start_time for req in requests] == [10.1, 10.4, 10.7]
    assert [req.end_time for req in requests] == [10.3, 10.6, 11.0]


@patch("services.transcription.service.data_pb2_grpc.DataServiceStub")
@patch("services.transcription.service.grpc.insecure_channel")
def test_data_client_keeps_speaker_aliases_across_results(
    mock_channel_cls,
    mock_stub_cls,
):
    mock_channel_cls.return_value = MagicMock()
    stub = MagicMock()
    mock_stub_cls.return_value = stub

    client = DataClient(address="localhost:50053")
    client.store_transcription(
        transcription_pb2.TranscriptionResult(
            text="",
            confidence=0.9,
            start_time=1.0,
            end_time=2.0,
            speaker_turns=[
                transcription_pb2.SpeakerTurn(
                    speaker_label="spk_0",
                    text="hello",
                    start_time=1.1,
                    end_time=1.3,
                )
            ],
        )
    )
    client.store_transcription(
        transcription_pb2.TranscriptionResult(
            text="",
            confidence=0.9,
            start_time=2.0,
            end_time=3.0,
            speaker_turns=[
                transcription_pb2.SpeakerTurn(
                    speaker_label="spk_1",
                    text="world",
                    start_time=2.1,
                    end_time=2.3,
                ),
                transcription_pb2.SpeakerTurn(
                    speaker_label="spk_0",
                    text="again",
                    start_time=2.4,
                    end_time=2.7,
                ),
            ],
        )
    )

    texts = [
        call.args[0].text for call in stub.StoreTranscription.call_args_list
    ]
    assert texts == [
        "Speaker 1: hello",
        "Speaker 2: world",
        "Speaker 1: again",
    ]


class _RecordingCommandClient:
    def __init__(self, responses: list[assistant_pb2.CommandResponse] | None = None):
        self.calls: list[str] = []
        self.call_details: list[tuple[str, bool]] = []
        self._responses = list(responses or [])

    def send_command(self, text: str, *, is_follow_up: bool = False):
        self.calls.append(text)
        self.call_details.append((text, is_follow_up))
        if self._responses:
            return self._responses.pop(0)
        return None


def _result(
    *,
    text: str,
    start_time: float,
    end_time: float,
    is_partial: bool,
    utterance_id: str = "",
    speaker_label: str = "",
    speaker_turns: list[transcription_pb2.SpeakerTurn] | None = None,
) -> transcription_pb2.TranscriptionResult:
    return transcription_pb2.TranscriptionResult(
        text=text,
        confidence=1.0,
        start_time=start_time,
        end_time=end_time,
        is_partial=is_partial,
        utterance_id=utterance_id,
        speaker_label=speaker_label,
        speaker_turns=speaker_turns or [],
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


def test_follow_up_continues_without_wake_word_then_stops_when_unrelated():
    responses = [
        assistant_pb2.CommandResponse(
            success=True,
            response_text="Sure, here's the answer.",
            tool_name="",
            responded=True,
            continue_follow_up=True,
        ),
        assistant_pb2.CommandResponse(
            success=True,
            response_text="",
            tool_name="",
            responded=False,
            continue_follow_up=False,
        ),
    ]
    client = _RecordingCommandClient(responses=responses)
    router = TriggerRouter(
        client,
        capture_seconds=0.1,
        preroll_seconds=0.5,
        final_flush_grace_seconds=0.0,
    )

    router.observe(
        _result(
            text="Hey Cleo how many calories are in this",
            start_time=10.0,
            end_time=10.4,
            is_partial=False,
            utterance_id="u1",
        )
    )
    router.observe(
        _result(
            text="thanks",
            start_time=10.6,
            end_time=10.8,
            is_partial=False,
            utterance_id="u2",
        )
    )
    router.observe(
        _result(
            text="what about protein",
            start_time=11.0,
            end_time=11.2,
            is_partial=False,
            utterance_id="u3",
        )
    )
    router.observe(
        _result(
            text="random side chatter",
            start_time=11.4,
            end_time=11.6,
            is_partial=False,
            utterance_id="u4",
        )
    )

    assert client.call_details == [
        ("Hey Cleo how many calories are in this", False),
        ("what about protein", True),
    ]


def test_trigger_router_dispatches_on_speaker_handoff_without_waiting_for_other_finals():
    responses = [
        assistant_pb2.CommandResponse(
            success=True,
            response_text="Got it.",
            tool_name="",
            responded=True,
            continue_follow_up=True,
        ),
    ]
    client = _RecordingCommandClient(responses=responses)
    router = TriggerRouter(
        client,
        capture_seconds=3.0,
        preroll_seconds=0.5,
        early_final_seconds=10.0,
        final_flush_grace_seconds=999.0,
        speaker_handoff_seconds=0.45,
    )

    router.observe(
        _result(
            text="Hey Cleo what are my macros",
            start_time=10.0,
            end_time=10.4,
            is_partial=False,
            utterance_id="u1",
            speaker_label="spk_0",
        )
    )
    assert client.calls == []

    router.observe(
        _result(
            text="background chatter still going",
            start_time=10.5,
            end_time=10.95,
            is_partial=True,
            utterance_id="u2",
            speaker_label="spk_1",
        )
    )

    assert len(client.calls) == 1
    assert client.calls[0].lower().startswith("hey cleo")


def test_transcribe_backend_extracts_diarized_turns():
    class _Item:
        def __init__(self, item_type, content, speaker, start_time, end_time):
            self.item_type = item_type
            self.content = content
            self.speaker = speaker
            self.start_time = start_time
            self.end_time = end_time

    class _Alternative:
        def __init__(self, items):
            self.items = items

    alt = _Alternative(
        [
            _Item("pronunciation", "Hey", "spk_0", 1.00, 1.12),
            _Item("pronunciation", "Cleo", "spk_0", 1.13, 1.32),
            _Item("punctuation", ",", "", 0.0, 0.0),
            _Item("pronunciation", "hello", "spk_1", 1.50, 1.65),
        ]
    )

    turns = AmazonTranscribeBackend._speaker_turns_from_alternative(
        alt,
        stream_start_epoch=100.0,
        result_start_offset=1.0,
        result_end_offset=1.7,
    )

    assert len(turns) == 2
    assert turns[0].speaker_label == "spk_0"
    assert turns[0].text == "Hey Cleo,"
    assert turns[0].start_time == 101.0
    assert turns[1].speaker_label == "spk_1"
    assert turns[1].text == "hello"


def test_trigger_router_ignores_non_invoking_speaker_follow_up():
    responses = [
        assistant_pb2.CommandResponse(
            success=True,
            response_text="Sure.",
            tool_name="",
            responded=True,
            continue_follow_up=True,
        ),
        assistant_pb2.CommandResponse(
            success=True,
            response_text="Protein is around 12g.",
            tool_name="",
            responded=True,
            continue_follow_up=True,
        ),
    ]
    client = _RecordingCommandClient(responses=responses)
    router = TriggerRouter(
        client,
        capture_seconds=0.1,
        preroll_seconds=0.5,
        final_flush_grace_seconds=0.0,
    )

    router.observe(
        _result(
            text="Hey Cleo how many calories are in this",
            start_time=10.0,
            end_time=10.4,
            is_partial=False,
            utterance_id="u1",
            speaker_label="spk_0",
        )
    )
    router.observe(
        _result(
            text="thanks",
            start_time=10.6,
            end_time=10.8,
            is_partial=False,
            utterance_id="u2",
            speaker_label="spk_0",
        )
    )

    router.observe(
        _result(
            text="you should ask about sodium",
            start_time=11.0,
            end_time=11.2,
            is_partial=False,
            utterance_id="u3",
            speaker_label="spk_1",
        )
    )
    router.observe(
        _result(
            text="what about protein",
            start_time=11.4,
            end_time=11.6,
            is_partial=False,
            utterance_id="u4",
            speaker_label="spk_0",
        )
    )

    assert len(client.call_details) == 2
    assert client.call_details[0][1] is False
    assert client.call_details[1] == ("what about protein", True)


def test_debug_client_formats_speaker_lines_with_s_tags():
    client = FrontendTranscriptDebugClient()

    result = transcription_pb2.TranscriptionResult(
        text="ignored because turns are present",
        speaker_turns=[
            transcription_pb2.SpeakerTurn(speaker_label="spk_7", text="first line"),
            transcription_pb2.SpeakerTurn(speaker_label="spk_9", text="second line"),
        ],
    )
    lines = client._format_debug_lines(result)

    assert lines == ["S1: first line", "S2: second line"]

    later = transcription_pb2.TranscriptionResult(
        text="follow up line",
        speaker_label="spk_9",
    )
    later_lines = client._format_debug_lines(later)
    assert later_lines == ["S2: follow up line"]
