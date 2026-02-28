"""Tests for the recording tool service."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from apps.recording import RecordingSegment, RecordingServicer
from generated import data_pb2


def _make_servicer() -> RecordingServicer:
    servicer = RecordingServicer.__new__(RecordingServicer)
    servicer._data_address = "localhost:50053"
    servicer._lock = threading.Lock()
    servicer._session_start = None
    servicer._data_client = MagicMock()
    servicer._frontend_channel = MagicMock()
    servicer._frontend = MagicMock()
    return servicer


class TestRecordingProperties:
    def test_tool_name(self):
        servicer = RecordingServicer.__new__(RecordingServicer)
        assert servicer.tool_name == "recording"

    def test_tool_type_is_active(self):
        servicer = RecordingServicer.__new__(RecordingServicer)
        assert servicer.tool_type == "active"

    def test_tool_description_not_empty(self):
        servicer = RecordingServicer.__new__(RecordingServicer)
        assert len(servicer.tool_description) > 0

    def test_tool_input_schema_has_action(self):
        servicer = RecordingServicer.__new__(RecordingServicer)
        schema = servicer.tool_input_schema
        assert schema["type"] == "object"
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["start", "stop"]
        assert "action" in schema["required"]


class TestRecordingActions:
    def test_resolve_action_explicit(self):
        servicer = _make_servicer()
        assert servicer._resolve_action({"action": "start"}) == "start"
        assert servicer._resolve_action({"action": "stop"}) == "stop"

    def test_resolve_action_from_query(self):
        servicer = _make_servicer()
        assert servicer._resolve_action({"query": "start recording"}) == "start"
        assert servicer._resolve_action({"query": "stop recording"}) == "stop"

    def test_start_sets_indicator_and_session(self):
        servicer = _make_servicer()

        success, msg = servicer.execute({"action": "start"})

        assert success
        assert "started" in msg.lower()
        assert servicer._session_start is not None
        servicer._frontend.SetAppIndicator.assert_called_once()
        servicer._frontend.ShowNotification.assert_called_once()

    def test_start_when_already_recording(self):
        servicer = _make_servicer()
        servicer._session_start = 1000.0

        success, msg = servicer.execute({"action": "start"})

        assert success
        assert "already" in msg.lower()

    def test_stop_when_not_recording(self):
        servicer = _make_servicer()

        success, msg = servicer.execute({"action": "stop"})

        assert not success
        assert "no recording" in msg.lower()

    def test_stop_starts_finalize_thread(self):
        servicer = _make_servicer()
        servicer._session_start = 1000.0

        with patch("threading.Thread") as mock_thread:
            success, msg = servicer.execute({"action": "stop"})

        assert success
        assert "stopped" in msg.lower()
        assert servicer._session_start is None
        mock_thread.assert_called_once()
        servicer._frontend.SetAppIndicator.assert_called_once()

    def test_invalid_action(self):
        servicer = _make_servicer()

        success, msg = servicer.execute({"action": "invalid"})

        assert not success
        assert "must be" in msg.lower()


class TestRecordingFinalize:
    def test_finalize_recording_impl_success(self):
        servicer = _make_servicer()
        segments = [
            RecordingSegment(
                clip_id=11,
                start_timestamp=1000.0,
                end_timestamp=1005.0,
                num_frames=150,
                mp4_data=b"clip-bytes",
            )
        ]

        servicer._data_client.store_clip.return_value = data_pb2.StoreVideoClipResponse(
            clip_id=42,
            faiss_id=7,
        )
        servicer._load_segments_in_range = MagicMock(return_value=segments)
        servicer._compose_recording_mp4 = MagicMock(return_value=b"recording-mp4")
        servicer._store_recording_metadata = MagicMock()

        servicer._finalize_recording_impl(1000.0, 1005.0)

        servicer._data_client.store_clip.assert_called_once()
        servicer._store_recording_metadata.assert_called_once_with(42, 1000.0, 1005.0)
        servicer._frontend.ShowNotification.assert_called_once()

    def test_finalize_recording_impl_no_segments(self):
        servicer = _make_servicer()
        servicer._load_segments_in_range = MagicMock(return_value=[])

        with pytest.raises(RuntimeError, match="No video clips"):
            servicer._finalize_recording_impl(1000.0, 1005.0)

    def test_finalize_recording_impl_store_failure(self):
        servicer = _make_servicer()
        segments = [
            RecordingSegment(
                clip_id=11,
                start_timestamp=1000.0,
                end_timestamp=1005.0,
                num_frames=150,
                mp4_data=b"clip-bytes",
            )
        ]
        servicer._load_segments_in_range = MagicMock(return_value=segments)
        servicer._compose_recording_mp4 = MagicMock(return_value=b"recording-mp4")
        servicer._data_client.store_clip.return_value = None

        with pytest.raises(RuntimeError, match="Failed to store"):
            servicer._finalize_recording_impl(1000.0, 1005.0)


class TestRecordingComposition:
    def test_compose_returns_empty_without_segments(self):
        servicer = _make_servicer()
        assert servicer._compose_recording_mp4([], 1000.0, 1005.0) == b""

    def test_compose_trims_when_window_is_inside_clip(self):
        servicer = _make_servicer()
        segments = [
            RecordingSegment(
                clip_id=11,
                start_timestamp=1000.0,
                end_timestamp=1030.0,
                num_frames=900,
                mp4_data=b"clip-bytes",
            )
        ]
        servicer._concat_mp4_segments = MagicMock(return_value=b"combined")
        servicer._trim_mp4_window = MagicMock(return_value=b"trimmed")

        result = servicer._compose_recording_mp4(segments, 1005.0, 1015.0)

        assert result == b"trimmed"
        servicer._trim_mp4_window.assert_called_once()

    def test_compose_skips_trim_when_window_matches_clip(self):
        servicer = _make_servicer()
        segments = [
            RecordingSegment(
                clip_id=11,
                start_timestamp=1000.0,
                end_timestamp=1010.0,
                num_frames=300,
                mp4_data=b"clip-bytes",
            )
        ]
        servicer._concat_mp4_segments = MagicMock(return_value=b"combined")
        servicer._trim_mp4_window = MagicMock(return_value=b"trimmed")

        result = servicer._compose_recording_mp4(segments, 1000.0, 1010.0)

        assert result == b"combined"
        servicer._trim_mp4_window.assert_not_called()

    def test_build_embed_source_trims_when_duration_exceeds_limit(self):
        servicer = _make_servicer()
        servicer._trim_mp4_window = MagicMock(return_value=b"embed-trimmed")

        result = servicer._build_embed_source(b"full-mp4", duration_seconds=45.0)

        assert result == b"embed-trimmed"
        servicer._trim_mp4_window.assert_called_once()

    def test_build_embed_source_keeps_short_recording(self):
        servicer = _make_servicer()
        servicer._trim_mp4_window = MagicMock(return_value=b"unused")

        result = servicer._build_embed_source(b"full-mp4", duration_seconds=10.0)

        assert result == b"full-mp4"
        servicer._trim_mp4_window.assert_not_called()


class TestReadClipBytes:
    def test_read_clip_bytes_rejects_out_of_order_chunks(self):
        stub = MagicMock()
        stub.GetVideoClip.return_value = iter(
            [
                SimpleNamespace(chunk_index=0, data=b"a", is_last=False),
                SimpleNamespace(chunk_index=2, data=b"b", is_last=True),
            ]
        )

        with pytest.raises(RuntimeError, match="out-of-order"):
            RecordingServicer._read_clip_bytes(stub, clip_id=9)


class TestClipWait:
    def test_wait_for_overlapping_clips_polls_until_stop_is_covered(self):
        servicer = _make_servicer()
        stub = MagicMock()

        first = SimpleNamespace(clips=[])
        second = SimpleNamespace(
            clips=[
                SimpleNamespace(
                    clip_id=77,
                    start_timestamp=1000.0,
                    end_timestamp=1012.0,
                    num_frames=360,
                )
            ]
        )
        stub.GetVideoClipsInRange.side_effect = [first, second]

        with patch("apps.recording.time.sleep", return_value=None):
            clips = servicer._wait_for_overlapping_clips(
                stub,
                start_timestamp=1002.0,
                stop_timestamp=1010.0,
            )

        assert len(clips) == 1
        assert clips[0].clip_id == 77
        assert stub.GetVideoClipsInRange.call_count == 2
