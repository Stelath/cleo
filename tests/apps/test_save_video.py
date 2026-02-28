"""Tests for the Save Video (bookmark) tool service."""

import time
from unittest.mock import MagicMock, patch

import pytest

from generated import data_pb2
from apps.save_video import SaveVideoServicer, _FORWARD_SECONDS


class TestSaveVideoProperties:
    """Verify tool metadata used for assistant registration."""

    def _servicer(self):
        s = SaveVideoServicer.__new__(SaveVideoServicer)
        return s

    def test_tool_name(self):
        assert self._servicer().tool_name == "save_video"

    def test_tool_type_is_on_demand(self):
        assert self._servicer().tool_type == "on_demand"

    def test_tool_description_contains_key_phrases(self):
        desc = self._servicer().tool_description.lower()
        assert "clip" in desc
        assert "save" in desc

    def test_tool_input_schema_is_empty_object(self):
        schema = self._servicer().tool_input_schema
        assert schema["type"] == "object"
        assert schema["properties"] == {}


class TestSaveVideoExecute:
    """Unit tests with fully mocked gRPC stubs."""

    def _make_servicer(self, store_response=None):
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        servicer._data_channel = MagicMock()
        servicer._data = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._frontend = MagicMock()

        if store_response is None:
            store_response = data_pb2.StoreSavedClipResponse(id=42)
        servicer._data.StoreSavedClip.return_value = store_response

        return servicer

    def test_execute_stores_bookmark(self):
        servicer = self._make_servicer()

        success, msg = servicer.execute({})

        assert success is True
        assert "bookmark #42" in msg

        # Verify StoreSavedClip was called exactly once
        servicer._data.StoreSavedClip.assert_called_once()
        req = servicer._data.StoreSavedClip.call_args[0][0]
        assert req.label == "user clip"
        assert req.start_timestamp > 0
        assert req.end_timestamp > req.start_timestamp

    def test_execute_timestamps_span_correct_window(self):
        servicer = self._make_servicer()
        before = time.time()

        servicer.execute({})

        after = time.time()
        req = servicer._data.StoreSavedClip.call_args[0][0]

        # start_ts should be ~30s before now
        from services.config import SENSOR_CAMERA_BUFFER_SECONDS as BUF
        assert req.start_timestamp >= before - float(BUF) - 1
        assert req.start_timestamp <= after - float(BUF) + 1

        # end_ts should be ~60s after now
        assert req.end_timestamp >= before + _FORWARD_SECONDS - 1
        assert req.end_timestamp <= after + _FORWARD_SECONDS + 1

    def test_execute_sends_notification(self):
        servicer = self._make_servicer()

        servicer.execute({})

        servicer._frontend.ShowNotification.assert_called_once()

    def test_execute_returns_duration_in_message(self):
        servicer = self._make_servicer()

        success, msg = servicer.execute({})

        assert "90s" in msg  # 30 past + 60 future

    def test_execute_handles_grpc_error(self):
        import grpc

        servicer = self._make_servicer()
        servicer._data.StoreSavedClip.side_effect = grpc.RpcError()

        success, msg = servicer.execute({})

        assert success is False
        assert "Failed" in msg

    def test_notification_failure_does_not_crash(self):
        servicer = self._make_servicer()
        servicer._frontend.ShowNotification.side_effect = Exception("nope")

        # Should still succeed — notification is best-effort
        success, msg = servicer.execute({})
        assert success is True
