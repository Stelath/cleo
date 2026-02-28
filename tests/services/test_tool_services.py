"""Tests for tool services — base class and individual tool implementations."""

import json
from unittest.mock import MagicMock

import pytest

from generated import tool_pb2
from apps.color_blind import ColorBlindnessServicer
from apps.notetaking import NotetakingServicer
from apps.tool_base import ToolServiceBase

class DummyTool(ToolServiceBase):
    @property
    def tool_name(self) -> str:
        return "dummy_tool"

    @property
    def tool_description(self) -> str:
        return "Dummy description"

    @property
    def tool_input_schema(self) -> dict:
        return {"type": "object"}

    def execute(self, params: dict) -> tuple[bool, str]:
        return True, "mocked executed"


class TestToolServiceBase:
    def test_wrong_tool_name_rejected(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="wrong_name",
            parameters_json=json.dumps({"query": "test"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        mock_grpc_context.set_code.assert_called()

    def test_invalid_json_rejected(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="dummy_tool",
            parameters_json="{bad json",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Invalid parameters" in response.result_text

    def test_empty_params_json_is_ok(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="dummy_tool",
            parameters_json="",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ToolServiceBase()

    def test_execute_exception_caught(self, mock_grpc_context):
        """If execute() raises, base class catches it and returns an error response."""

        class BrokenTool(ToolServiceBase):
            @property
            def tool_name(self) -> str:
                return "broken"

            @property
            def tool_description(self) -> str:
                return "A broken tool"

            @property
            def tool_input_schema(self) -> dict:
                return {"type": "object"}

            def execute(self, params: dict) -> tuple[bool, str]:
                raise RuntimeError("something went wrong")

        servicer = BrokenTool()
        request = tool_pb2.ToolRequest(tool_name="broken", parameters_json="{}")
        response = servicer.Execute(request, mock_grpc_context)

        assert not response.success
        assert "something went wrong" in response.result_text
        mock_grpc_context.set_code.assert_called()

    def test_wrong_tool_sets_invalid_argument_status(self, mock_grpc_context):
        import grpc

        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="wrong",
            parameters_json="{}",
        )
        servicer.Execute(request, mock_grpc_context)
        mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


class TestToolServiceProperties:
    """Verify tool_description, tool_input_schema, and tool_type on all servicers."""

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_description_is_nonempty(self, servicer_cls):
        servicer = servicer_cls()
        assert isinstance(servicer.tool_description, str)
        assert len(servicer.tool_description) > 0

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_input_schema_has_type(self, servicer_cls):
        servicer = servicer_cls()
        schema = servicer.tool_input_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_type_defaults_to_on_demand(self, servicer_cls):
        servicer = servicer_cls()
        assert servicer.tool_type == "on_demand"
class TestNotetakingServicer:
    def test_start_and_stop_flow(self):
        import threading

        mock_data = MagicMock()
        mock_data.GetTranscriptionsInRange.return_value = MagicMock(entries=[])
        mock_data.GetVideoClipsInRange.return_value = MagicMock(clips=[])
        mock_data.StoreNoteSummary.return_value = MagicMock(id=1)
        mock_bedrock = MagicMock()

        servicer = NotetakingServicer.__new__(NotetakingServicer)
        servicer._bedrock = mock_bedrock
        servicer._data = mock_data
        servicer._channel = MagicMock()
        servicer._frontend = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._lock = threading.Lock()
        servicer._session_start = None

        start_response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "start"}),
            ),
            MagicMock(),
        )
        assert start_response.success
        assert "Started notetaking" in start_response.result_text

        stop_response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "stop"}),
            ),
            MagicMock(),
        )
        assert stop_response.success
        assert "No transcript or video activity" in stop_response.result_text
        mock_data.StoreNoteSummary.assert_called_once()
        assert servicer._frontend.ShowNotification.call_count == 2
        start_notification = servicer._frontend.ShowNotification.call_args_list[0].args[0]
        saved_notification = servicer._frontend.ShowNotification.call_args_list[1].args[0]
        assert start_notification.title == "Notetaking started"
        assert start_notification.message == "Capturing notes until you stop the session."
        assert start_notification.style == "info"
        assert saved_notification.title == "Note saved"
        assert saved_notification.message == "Saved note summary #1"
        assert saved_notification.style == "success"

    def test_stop_requires_active_session(self, mock_grpc_context):
        import threading

        servicer = NotetakingServicer.__new__(NotetakingServicer)
        servicer._bedrock = MagicMock()
        servicer._data = MagicMock()
        servicer._channel = MagicMock()
        servicer._frontend = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._lock = threading.Lock()
        servicer._session_start = None

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "stop"}),
            ),
            mock_grpc_context,
        )
        assert not response.success
        assert response.result_text == "Notetaking is not active"

    def test_tool_name(self):
        servicer = NotetakingServicer.__new__(NotetakingServicer)
        assert servicer.tool_name == "notetaking"
