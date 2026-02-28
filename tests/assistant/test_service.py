"""Tests for assistant.service — full flow with mocked Bedrock and tool gRPC calls."""

from unittest.mock import MagicMock, patch

import pytest

from services.assistant.bedrock import TextResult, ToolUseResult
from services.assistant.registry import ToolDefinition, ToolRegistry
from services.assistant.service import AssistantServiceServicer
from generated import assistant_pb2, frontend_pb2, tool_pb2


@pytest.fixture
def mock_bedrock():
    return MagicMock()


@pytest.fixture
def registry_with_tool():
    tool = ToolDefinition(
        name="color_blindness_assist",
        description="Color help",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        grpc_address="localhost:50060",
    )
    return ToolRegistry(tools=[tool])


@pytest.fixture
def servicer(registry_with_tool, mock_bedrock):
    return AssistantServiceServicer(
        registry=registry_with_tool,
        bedrock_client=mock_bedrock,
    )


@pytest.fixture(autouse=True)
def clear_assistant_debug_hud_env(monkeypatch):
    monkeypatch.delenv("CLEO_DEBUG_ASSISTANT_HUD", raising=False)


class TestAssistantService:
    def test_empty_command(self, servicer, mock_grpc_context):
        request = assistant_pb2.CommandRequest(text="")
        response = servicer.ProcessCommand(request, mock_grpc_context)
        assert not response.success
        assert response.response_text == "Empty command"

    def test_text_response_flow(self, servicer, mock_bedrock, mock_grpc_context):
        mock_bedrock.converse.return_value = TextResult(text="Hello from Cleo!")
        servicer._speak_response_text = MagicMock()
        request = assistant_pb2.CommandRequest(text="hello")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == "Hello from Cleo!"
        assert response.tool_name == ""
        assert response.responded is True
        assert response.continue_follow_up is True
        servicer._speak_response_text.assert_called_once_with("Hello from Cleo!")

    def test_follow_up_non_continuation_returns_no_response(
        self,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        mock_bedrock.classify_follow_up.return_value = False

        request = assistant_pb2.CommandRequest(
            text="yeah cool",
            is_follow_up=True,
        )
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == ""
        assert response.responded is False
        assert response.continue_follow_up is False
        mock_bedrock.classify_follow_up.assert_called_once_with("yeah cool")
        mock_bedrock.converse.assert_not_called()

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_use_flow(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        # Bedrock returns tool use
        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_1",
            tool_name="color_blindness_assist",
            parameters={"query": "what color is this shirt?"},
        )

        # Tool gRPC returns success
        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.return_value = tool_pb2.ToolResponse(
            success=True, result_text="The shirt is blue"
        )
        mock_tool_stub_cls.return_value = mock_tool_stub
        mock_frontend_stub = MagicMock()
        mock_frontend_stub_cls.return_value = mock_frontend_stub

        request = assistant_pb2.CommandRequest(text="help me with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == "The shirt is blue"
        assert response.tool_name == "color_blindness_assist"
        mock_frontend_stub.ShowNotification.assert_called_once_with(
            frontend_pb2.NotificationRequest(
                title="Tool called",
                message="color_blindness_assist",
                style="debug",
                duration_ms=2000,
            )
        )

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    def test_debug_hud_shows_llm_response_text(
        self,
        mock_frontend_stub_cls,
        mock_channel,
        registry_with_tool,
        mock_bedrock,
        mock_grpc_context,
        monkeypatch,
    ):
        monkeypatch.setenv("CLEO_DEBUG_ASSISTANT_HUD", "1")

        mock_bedrock.converse.return_value = TextResult(text="Hello from Cleo!")
        mock_frontend_stub = MagicMock()
        mock_frontend_stub_cls.return_value = mock_frontend_stub

        servicer = AssistantServiceServicer(
            registry=registry_with_tool,
            bedrock_client=mock_bedrock,
        )

        request = assistant_pb2.CommandRequest(text="hello")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        show_text_call = mock_frontend_stub.ShowText.call_args
        assert show_text_call is not None
        assert show_text_call.args[0].text == "LLM: Hello from Cleo!"
        assert show_text_call.kwargs["timeout"] == 2

    def test_unknown_tool_from_bedrock(self, servicer, mock_bedrock, mock_grpc_context):
        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_2",
            tool_name="nonexistent_tool",
            parameters={},
        )

        request = assistant_pb2.CommandRequest(text="do something weird")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert not response.success
        assert "Unknown tool" in response.response_text

    def test_bedrock_error_handled(self, servicer, mock_bedrock, mock_grpc_context):
        mock_bedrock.converse.side_effect = Exception("API timeout")

        request = assistant_pb2.CommandRequest(text="help")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert not response.success
        assert "Assistant error" in response.response_text

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_grpc_error_handled(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        import grpc

        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_3",
            tool_name="color_blindness_assist",
            parameters={"query": "colors"},
        )

        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.side_effect = grpc.RpcError()
        mock_tool_stub_cls.return_value = mock_tool_stub
        mock_frontend_stub_cls.return_value = MagicMock()

        request = assistant_pb2.CommandRequest(text="help with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert not response.success
        assert "Failed to reach tool service" in response.response_text

    def test_whitespace_only_command(self, servicer, mock_grpc_context):
        request = assistant_pb2.CommandRequest(text="   \n\t  ")
        response = servicer.ProcessCommand(request, mock_grpc_context)
        assert not response.success
        assert response.response_text == "Empty command"

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_failure_passed_through(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        """When the tool returns success=False, the response reflects that."""
        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_4",
            tool_name="color_blindness_assist",
            parameters={"query": "colors"},
        )

        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.return_value = tool_pb2.ToolResponse(
            success=False, result_text="Camera unavailable"
        )
        mock_tool_stub_cls.return_value = mock_tool_stub
        mock_frontend_stub_cls.return_value = MagicMock()

        request = assistant_pb2.CommandRequest(text="help with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert not response.success
        assert response.response_text == "Camera unavailable"
        assert response.tool_name == "color_blindness_assist"

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_channel_closed_on_grpc_error(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel_cls,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        """Channel is closed even when the tool gRPC call fails."""
        import grpc

        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_5",
            tool_name="color_blindness_assist",
            parameters={},
        )

        frontend_channel = MagicMock()
        tool_channel = MagicMock()
        mock_channel_cls.side_effect = [frontend_channel, tool_channel]
        mock_frontend_stub_cls.return_value = MagicMock()
        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.side_effect = grpc.RpcError()
        mock_tool_stub_cls.return_value = mock_tool_stub

        request = assistant_pb2.CommandRequest(text="colors")
        servicer.ProcessCommand(request, mock_grpc_context)

        frontend_channel.close.assert_called_once()
        tool_channel.close.assert_called_once()

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_frontend_notification_error_does_not_block_tool_call(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel_cls,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        import grpc

        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_6",
            tool_name="color_blindness_assist",
            parameters={"query": "colors"},
        )

        frontend_channel = MagicMock()
        tool_channel = MagicMock()
        speak_channel = MagicMock()
        mock_channel_cls.side_effect = [frontend_channel, tool_channel, speak_channel]

        mock_frontend_stub = MagicMock()
        mock_frontend_stub.ShowNotification.side_effect = grpc.RpcError()
        mock_frontend_stub_cls.return_value = mock_frontend_stub

        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.return_value = tool_pb2.ToolResponse(
            success=True,
            result_text="The shirt is blue",
        )
        mock_tool_stub_cls.return_value = mock_tool_stub

        request = assistant_pb2.CommandRequest(text="help me with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == "The shirt is blue"
