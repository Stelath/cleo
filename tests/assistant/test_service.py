"""Tests for assistant.service — full flow with mocked Bedrock and tool gRPC calls."""

import time
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
    s = AssistantServiceServicer(
        registry=registry_with_tool,
        bedrock_client=mock_bedrock,
    )
    # Stub out frame capture so tests don't need a sensor service
    s._capture_frame_jpeg = MagicMock(return_value=None)
    return s


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
        mock_bedrock.converse.return_value = (TextResult(text="Hello from Cleo!"), [])
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
        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_1",
                tool_name="color_blindness_assist",
                parameters={"query": "what color is this shirt?"},
            ),
            [],
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

        mock_bedrock.converse.return_value = (TextResult(text="Hello from Cleo!"), [])
        mock_frontend_stub = MagicMock()
        mock_frontend_stub_cls.return_value = mock_frontend_stub

        servicer = AssistantServiceServicer(
            registry=registry_with_tool,
            bedrock_client=mock_bedrock,
        )
        servicer._capture_frame_jpeg = MagicMock(return_value=None)

        request = assistant_pb2.CommandRequest(text="hello")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        show_text_call = mock_frontend_stub.ShowText.call_args
        assert show_text_call is not None
        assert show_text_call.args[0].text == "LLM: Hello from Cleo!"
        assert show_text_call.kwargs["timeout"] == 2

    def test_unknown_tool_from_bedrock(self, servicer, mock_bedrock, mock_grpc_context):
        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_2",
                tool_name="nonexistent_tool",
                parameters={},
            ),
            [],
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

        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_3",
                tool_name="color_blindness_assist",
                parameters={"query": "colors"},
            ),
            [],
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
        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_4",
                tool_name="color_blindness_assist",
                parameters={"query": "colors"},
            ),
            [],
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

        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_5",
                tool_name="color_blindness_assist",
                parameters={},
            ),
            [],
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

        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_6",
                tool_name="color_blindness_assist",
                parameters={"query": "colors"},
            ),
            [],
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


class TestConversationHistory:
    """Tests for persistent conversation history in the service."""

    def test_history_stored_after_text_response(self, servicer, mock_bedrock, mock_grpc_context):
        history = [
            {"role": "user", "content": [{"text": "hello"}]},
            {"role": "assistant", "content": [{"text": "Hi!"}]},
        ]
        mock_bedrock.converse.return_value = (TextResult(text="Hi!"), history)
        servicer._speak_response_text = MagicMock()

        request = assistant_pb2.CommandRequest(text="hello")
        servicer.ProcessCommand(request, mock_grpc_context)

        assert servicer._conversation_history == history

    def test_history_passed_to_bedrock(self, servicer, mock_bedrock, mock_grpc_context):
        """On a second call, the stored history is passed to bedrock.converse."""
        first_history = [
            {"role": "user", "content": [{"text": "q1"}]},
            {"role": "assistant", "content": [{"text": "a1"}]},
        ]
        mock_bedrock.converse.return_value = (TextResult(text="a1"), first_history)
        servicer._speak_response_text = MagicMock()

        servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="q1"), mock_grpc_context,
        )

        second_history = first_history + [
            {"role": "user", "content": [{"text": "q2"}]},
            {"role": "assistant", "content": [{"text": "a2"}]},
        ]
        mock_bedrock.converse.return_value = (TextResult(text="a2"), second_history)

        servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="q2"), mock_grpc_context,
        )

        # Second call should have received the first history
        _, call_kwargs = mock_bedrock.converse.call_args
        assert call_kwargs["messages"] == first_history

    def test_history_cleared_on_follow_up_end(self, servicer, mock_bedrock, mock_grpc_context):
        """When follow-up is classified as NO, history is cleared."""
        servicer._conversation_history = [
            {"role": "user", "content": [{"text": "old"}]},
            {"role": "assistant", "content": [{"text": "response"}]},
        ]
        mock_bedrock.classify_follow_up.return_value = False

        request = assistant_pb2.CommandRequest(text="bye", is_follow_up=True)
        servicer.ProcessCommand(request, mock_grpc_context)

        assert servicer._conversation_history == []

    @patch("services.assistant.service.time")
    def test_stale_history_cleared(self, mock_time, registry_with_tool, mock_bedrock, mock_grpc_context):
        """History is cleared if idle for longer than the timeout."""
        mock_time.monotonic.return_value = 1000.0

        servicer = AssistantServiceServicer(
            registry=registry_with_tool,
            bedrock_client=mock_bedrock,
        )
        servicer._capture_frame_jpeg = MagicMock(return_value=None)
        servicer._speak_response_text = MagicMock()

        # Simulate stale history
        servicer._conversation_history = [
            {"role": "user", "content": [{"text": "old"}]},
        ]
        servicer._last_interaction_time = 100.0  # 900 seconds ago > 300 timeout

        mock_bedrock.converse.return_value = (TextResult(text="fresh"), [
            {"role": "user", "content": [{"text": "fresh q"}]},
            {"role": "assistant", "content": [{"text": "fresh"}]},
        ])

        servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="fresh q"), mock_grpc_context,
        )

        # bedrock should have received empty history (cleared)
        _, call_kwargs = mock_bedrock.converse.call_args
        assert call_kwargs["messages"] == []

    def test_frame_capture_passed_to_bedrock(self, servicer, mock_bedrock, mock_grpc_context):
        """Frame JPEG bytes are passed to bedrock.converse as image_bytes."""
        fake_jpeg = b"\xff\xd8\xff\xe0fake"
        servicer._capture_frame_jpeg.return_value = fake_jpeg
        mock_bedrock.converse.return_value = (TextResult(text="I see something"), [])
        servicer._speak_response_text = MagicMock()

        servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="what do you see?"), mock_grpc_context,
        )

        _, call_kwargs = mock_bedrock.converse.call_args
        assert call_kwargs["image_bytes"] == fake_jpeg

    def test_frame_capture_failure_is_graceful(self, servicer, mock_bedrock, mock_grpc_context):
        """If frame capture returns None, converse still works without image."""
        servicer._capture_frame_jpeg.return_value = None
        mock_bedrock.converse.return_value = (TextResult(text="ok"), [])
        servicer._speak_response_text = MagicMock()

        response = servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="hello"), mock_grpc_context,
        )

        assert response.success
        _, call_kwargs = mock_bedrock.converse.call_args
        assert call_kwargs["image_bytes"] is None

    @patch("services.assistant.service.grpc.insecure_channel")
    @patch("services.assistant.service.frontend_pb2_grpc.FrontendServiceStub")
    @patch("services.assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_result_appended_to_history(
        self,
        mock_tool_stub_cls,
        mock_frontend_stub_cls,
        mock_channel,
        servicer,
        mock_bedrock,
        mock_grpc_context,
    ):
        """After a tool call, the tool result is appended to conversation history."""
        returned_history = [
            {"role": "user", "content": [{"text": "help with colors"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "tu_7", "name": "color_blindness_assist"}}]},
        ]
        mock_bedrock.converse.return_value = (
            ToolUseResult(
                tool_use_id="tu_7",
                tool_name="color_blindness_assist",
                parameters={"query": "colors"},
            ),
            returned_history,
        )

        mock_tool_stub = MagicMock()
        mock_tool_stub.Execute.return_value = tool_pb2.ToolResponse(
            success=True, result_text="The shirt is blue",
        )
        mock_tool_stub_cls.return_value = mock_tool_stub
        mock_frontend_stub_cls.return_value = MagicMock()

        servicer.ProcessCommand(
            assistant_pb2.CommandRequest(text="help with colors"), mock_grpc_context,
        )

        # History should include the toolResult
        assert len(servicer._conversation_history) == 3
        tool_result_msg = servicer._conversation_history[2]
        assert tool_result_msg["role"] == "user"
        assert "toolResult" in tool_result_msg["content"][0]
        assert tool_result_msg["content"][0]["toolResult"]["toolUseId"] == "tu_7"
