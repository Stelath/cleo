"""Tests for assistant.service — full flow with mocked Bedrock and tool gRPC calls."""

from unittest.mock import MagicMock, patch

import pytest

from assistant.bedrock import TextResult, ToolUseResult
from assistant.registry import ToolDefinition, ToolRegistry
from assistant.service import AssistantServiceServicer
from generated import assistant_pb2, tool_pb2


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


class TestAssistantService:
    def test_empty_command(self, servicer, mock_grpc_context):
        request = assistant_pb2.CommandRequest(text="")
        response = servicer.ProcessCommand(request, mock_grpc_context)
        assert not response.success
        assert response.response_text == "Empty command"

    def test_text_response_flow(self, servicer, mock_bedrock, mock_grpc_context):
        mock_bedrock.converse.return_value = TextResult(text="Hello from Cleo!")
        request = assistant_pb2.CommandRequest(text="hello")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == "Hello from Cleo!"
        assert response.tool_name == ""

    @patch("assistant.service.grpc.insecure_channel")
    @patch("assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_use_flow(
        self, mock_stub_cls, mock_channel, servicer, mock_bedrock, mock_grpc_context
    ):
        # Bedrock returns tool use
        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_1",
            tool_name="color_blindness_assist",
            parameters={"query": "what color is this shirt?"},
        )

        # Tool gRPC returns success
        mock_stub = MagicMock()
        mock_stub.Execute.return_value = tool_pb2.ToolResponse(
            success=True, result_text="The shirt is blue"
        )
        mock_stub_cls.return_value = mock_stub

        request = assistant_pb2.CommandRequest(text="help me with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert response.success
        assert response.response_text == "The shirt is blue"
        assert response.tool_name == "color_blindness_assist"

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

    @patch("assistant.service.grpc.insecure_channel")
    @patch("assistant.service.tool_pb2_grpc.ToolServiceStub")
    def test_tool_grpc_error_handled(
        self, mock_stub_cls, mock_channel, servicer, mock_bedrock, mock_grpc_context
    ):
        import grpc

        mock_bedrock.converse.return_value = ToolUseResult(
            tool_use_id="tu_3",
            tool_name="color_blindness_assist",
            parameters={"query": "colors"},
        )

        mock_stub = MagicMock()
        mock_stub.Execute.side_effect = grpc.RpcError()
        mock_stub_cls.return_value = mock_stub

        request = assistant_pb2.CommandRequest(text="help with colors")
        response = servicer.ProcessCommand(request, mock_grpc_context)

        assert not response.success
        assert "Failed to reach tool service" in response.response_text
