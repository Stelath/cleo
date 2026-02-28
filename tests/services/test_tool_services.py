"""Tests for tool services — base class and individual tool implementations."""

import json
from unittest.mock import MagicMock

import pytest

from generated import tool_pb2
from apps.color_blind import ColorBlindnessServicer
from apps.navigation_assist import NavigationAssistServicer
from apps.object_recognition import ObjectRecognitionServicer
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


class TestObjectRecognitionServicer:
    def test_execute_returns_success(self, mock_grpc_context):
        servicer = ObjectRecognitionServicer()
        request = tool_pb2.ToolRequest(
            tool_name="object_recognition",
            parameters_json=json.dumps({"query": "what is this?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "what is this?" in response.result_text

    def test_tool_name(self):
        assert ObjectRecognitionServicer().tool_name == "object_recognition"


class TestNavigationAssistServicer:
    def test_execute_returns_success(self, mock_grpc_context):
        servicer = NavigationAssistServicer()
        request = tool_pb2.ToolRequest(
            tool_name="navigation_assist",
            parameters_json=json.dumps({"query": "find nearest coffee shop"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "find nearest coffee shop" in response.result_text

    def test_tool_name(self):
        assert NavigationAssistServicer().tool_name == "navigation_assist"


class TestToolServiceProperties:
    """Verify tool_description, tool_input_schema, and tool_type on all servicers."""

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer, ObjectRecognitionServicer, NavigationAssistServicer],
    )
    def test_tool_description_is_nonempty(self, servicer_cls):
        servicer = servicer_cls()
        assert isinstance(servicer.tool_description, str)
        assert len(servicer.tool_description) > 0

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer, ObjectRecognitionServicer, NavigationAssistServicer],
    )
    def test_tool_input_schema_has_type(self, servicer_cls):
        servicer = servicer_cls()
        schema = servicer.tool_input_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer, ObjectRecognitionServicer, NavigationAssistServicer],
    )
    def test_tool_type_defaults_to_on_demand(self, servicer_cls):
        servicer = servicer_cls()
        assert servicer.tool_type == "on_demand"
