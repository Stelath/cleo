"""Tests for tool services — base class and individual tool implementations."""

import json
from unittest.mock import MagicMock

import pytest

from generated import tool_pb2
from apps.color_blind import ColorBlindnessServicer
from apps.navigation_assist import NavigationAssistServicer
from apps.object_recognition import ObjectRecognitionServicer
from apps.tool_base import ToolServiceBase


class TestToolServiceBase:
    def test_wrong_tool_name_rejected(self, mock_grpc_context):
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="wrong_name",
            parameters_json=json.dumps({"query": "test"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        mock_grpc_context.set_code.assert_called()

    def test_invalid_json_rejected(self, mock_grpc_context):
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="{bad json",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Invalid parameters" in response.result_text

    def test_empty_params_json_is_ok(self, mock_grpc_context):
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ToolServiceBase()


class TestColorBlindnessServicer:
    def test_execute_returns_success(self, mock_grpc_context):
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json=json.dumps({"query": "what color is this?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "what color is this?" in response.result_text

    def test_tool_name(self):
        assert ColorBlindnessServicer().tool_name == "color_blindness_assist"


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
