"""Tests for assistant.registry — static and dynamic tool registry."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from services.assistant.registry import ToolDefinition, ToolRegistry, _CACHE_TTL_SECONDS


@pytest.fixture
def sample_tools():
    return [
        ToolDefinition(
            name="tool_a",
            description="Tool A does things",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            grpc_address="localhost:50060",
        ),
        ToolDefinition(
            name="tool_b",
            description="Tool B does other things",
            input_schema={"type": "object", "properties": {"y": {"type": "integer"}}},
            grpc_address="localhost:50061",
        ),
    ]


# ── Static mode tests (existing behavior) ──


class TestStaticRegistry:
    def test_loads_tools(self, sample_tools):
        registry = ToolRegistry(tools=sample_tools)
        assert registry.tool_names == ["tool_a", "tool_b"]

    def test_get_existing_tool(self, sample_tools):
        registry = ToolRegistry(tools=sample_tools)
        tool = registry.get("tool_a")
        assert tool is not None
        assert tool.name == "tool_a"
        assert tool.grpc_address == "localhost:50060"

    def test_get_missing_tool_returns_none(self, sample_tools):
        registry = ToolRegistry(tools=sample_tools)
        assert registry.get("nonexistent") is None

    def test_bedrock_tool_config_format(self, sample_tools):
        registry = ToolRegistry(tools=sample_tools)
        config = registry.bedrock_tool_config()

        assert "tools" in config
        tools = config["tools"]
        assert len(tools) == 2

        spec = tools[0]["toolSpec"]
        assert spec["name"] == "tool_a"
        assert spec["description"] == "Tool A does things"
        assert spec["inputSchema"]["json"] == sample_tools[0].input_schema

    def test_default_registry_has_four_tools(self):
        registry = ToolRegistry()
        assert len(registry.tool_names) == 4
        assert "color_blindness_assist" in registry.tool_names
        assert "object_recognition" in registry.tool_names
        assert "navigation_assist" in registry.tool_names
        assert "notetaking" in registry.tool_names

    def test_empty_registry(self):
        registry = ToolRegistry(tools=[])
        assert registry.tool_names == []
        assert registry.get("anything") is None
        config = registry.bedrock_tool_config()
        assert config == {"tools": []}


# ── Dynamic mode tests ──


def _make_app_info(**kwargs):
    """Build a mock AppInfo protobuf-like object."""
    defaults = dict(
        id=1,
        name="test_tool",
        description="A test tool",
        app_type="on_demand",
        grpc_address="localhost:50060",
        input_schema_json='{"type": "object"}',
        enabled=True,
        registered_at=time.time(),
        updated_at=time.time(),
    )
    defaults.update(kwargs)
    info = MagicMock()
    for k, v in defaults.items():
        setattr(info, k, v)
    return info


def _mock_list_apps_response(apps):
    """Create a mock ListAppsResponse."""
    resp = MagicMock()
    resp.apps = apps
    return resp


class TestDynamicRegistry:
    def test_fetches_from_data_service(self):
        app = _make_app_info(name="my_tool", description="My tool")
        mock_resp = _mock_list_apps_response([app])

        with patch("services.assistant.registry.grpc") as mock_grpc:
            mock_stub = MagicMock()
            mock_stub.ListApps.return_value = mock_resp
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("generated.data_pb2_grpc.DataServiceStub", return_value=mock_stub):
                registry = ToolRegistry(data_address="localhost:50053")
                names = registry.tool_names
                assert "my_tool" in names
                tool = registry.get("my_tool")
                assert tool.description == "My tool"

    def test_caches_within_ttl(self):
        app = _make_app_info(name="cached_tool")
        mock_resp = _mock_list_apps_response([app])

        with patch("services.assistant.registry.grpc") as mock_grpc:
            mock_stub = MagicMock()
            mock_stub.ListApps.return_value = mock_resp
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("generated.data_pb2_grpc.DataServiceStub", return_value=mock_stub):
                registry = ToolRegistry(data_address="localhost:50053")

                # First call — populates cache
                registry.tool_names
                # Second call — should use cache, not call ListApps again
                registry.tool_names

                assert mock_stub.ListApps.call_count == 1

    def test_refreshes_after_ttl(self):
        app = _make_app_info(name="refreshed_tool")
        mock_resp = _mock_list_apps_response([app])

        with patch("services.assistant.registry.grpc") as mock_grpc:
            mock_stub = MagicMock()
            mock_stub.ListApps.return_value = mock_resp
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("generated.data_pb2_grpc.DataServiceStub", return_value=mock_stub):
                registry = ToolRegistry(data_address="localhost:50053")

                # First call
                registry.tool_names
                assert mock_stub.ListApps.call_count == 1

                # Expire cache by manipulating _cache_time
                registry._cache_time -= _CACHE_TTL_SECONDS + 1

                # Second call — should re-fetch
                registry.tool_names
                assert mock_stub.ListApps.call_count == 2

    def test_survives_data_service_failure(self):
        app = _make_app_info(name="resilient_tool")
        mock_resp = _mock_list_apps_response([app])

        with patch("services.assistant.registry.grpc") as mock_grpc:
            mock_stub = MagicMock()
            mock_stub.ListApps.return_value = mock_resp
            mock_channel = MagicMock()
            mock_grpc.insecure_channel.return_value = mock_channel

            with patch("generated.data_pb2_grpc.DataServiceStub", return_value=mock_stub):
                registry = ToolRegistry(data_address="localhost:50053")

                # Populate cache
                assert "resilient_tool" in registry.tool_names

                # Expire cache
                registry._cache_time -= _CACHE_TTL_SECONDS + 1

                # Make ListApps fail
                mock_stub.ListApps.side_effect = Exception("DataService down")

                # Should still return stale cache
                assert "resilient_tool" in registry.tool_names

    def test_no_data_address_returns_empty(self):
        """Dynamic mode with data_address=None returns empty tools."""
        registry = ToolRegistry()
        assert registry.tool_names == []
