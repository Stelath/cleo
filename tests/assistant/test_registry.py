"""Tests for assistant.registry — tool registration and Bedrock config generation."""

import pytest

from assistant.registry import ToolDefinition, ToolRegistry


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


class TestToolRegistry:
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

    def test_default_registry_has_three_tools(self):
        registry = ToolRegistry()
        assert len(registry.tool_names) == 3
        assert "color_blindness_assist" in registry.tool_names
        assert "object_recognition" in registry.tool_names
        assert "navigation_assist" in registry.tool_names

    def test_empty_registry(self):
        registry = ToolRegistry(tools=[])
        assert registry.tool_names == []
        assert registry.get("anything") is None
        config = registry.bedrock_tool_config()
        assert config == {"tools": []}
