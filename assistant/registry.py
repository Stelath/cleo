"""Tool registry: static list of tool definitions for Bedrock tool-use routing."""

from dataclasses import dataclass
from typing import Any

from core.ports import COLOR_BLIND_PORT, NAVIGATION_ASSIST_PORT, OBJECT_RECOGNITION_PORT


@dataclass(frozen=True)
class ToolDefinition:
    """A tool that can be invoked by the assistant."""

    name: str
    description: str
    input_schema: dict[str, Any]
    grpc_address: str


# Default tools available to the assistant
_DEFAULT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="color_blindness_assist",
        description=(
            "Help a color-blind user identify or distinguish colors in their "
            "current view. Use when the user asks about colors, color matching, "
            "or needs help telling colors apart."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants to know about colors",
                },
            },
            "required": ["query"],
        },
        grpc_address=f"localhost:{COLOR_BLIND_PORT}",
    ),
    ToolDefinition(
        name="object_recognition",
        description=(
            "Identify and describe objects in the user's current view. Use when "
            "the user asks 'what is this?', 'what am I looking at?', or wants "
            "to identify something in front of them."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What the user wants identified or described",
                },
            },
            "required": ["query"],
        },
        grpc_address=f"localhost:{OBJECT_RECOGNITION_PORT}",
    ),
    ToolDefinition(
        name="navigation_assist",
        description=(
            "Help the user navigate or find directions. Use when the user asks "
            "for directions, nearby places, or help getting somewhere."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Navigation request or destination",
                },
            },
            "required": ["query"],
        },
        grpc_address=f"localhost:{NAVIGATION_ASSIST_PORT}",
    ),
]


class ToolRegistry:
    """Registry of available tools with Bedrock-compatible config generation."""

    def __init__(self, tools: list[ToolDefinition] | None = None):
        self._tools = {t.name: t for t in (_DEFAULT_TOOLS if tools is None else tools)}

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def bedrock_tool_config(self) -> dict:
        """Return the toolConfig dict expected by Bedrock Converse API."""
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
            )
        return {"tools": tools}
