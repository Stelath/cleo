"""Tool registry: dynamic discovery via DataService or static list for tests."""

import json
import time
from dataclasses import dataclass
from typing import Any

import grpc
import structlog

log = structlog.get_logger()

_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class ToolDefinition:
    """A tool that can be invoked by the assistant."""

    name: str
    description: str
    input_schema: dict[str, Any]
    grpc_address: str


class ToolRegistry:
    """Registry of available tools with Bedrock-compatible config generation.

    Operates in one of two modes:

    * **Static** — ``ToolRegistry(tools=[...])`` uses a fixed list of tools.
      Ideal for tests.
    * **Dynamic** — ``ToolRegistry(data_address="localhost:50053")`` queries
      DataService.ListApps on demand and caches for 30 s.
    """

    def __init__(
        self,
        tools: list[ToolDefinition] | None = None,
        data_address: str | None = None,
    ):
        if tools is not None:
            # Static mode — fixed tool list, no DataService dependency
            self._static_tools: dict[str, ToolDefinition] | None = {
                t.name: t for t in tools
            }
            self._data_address: str | None = None
        else:
            # Dynamic mode — query DataService
            self._static_tools = None
            self._data_address = data_address
        self._cache: dict[str, ToolDefinition] = {}
        self._cache_time: float = 0.0

    # ── Public API ──

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name."""
        return self._get_tools().get(name)

    @property
    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._get_tools().keys())

    def bedrock_tool_config(self) -> dict:
        """Return the toolConfig dict expected by Bedrock Converse API."""
        tools = []
        for tool in self._get_tools().values():
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

    # ── Internal ──

    def _get_tools(self) -> dict[str, ToolDefinition]:
        """Return the current tool map, refreshing the cache if needed."""
        if self._static_tools is not None:
            return self._static_tools

        now = time.monotonic()
        if self._cache and (now - self._cache_time) < _CACHE_TTL_SECONDS:
            return self._cache

        refreshed = self._fetch_from_data_service()
        if refreshed is not None:
            self._cache = refreshed
            self._cache_time = now
        # If fetch failed, keep serving stale cache
        return self._cache

    def _fetch_from_data_service(self) -> dict[str, ToolDefinition] | None:
        """Query DataService.ListApps for enabled on_demand tools."""
        if self._data_address is None:
            return {}

        try:
            from generated import data_pb2, data_pb2_grpc

            channel = grpc.insecure_channel(self._data_address)
            stub = data_pb2_grpc.DataServiceStub(channel)
            resp = stub.ListApps(
                data_pb2.ListAppsRequest(enabled_only=True, app_type="on_demand"),
                timeout=5,
            )
            channel.close()

            tools: dict[str, ToolDefinition] = {}
            for app in resp.apps:
                try:
                    schema = json.loads(app.input_schema_json)
                except json.JSONDecodeError:
                    schema = {}
                tools[app.name] = ToolDefinition(
                    name=app.name,
                    description=app.description,
                    input_schema=schema,
                    grpc_address=app.grpc_address,
                )

            log.debug("registry.refreshed", tool_count=len(tools))
            return tools
        except Exception as e:
            log.warning("registry.fetch_failed", error=str(e))
            return None
