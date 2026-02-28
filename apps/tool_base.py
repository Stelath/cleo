"""Base class for gRPC tool services."""

import json
import signal
from abc import ABC, abstractmethod
from concurrent import futures

import grpc
import structlog

from generated import tool_pb2, tool_pb2_grpc

log = structlog.get_logger()


class ToolServiceBase(tool_pb2_grpc.ToolServiceServicer, ABC):
    """Abstract base for all tool services.

    Subclasses implement `tool_name` and `execute(params) -> (success, result_text)`.
    The base class handles JSON parsing, error wrapping, and gRPC plumbing.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """The registered tool name (must match registry entry)."""

    @abstractmethod
    def execute(self, params: dict) -> tuple[bool, str]:
        """Execute the tool with parsed parameters.

        Returns:
            (success, result_text) tuple.
        """

    def Execute(self, request, context):
        """gRPC Execute RPC — parses JSON params and delegates to execute()."""
        log.info("tool.execute", tool=self.tool_name, request_tool=request.tool_name)

        if request.tool_name != self.tool_name:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"This service handles '{self.tool_name}', got '{request.tool_name}'"
            )
            return tool_pb2.ToolResponse(success=False, result_text="Wrong tool service")

        try:
            params = json.loads(request.parameters_json) if request.parameters_json else {}
        except json.JSONDecodeError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid JSON parameters: {e}")
            return tool_pb2.ToolResponse(success=False, result_text="Invalid parameters")

        try:
            success, result_text = self.execute(params)
            return tool_pb2.ToolResponse(success=success, result_text=result_text)
        except Exception as e:
            log.error("tool.execute_error", tool=self.tool_name, error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tool_pb2.ToolResponse(success=False, result_text=f"Tool error: {e}")


def serve_tool(servicer: ToolServiceBase, port: int):
    """Start a gRPC server for a single tool service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    tool_pb2_grpc.add_ToolServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("tool_service.started", tool=servicer.tool_name, port=port)

    def _shutdown(signum, frame):
        log.info("tool_service.stopping", tool=servicer.tool_name, signal=signum)
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()
