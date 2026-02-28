"""Base class for gRPC tool services."""

import json
import signal
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any

import grpc
import structlog

from generated import tool_pb2, tool_pb2_grpc

log = structlog.get_logger()


class ToolServiceBase(tool_pb2_grpc.ToolServiceServicer, ABC):
    """Abstract base for all tool services.

    Subclasses implement `tool_name`, `tool_description`, `tool_input_schema`,
    and `execute(params) -> (success, result_text)`.
    The base class handles JSON parsing, error wrapping, and gRPC plumbing.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """The registered tool name (must match registry entry)."""

    @property
    @abstractmethod
    def tool_description(self) -> str:
        """LLM-facing description used in Bedrock tool-use config."""

    @property
    @abstractmethod
    def tool_input_schema(self) -> dict[str, Any]:
        """JSON schema for the tool's input parameters."""

    @property
    def tool_type(self) -> str:
        """App type: 'on_demand' (default, one-shot) or 'active' (session, start/stop)."""
        return "on_demand"

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


def _register_with_data_service(
    servicer: ToolServiceBase,
    grpc_address: str,
    data_address: str,
) -> None:
    """Register the tool with DataService, retrying with exponential backoff."""
    from generated import data_pb2, data_pb2_grpc

    max_retries = 10
    delay = 1.0

    for attempt in range(max_retries):
        try:
            channel = grpc.insecure_channel(data_address)
            stub = data_pb2_grpc.DataServiceStub(channel)
            resp = stub.RegisterApp(
                data_pb2.RegisterAppRequest(
                    name=servicer.tool_name,
                    description=servicer.tool_description,
                    app_type=servicer.tool_type,
                    grpc_address=grpc_address,
                    input_schema_json=json.dumps(servicer.tool_input_schema),
                ),
                timeout=5,
            )
            log.info(
                "tool_service.registered",
                tool=servicer.tool_name,
                id=resp.id,
                created=resp.created,
            )
            channel.close()
            return
        except grpc.RpcError as e:
            log.warning(
                "tool_service.register_retry",
                tool=servicer.tool_name,
                attempt=attempt + 1,
                error=str(e),
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        except Exception:
            channel.close()
            raise

    log.error("tool_service.register_failed", tool=servicer.tool_name)


def serve_tool(
    servicer: ToolServiceBase,
    port: int,
    data_address: str | None = None,
):
    """Start a gRPC server for a single tool service.

    After the server is listening, a background daemon thread registers the
    tool with DataService so the assistant can discover it dynamically.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    tool_pb2_grpc.add_ToolServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("tool_service.started", tool=servicer.tool_name, port=port)

    # Self-register with DataService in a background thread
    if data_address is None:
        from services.config import DATA_ADDRESS
        data_address = DATA_ADDRESS

    grpc_address = f"localhost:{port}"
    reg_thread = threading.Thread(
        target=_register_with_data_service,
        args=(servicer, grpc_address, data_address),
        daemon=True,
    )
    reg_thread.start()

    def _shutdown(signum, frame):
        log.info("tool_service.stopping", tool=servicer.tool_name, signal=signum)
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()
