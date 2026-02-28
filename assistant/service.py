"""Assistant gRPC service: receives commands, routes via Bedrock tool-use."""

import json
import signal
from concurrent import futures

import grpc
import structlog

from assistant.bedrock import BedrockClient, TextResult, ToolUseResult
from assistant.registry import ToolRegistry
from core.ports import ASSISTANT_PORT
from generated import assistant_pb2, assistant_pb2_grpc
from generated import tool_pb2, tool_pb2_grpc

log = structlog.get_logger()


class AssistantServiceServicer(assistant_pb2_grpc.AssistantServiceServicer):
    """Receives transcribed commands, uses Bedrock to pick a tool, calls it via gRPC."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        bedrock_client: BedrockClient | None = None,
    ):
        self._registry = registry or ToolRegistry()
        self._bedrock = bedrock_client or BedrockClient()

    def ProcessCommand(self, request, context):
        text = request.text.strip()
        if not text:
            return assistant_pb2.CommandResponse(
                success=False,
                response_text="Empty command",
                tool_name="",
            )

        log.info("assistant.process_command", text=text)

        try:
            result = self._bedrock.converse(
                user_text=text,
                tool_config=self._registry.bedrock_tool_config(),
            )
        except Exception as e:
            log.error("assistant.bedrock_error", error=str(e))
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Assistant error: {e}",
                tool_name="",
            )

        if isinstance(result, TextResult):
            log.info("assistant.text_response", text=result.text[:100])
            return assistant_pb2.CommandResponse(
                success=True,
                response_text=result.text,
                tool_name="",
            )

        # ToolUseResult — look up the tool and call it via gRPC
        tool_def = self._registry.get(result.tool_name)
        if tool_def is None:
            log.error("assistant.unknown_tool", tool=result.tool_name)
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Unknown tool: {result.tool_name}",
                tool_name=result.tool_name,
            )

        log.info(
            "assistant.calling_tool",
            tool=result.tool_name,
            address=tool_def.grpc_address,
        )

        channel = grpc.insecure_channel(tool_def.grpc_address)
        try:
            stub = tool_pb2_grpc.ToolServiceStub(channel)
            tool_response = stub.Execute(
                tool_pb2.ToolRequest(
                    tool_name=result.tool_name,
                    parameters_json=json.dumps(result.parameters),
                )
            )
            return assistant_pb2.CommandResponse(
                success=tool_response.success,
                response_text=tool_response.result_text,
                tool_name=result.tool_name,
            )
        except grpc.RpcError as e:
            log.error("assistant.tool_rpc_error", tool=result.tool_name, error=str(e))
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Failed to reach tool service: {e}",
                tool_name=result.tool_name,
            )
        finally:
            channel.close()


def serve(port: int = ASSISTANT_PORT):
    """Start the assistant gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = AssistantServiceServicer()
    assistant_pb2_grpc.add_AssistantServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("assistant_service.started", port=port)

    def _shutdown(signum, frame):
        log.info("assistant_service.stopping", signal=signum)
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
