"""Assistant gRPC service: receives commands, routes via Bedrock tool-use."""

import json
import os
import signal
import time
from concurrent import futures

import grpc
import structlog

from services.assistant.bedrock import BedrockClient, TextResult, ToolUseResult
from services.assistant.registry import ToolRegistry
from services.config import ASSISTANT_PORT, DATA_ADDRESS, FRONTEND_ADDRESS, SENSOR_ADDRESS
from generated import assistant_pb2, assistant_pb2_grpc
from generated import frontend_pb2, frontend_pb2_grpc
from generated import sensor_pb2, sensor_pb2_grpc
from generated import tool_pb2, tool_pb2_grpc
from services.media.camera_transport import (
    AssembledCameraFrame,
    CameraFrameAssembler,
    assembled_frame_to_rgb,
    encode_rgb_to_jpeg,
)

log = structlog.get_logger()

_DEBUG_ASSISTANT_HUD_ENV = "CLEO_DEBUG_ASSISTANT_HUD"
_DEBUG_ASSISTANT_HUD_MAX_CHARS = 180


def _truncate(value: str, *, max_chars: int = _DEBUG_ASSISTANT_HUD_MAX_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."


class AssistantServiceServicer(assistant_pb2_grpc.AssistantServiceServicer):
    """Receives transcribed commands, uses Bedrock to pick a tool, calls it via gRPC."""

    _IDLE_TIMEOUT_SECONDS = 300  # 5 minutes
    _MAX_HISTORY_MESSAGES = 20

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        bedrock_client: BedrockClient | None = None,
        sensor_address: str = SENSOR_ADDRESS,
    ):
        self._registry = registry or ToolRegistry()
        self._bedrock = bedrock_client or BedrockClient()
        self._sensor_address = sensor_address
        self._conversation_history: list[dict] = []
        self._last_interaction_time: float = 0.0
        enabled = os.getenv(_DEBUG_ASSISTANT_HUD_ENV, "").strip().lower()
        self._debug_hud_enabled = enabled in {"1", "true", "yes", "on"}
        if self._debug_hud_enabled:
            log.info("assistant.debug_hud_enabled", env_var=_DEBUG_ASSISTANT_HUD_ENV)

    def _show_llm_debug_response(self, text: str) -> None:
        if not self._debug_hud_enabled or not text:
            return

        channel = grpc.insecure_channel(FRONTEND_ADDRESS)
        try:
            stub = frontend_pb2_grpc.FrontendServiceStub(channel)
            stub.ShowText(
                frontend_pb2.TextRequest(
                    text=f"LLM: {_truncate(text)}",
                    position="upper-right",
                ),
                timeout=2,
            )
        except grpc.RpcError as e:
            log.warning("assistant.debug_hud_publish_failed", error=str(e))
        finally:
            channel.close()

    def _show_tool_debug_notification(self, tool_name: str) -> None:
        """Best-effort HUD notification so the user can see which tool was invoked."""
        channel = grpc.insecure_channel(FRONTEND_ADDRESS)
        try:
            stub = frontend_pb2_grpc.FrontendServiceStub(channel)
            stub.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title="Tool called",
                    message=tool_name,
                    style="debug",
                    duration_ms=2000,
                )
            )
        except grpc.RpcError as e:
            log.warning(
                "assistant.frontend_notification_error",
                tool=tool_name,
                error=str(e),
            )
        finally:
            channel.close()

    def _speak_response_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        channel = grpc.insecure_channel(FRONTEND_ADDRESS)
        try:
            stub = frontend_pb2_grpc.FrontendServiceStub(channel)
            stub.SpeakText(
                frontend_pb2.SpeakTextRequest(text=cleaned),
                timeout=10,
            )
        except grpc.RpcError as e:
            log.warning("assistant.frontend_speak_error", error=str(e))
        finally:
            channel.close()

    def _capture_frame_jpeg(self) -> bytes | None:
        """Grab one JPEG frame from the sensor service.  Returns *None* on failure."""
        channel = grpc.insecure_channel(self._sensor_address)
        try:
            stub = sensor_pb2_grpc.SensorServiceStub(channel)
            stream = stub.CaptureFrame(sensor_pb2.CaptureRequest())
            assembler = CameraFrameAssembler()
            for chunk in stream:
                frame = assembler.push(chunk)
                if frame is not None:
                    if frame.encoding == sensor_pb2.FRAME_ENCODING_JPEG:
                        return frame.data
                    frame_rgb = assembled_frame_to_rgb(frame)
                    return encode_rgb_to_jpeg(frame_rgb)
        except Exception as e:
            log.warning("assistant.frame_capture_failed", error=str(e))
        finally:
            channel.close()
        return None

    def _maybe_clear_stale_history(self) -> None:
        """Reset conversation if idle for too long."""
        if (
            self._conversation_history
            and time.monotonic() - self._last_interaction_time > self._IDLE_TIMEOUT_SECONDS
        ):
            log.info("assistant.history_expired")
            self._conversation_history = []

    def _trim_history(self) -> None:
        """Cap conversation history at *_MAX_HISTORY_MESSAGES* messages."""
        if len(self._conversation_history) > self._MAX_HISTORY_MESSAGES:
            self._conversation_history = self._conversation_history[-self._MAX_HISTORY_MESSAGES:]

    def ProcessCommand(self, request, context):
        text = request.text.strip()
        is_follow_up = bool(request.is_follow_up)
        if not text:
            return assistant_pb2.CommandResponse(
                success=False,
                response_text="Empty command",
                tool_name="",
                responded=False,
                continue_follow_up=False,
            )

        log.info("assistant.process_command", text=text, is_follow_up=is_follow_up)

        if is_follow_up:
            try:
                should_respond = self._bedrock.classify_follow_up(text)
            except Exception as e:
                log.error("assistant.follow_up_classifier_error", error=str(e))
                return assistant_pb2.CommandResponse(
                    success=False,
                    response_text=f"Assistant error: {e}",
                    tool_name="",
                    responded=False,
                    continue_follow_up=False,
                )

            if not should_respond:
                log.info("assistant.follow_up_ended")
                self._conversation_history = []
                return assistant_pb2.CommandResponse(
                    success=True,
                    response_text="",
                    tool_name="",
                    responded=False,
                    continue_follow_up=False,
                )

        # Manage conversation state
        self._maybe_clear_stale_history()
        self._trim_history()

        # Always try to capture a frame for vision context
        image_bytes = self._capture_frame_jpeg()

        try:
            result, self._conversation_history = self._bedrock.converse(
                user_text=text,
                tool_config=self._registry.bedrock_tool_config(),
                messages=self._conversation_history,
                image_bytes=image_bytes,
            )
        except Exception as e:
            log.error("assistant.bedrock_error", error=str(e))
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Assistant error: {e}",
                tool_name="",
                responded=False,
                continue_follow_up=False,
            )

        self._last_interaction_time = time.monotonic()

        if isinstance(result, TextResult):
            response_text = result.text.strip()
            self._show_llm_debug_response(response_text)
            log.info("assistant.text_response", text=response_text[:100])
            responded = bool(response_text)
            if responded:
                self._speak_response_text(response_text)
            if not responded:
                self._conversation_history = []
            return assistant_pb2.CommandResponse(
                success=True,
                response_text=response_text,
                tool_name="",
                responded=responded,
                continue_follow_up=responded,
            )

        self._show_llm_debug_response(
            f"tool={result.tool_name} params={json.dumps(result.parameters, default=str)}"
        )

        # ToolUseResult — look up the tool and call it via gRPC
        tool_def = self._registry.get(result.tool_name)
        if tool_def is None:
            log.error("assistant.unknown_tool", tool=result.tool_name)
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Unknown tool: {result.tool_name}",
                tool_name=result.tool_name,
                responded=False,
                continue_follow_up=False,
            )

        log.info(
            "assistant.calling_tool",
            tool=result.tool_name,
            address=tool_def.grpc_address,
        )
        self._show_tool_debug_notification(result.tool_name)

        # Speak LLM's accompanying text (if any) before tool execution.
        # Only the LLM's own text is spoken — tool output is never read aloud.
        if result.response_text:
            self._speak_response_text(result.response_text)

        channel = grpc.insecure_channel(tool_def.grpc_address)
        try:
            stub = tool_pb2_grpc.ToolServiceStub(channel)
            tool_response = stub.Execute(
                tool_pb2.ToolRequest(
                    tool_name=result.tool_name,
                    parameters_json=json.dumps(result.parameters),
                )
            )

            # Append tool result to conversation history so the LLM
            # can reference it in follow-ups.
            self._conversation_history.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": result.tool_use_id,
                        "content": [{"text": tool_response.result_text}],
                    }
                }],
            })

            response_text = tool_response.result_text.strip()
            responded = bool(response_text) and bool(tool_response.success)
            if not responded:
                self._conversation_history = []
            return assistant_pb2.CommandResponse(
                success=tool_response.success,
                response_text=tool_response.result_text,
                tool_name=result.tool_name,
                responded=responded,
                continue_follow_up=responded,
            )
        except grpc.RpcError as e:
            log.error("assistant.tool_rpc_error", tool=result.tool_name, error=str(e))
            return assistant_pb2.CommandResponse(
                success=False,
                response_text=f"Failed to reach tool service: {e}",
                tool_name=result.tool_name,
                responded=False,
                continue_follow_up=False,
            )
        finally:
            channel.close()


def serve(port: int = ASSISTANT_PORT):
    """Start the assistant gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    registry = ToolRegistry(data_address=DATA_ADDRESS)
    servicer = AssistantServiceServicer(registry=registry)
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
