"""Bedrock Converse API client for tool-use routing."""

import os
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()

_MODEL_ID = os.environ.get(
    "CLEO_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0"
)
_BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
_SYSTEM_PROMPT = (
    "You are Cleo, an AI assistant running on AR smart glasses. "
    "The user speaks commands aloud after saying 'hey cleo'. "
    "Be concise — responses are displayed on a small heads-up display. "
    "Tool use is preferred whenever a user request maps to an available tool. "
    "If transcript context contains chatter, focus on the words after the last wake phrase. "
    "For nutrition, calories, macros, protein, fat, carbs, food, barcode, or meal questions, "
    "call the food_macros tool. "
    "For color blindness correction or color assist requests, call the color_blindness_assist tool. "
    "For weather, temperature, forecast, rain, or outside conditions questions, call the weather tool. "
    "If no tool fits, respond with a short helpful text answer."
)
_LOG_TEXT_MAX_CHARS = 400


def _truncate(value: str, max_chars: int = _LOG_TEXT_MAX_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."


@dataclass
class ToolUseResult:
    """Bedrock chose to invoke a tool."""

    tool_use_id: str
    tool_name: str
    parameters: dict[str, Any]


@dataclass
class TextResult:
    """Bedrock responded with plain text (no tool invoked)."""

    text: str


class BedrockClient:
    """Thin wrapper around AWS Bedrock Converse API for tool-use routing."""

    def __init__(self, client: Any = None, model_id: str = _MODEL_ID, region: str = _BEDROCK_REGION):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def converse(self, user_text: str, tool_config: dict) -> ToolUseResult | TextResult:
        """Send user text to Bedrock and return either a tool-use or text result."""
        tools = tool_config.get("tools", []) if isinstance(tool_config, dict) else []
        log.info(
            "assistant.llm_prompt",
            model_id=self._model_id,
            tool_count=len(tools),
            system_prompt=_truncate(_SYSTEM_PROMPT),
            user_text=_truncate(user_text),
        )

        converse_args: dict[str, Any] = {
            "modelId": self._model_id,
            "system": [{"text": _SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
        }
        if tools:
            converse_args["toolConfig"] = tool_config
        else:
            log.warning("assistant.no_tools_registered")

        response = self._client.converse(**converse_args)

        stop_reason = response.get("stopReason", "")
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])

        log.debug(
            "bedrock.response",
            stop_reason=stop_reason,
            num_blocks=len(content_blocks),
        )

        # Check for tool use blocks first
        for block in content_blocks:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                log.info(
                    "assistant.llm_response",
                    response_kind="tool_use",
                    stop_reason=stop_reason,
                    tool_name=tool_use.get("name", ""),
                    tool_use_id=tool_use.get("toolUseId", ""),
                    parameters=tool_use.get("input", {}),
                )
                return ToolUseResult(
                    tool_use_id=tool_use["toolUseId"],
                    tool_name=tool_use["name"],
                    parameters=tool_use.get("input", {}),
                )

        # Fall back to text
        text_parts = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
        text_result = "\n".join(text_parts) if text_parts else ""
        log.info(
            "assistant.llm_response",
            response_kind="text",
            stop_reason=stop_reason,
            text=_truncate(text_result),
        )
        return TextResult(text=text_result)
