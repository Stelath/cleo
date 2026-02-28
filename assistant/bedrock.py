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
    "Use the provided tools when the user's request matches a tool's purpose. "
    "If no tool fits, respond with a short helpful text answer."
)


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
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": _SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            toolConfig=tool_config,
        )

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

        return TextResult(text="\n".join(text_parts) if text_parts else "")
