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
    "A camera image is attached to every message for context. "
    "Do NOT describe or comment on what you see unless the user explicitly asks about it "
    "or the image is directly relevant to answering their question. "
    "Tool use is preferred whenever a user request maps to an available tool. "
    "If transcript context contains chatter, focus on the words after the last wake phrase. "
    "For nutrition, calories, macros, protein, fat, carbs, food, barcode, or meal questions, "
    "call the food_macros tool. "
    "For color blindness correction or color assist requests, call the color_blindness_assist tool. "
    "For weather, temperature, forecast, rain, or outside conditions questions, call the weather tool. "
    "For registering a visible object to track later, call track_item_register with the item title. "
    "For requests like finding where an item was left last (for example phone, keys, wallet), call track_item_locate with the item title. "
    "For explicit recording-session commands like 'start recording' or 'stop recording', "
    "call the recording tool with action='start' to begin or action='stop' to stop and save. "
    "For one-shot clipping requests like 'clip that', 'save what happened', or 'save the last minute', "
    "call the save_video tool instead of recording. "
    "If no tool fits, respond with a short helpful text answer."
)
_FOLLOW_UP_CLASSIFIER_PROMPT = (
    "You are a strict classifier for spoken follow-up utterances to an AI assistant. "
    "Answer with exactly one token: YES or NO. "
    "Return YES only if the utterance is a direct continuation, clarification, or question "
    "for the assistant's immediately previous response. "
    "Return NO for ambient chatter, side conversations, acknowledgements that do not request more help, "
    "or unrelated speech."
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
    response_text: str = ""


@dataclass
class TextResult:
    """Bedrock responded with plain text (no tool invoked)."""

    text: str


class BedrockClient:
    """Thin wrapper around AWS Bedrock Converse API for tool-use routing."""

    def __init__(
        self,
        client: Any = None,
        model_id: str = _MODEL_ID,
        region: str = _BEDROCK_REGION,
    ):
        self._model_id = model_id
        if client is not None:
            self._client = client
        else:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=region)

    def converse(
        self,
        user_text: str,
        tool_config: dict,
        *,
        messages: list[dict] | None = None,
        image_bytes: bytes | None = None,
    ) -> tuple[ToolUseResult | TextResult, list[dict]]:
        """Send user text to Bedrock and return (result, updated_history).

        Parameters
        ----------
        messages:
            Existing conversation history.  A new user turn is appended.
        image_bytes:
            JPEG bytes to attach to the *current* user turn as a vision input.
        """
        tools = tool_config.get("tools", []) if isinstance(tool_config, dict) else []
        log.info(
            "assistant.llm_prompt",
            model_id=self._model_id,
            tool_count=len(tools),
            system_prompt=_truncate(_SYSTEM_PROMPT),
            user_text=_truncate(user_text),
            has_image=image_bytes is not None,
            history_len=len(messages) if messages else 0,
        )

        # Build user content blocks
        user_content: list[dict[str, Any]] = [{"text": user_text}]
        if image_bytes is not None:
            user_content.append(
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": image_bytes},
                    }
                }
            )

        # Start from existing history or empty
        history = list(messages) if messages else []

        # Strip image blocks from older user messages to save context
        for msg in history:
            if msg.get("role") == "user":
                msg["content"] = [
                    block for block in msg.get("content", []) if "image" not in block
                ]

        history.append({"role": "user", "content": user_content})

        converse_args: dict[str, Any] = {
            "modelId": self._model_id,
            "system": [{"text": _SYSTEM_PROMPT}],
            "messages": history,
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

        # Append assistant turn to history
        if content_blocks:
            history.append({"role": "assistant", "content": content_blocks})

        # Check for tool use blocks first
        for block in content_blocks:
            if "toolUse" in block:
                tool_use = block["toolUse"]
                # Collect any accompanying text blocks (LLM preamble)
                text_parts = [b["text"] for b in content_blocks if "text" in b]
                accompanying_text = "\n".join(text_parts).strip()
                log.info(
                    "assistant.llm_response",
                    response_kind="tool_use",
                    stop_reason=stop_reason,
                    tool_name=tool_use.get("name", ""),
                    tool_use_id=tool_use.get("toolUseId", ""),
                    parameters=tool_use.get("input", {}),
                    response_text=_truncate(accompanying_text)
                    if accompanying_text
                    else "",
                )
                return (
                    ToolUseResult(
                        tool_use_id=tool_use["toolUseId"],
                        tool_name=tool_use["name"],
                        parameters=tool_use.get("input", {}),
                        response_text=accompanying_text,
                    ),
                    history,
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
        return TextResult(text=text_result), history

    def classify_follow_up(self, user_text: str) -> bool:
        """Return True when *user_text* is a continuation/question for the assistant."""
        response = self._client.converse(
            modelId=self._model_id,
            system=[{"text": _FOLLOW_UP_CLASSIFIER_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_text}]}],
            inferenceConfig={
                "maxTokens": 3,
                "temperature": 0,
            },
        )

        content_blocks = (
            response.get("output", {}).get("message", {}).get("content", [])
        )
        text_parts: list[str] = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])

        decision = " ".join(text_parts).strip().upper()
        is_follow_up = decision.startswith("YES")
        log.info(
            "assistant.follow_up_classified",
            is_follow_up=is_follow_up,
            user_text=_truncate(user_text),
            classifier_output=_truncate(decision),
        )
        return is_follow_up
