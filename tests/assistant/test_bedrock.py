"""Tests for assistant.bedrock — Bedrock Converse API client."""

from unittest.mock import MagicMock, patch

import pytest

from services.assistant.bedrock import BedrockClient, TextResult, ToolUseResult


@pytest.fixture
def mock_boto3_client():
    return MagicMock()


def _text_response(text="ok"):
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
    }


def _tool_response(tool_use_id="tu_123", name="color_blindness_assist", inp=None, extra_text=None):
    content = []
    if extra_text:
        content.append({"text": extra_text})
    tool_block = {"toolUseId": tool_use_id, "name": name}
    if inp is not None:
        tool_block["input"] = inp
    content.append({"toolUse": tool_block})
    return {
        "stopReason": "tool_use",
        "output": {"message": {"role": "assistant", "content": content}},
    }


class TestBedrockClient:
    def test_tool_use_response(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _tool_response(
            inp={"query": "what color is this?"},
        )

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse("help me with colors", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.tool_name == "color_blindness_assist"
        assert result.parameters == {"query": "what color is this?"}
        assert result.tool_use_id == "tu_123"

    def test_text_response(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response("The weather is nice today.")

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse("what's the weather?", tool_config={"tools": []})

        assert isinstance(result, TextResult)
        assert result.text == "The weather is nice today."

    def test_empty_content_returns_empty_text(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {"message": {"role": "assistant", "content": []}},
        }

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse("hello", tool_config={"tools": []})

        assert isinstance(result, TextResult)
        assert result.text == ""

    def test_converse_passes_correct_args(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response()

        tool_config = {"tools": [{"toolSpec": {"name": "t1"}}]}
        client = BedrockClient(client=mock_boto3_client, model_id="test-model")
        client.converse("test input", tool_config=tool_config)

        call_kwargs = mock_boto3_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "test-model"
        assert call_kwargs["toolConfig"] == tool_config
        assert call_kwargs["messages"][0]["content"][0]["text"] == "test input"

    def test_converse_omits_empty_tool_config(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response()

        client = BedrockClient(client=mock_boto3_client, model_id="test-model")
        client.converse("test input", tool_config={"tools": []})

        call_kwargs = mock_boto3_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "test-model"
        assert "toolConfig" not in call_kwargs

    def test_bedrock_error_propagates(self, mock_boto3_client):
        mock_boto3_client.converse.side_effect = Exception("Bedrock unavailable")

        client = BedrockClient(client=mock_boto3_client)
        with pytest.raises(Exception, match="Bedrock unavailable"):
            client.converse("hello", tool_config={"tools": []})

    def test_tool_use_prioritized_over_text(self, mock_boto3_client):
        """When response contains both text and toolUse blocks, toolUse wins
        and accompanying text is captured in response_text."""
        mock_boto3_client.converse.return_value = _tool_response(
            tool_use_id="tu_mixed",
            name="navigator",
            inp={"query": "coffee shop"},
            extra_text="Let me help you with that.",
        )

        client = BedrockClient(client=mock_boto3_client)
        result, _ = client.converse("find coffee", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.tool_name == "navigator"
        assert result.response_text == "Let me help you with that."

    def test_tool_use_without_text_has_empty_response_text(self, mock_boto3_client):
        """ToolUseResult.response_text is empty when no text blocks accompany the tool call."""
        mock_boto3_client.converse.return_value = _tool_response(
            tool_use_id="tu_only", name="weather", inp={"location": "NYC"},
        )

        client = BedrockClient(client=mock_boto3_client)
        result, _ = client.converse("weather in NYC", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.response_text == ""

    def test_tool_use_missing_input_defaults_to_empty(self, mock_boto3_client):
        """toolUse block without 'input' key returns empty dict for parameters."""
        mock_boto3_client.converse.return_value = _tool_response(
            tool_use_id="tu_noinput", name="color_blindness_assist",
        )

        client = BedrockClient(client=mock_boto3_client)
        result, _ = client.converse("look", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.parameters == {}

    def test_system_prompt_passed_to_converse(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response()

        client = BedrockClient(client=mock_boto3_client)
        client.converse("test", tool_config={"tools": []})

        call_kwargs = mock_boto3_client.converse.call_args[1]
        system_texts = [b["text"] for b in call_kwargs["system"]]
        assert any("Cleo" in t for t in system_texts)

    @patch("services.assistant.bedrock.log")
    def test_logs_prompt_and_text_response(self, mock_log, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response("The weather is nice today.")

        client = BedrockClient(client=mock_boto3_client)
        client.converse("what's the weather?", tool_config={"tools": []})

        assert any(
            call.args[0] == "assistant.llm_prompt" for call in mock_log.info.call_args_list
        )
        assert any(
            call.args[0] == "assistant.llm_response" and call.kwargs.get("response_kind") == "text"
            for call in mock_log.info.call_args_list
        )

    @patch("services.assistant.bedrock.log")
    def test_logs_prompt_and_tool_response(self, mock_log, mock_boto3_client):
        mock_boto3_client.converse.return_value = _tool_response(
            inp={"query": "what color is this?"},
        )

        client = BedrockClient(client=mock_boto3_client)
        client.converse("help me with colors", tool_config={"tools": []})

        assert any(
            call.args[0] == "assistant.llm_prompt" for call in mock_log.info.call_args_list
        )
        assert any(
            call.args[0] == "assistant.llm_response"
            and call.kwargs.get("response_kind") == "tool_use"
            and call.kwargs.get("tool_name") == "color_blindness_assist"
            for call in mock_log.info.call_args_list
        )


class TestBedrockConversationHistory:
    """Tests for multi-turn conversation and image support."""

    def test_returns_updated_history(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response("hi there")

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse("hello", tool_config={"tools": []})

        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_passthrough(self, mock_boto3_client):
        """Existing messages are sent to Bedrock along with the new turn."""
        mock_boto3_client.converse.return_value = _text_response("follow-up answer")

        existing = [
            {"role": "user", "content": [{"text": "first question"}]},
            {"role": "assistant", "content": [{"text": "first answer"}]},
        ]

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse(
            "second question", tool_config={"tools": []}, messages=existing,
        )

        # Returned history: 2 existing + new user + assistant response = 4
        assert len(history) == 4
        assert history[0]["content"][0]["text"] == "first question"
        assert history[2]["content"][0]["text"] == "second question"
        assert history[3]["content"][0]["text"] == "follow-up answer"

    def test_image_bytes_added_to_user_content(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = _text_response("I see a cat")

        client = BedrockClient(client=mock_boto3_client)
        fake_jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data"
        result, history = client.converse(
            "what do you see?", tool_config={"tools": []}, image_bytes=fake_jpeg,
        )

        call_kwargs = mock_boto3_client.converse.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"text": "what do you see?"}
        assert user_content[1]["image"]["format"] == "jpeg"
        assert user_content[1]["image"]["source"]["bytes"] == fake_jpeg

    def test_images_stripped_from_older_messages(self, mock_boto3_client):
        """Image blocks should be removed from older user messages to save context."""
        mock_boto3_client.converse.return_value = _text_response("ok")

        existing = [
            {
                "role": "user",
                "content": [
                    {"text": "old question"},
                    {"image": {"format": "jpeg", "source": {"bytes": b"old-image"}}},
                ],
            },
            {"role": "assistant", "content": [{"text": "old answer"}]},
        ]

        client = BedrockClient(client=mock_boto3_client)
        result, history = client.converse(
            "new question", tool_config={"tools": []}, messages=existing,
        )

        call_kwargs = mock_boto3_client.converse.call_args[1]
        # Old user message should only have text, no image
        old_user_content = call_kwargs["messages"][0]["content"]
        assert len(old_user_content) == 1
        assert old_user_content[0] == {"text": "old question"}

    def test_no_image_when_none(self, mock_boto3_client):
        """When image_bytes=None, no image block is added."""
        mock_boto3_client.converse.return_value = _text_response("ok")

        client = BedrockClient(client=mock_boto3_client)
        client.converse("hello", tool_config={"tools": []}, image_bytes=None)

        call_kwargs = mock_boto3_client.converse.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert len(user_content) == 1
        assert "text" in user_content[0]

    def test_history_accumulates_across_calls(self, mock_boto3_client):
        """Calling converse twice builds up a conversation."""
        mock_boto3_client.converse.return_value = _text_response("answer 1")

        client = BedrockClient(client=mock_boto3_client)
        _, history = client.converse("question 1", tool_config={"tools": []})

        mock_boto3_client.converse.return_value = _text_response("answer 2")
        _, history = client.converse("question 2", tool_config={"tools": []}, messages=history)

        assert len(history) == 4  # q1, a1, q2, a2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"

    def test_does_not_mutate_original_messages(self, mock_boto3_client):
        """converse() should not mutate the caller's messages list."""
        mock_boto3_client.converse.return_value = _text_response("ok")

        original = [
            {"role": "user", "content": [{"text": "q1"}]},
            {"role": "assistant", "content": [{"text": "a1"}]},
        ]
        original_len = len(original)

        client = BedrockClient(client=mock_boto3_client)
        _, history = client.converse("q2", tool_config={"tools": []}, messages=original)

        assert len(original) == original_len  # original not modified
        assert len(history) == 4  # new list has all messages
