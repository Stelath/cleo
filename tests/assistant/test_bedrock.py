"""Tests for assistant.bedrock — Bedrock Converse API client."""

from unittest.mock import MagicMock

import pytest

from services.assistant.bedrock import BedrockClient, TextResult, ToolUseResult


@pytest.fixture
def mock_boto3_client():
    return MagicMock()


class TestBedrockClient:
    def test_tool_use_response(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_123",
                                "name": "color_blindness_assist",
                                "input": {"query": "what color is this?"},
                            }
                        }
                    ],
                }
            },
        }

        client = BedrockClient(client=mock_boto3_client)
        result = client.converse("help me with colors", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.tool_name == "color_blindness_assist"
        assert result.parameters == {"query": "what color is this?"}
        assert result.tool_use_id == "tu_123"

    def test_text_response(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "The weather is nice today."}],
                }
            },
        }

        client = BedrockClient(client=mock_boto3_client)
        result = client.converse("what's the weather?", tool_config={"tools": []})

        assert isinstance(result, TextResult)
        assert result.text == "The weather is nice today."

    def test_empty_content_returns_empty_text(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {"message": {"role": "assistant", "content": []}},
        }

        client = BedrockClient(client=mock_boto3_client)
        result = client.converse("hello", tool_config={"tools": []})

        assert isinstance(result, TextResult)
        assert result.text == ""

    def test_converse_passes_correct_args(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        }

        tool_config = {"tools": [{"toolSpec": {"name": "t1"}}]}
        client = BedrockClient(client=mock_boto3_client, model_id="test-model")
        client.converse("test input", tool_config=tool_config)

        call_kwargs = mock_boto3_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "test-model"
        assert call_kwargs["toolConfig"] == tool_config
        assert call_kwargs["messages"][0]["content"][0]["text"] == "test input"

    def test_bedrock_error_propagates(self, mock_boto3_client):
        mock_boto3_client.converse.side_effect = Exception("Bedrock unavailable")

        client = BedrockClient(client=mock_boto3_client)
        with pytest.raises(Exception, match="Bedrock unavailable"):
            client.converse("hello", tool_config={"tools": []})

    def test_tool_use_prioritized_over_text(self, mock_boto3_client):
        """When response contains both text and toolUse blocks, toolUse wins."""
        mock_boto3_client.converse.return_value = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me help you with that."},
                        {
                            "toolUse": {
                                "toolUseId": "tu_mixed",
                                "name": "navigation_assist",
                                "input": {"query": "coffee shop"},
                            }
                        },
                    ],
                }
            },
        }

        client = BedrockClient(client=mock_boto3_client)
        result = client.converse("find coffee", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.tool_name == "navigation_assist"

    def test_tool_use_missing_input_defaults_to_empty(self, mock_boto3_client):
        """toolUse block without 'input' key returns empty dict for parameters."""
        mock_boto3_client.converse.return_value = {
            "stopReason": "tool_use",
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_noinput",
                                "name": "object_recognition",
                            }
                        }
                    ],
                }
            },
        }

        client = BedrockClient(client=mock_boto3_client)
        result = client.converse("look", tool_config={"tools": []})

        assert isinstance(result, ToolUseResult)
        assert result.parameters == {}

    def test_system_prompt_passed_to_converse(self, mock_boto3_client):
        mock_boto3_client.converse.return_value = {
            "stopReason": "end_turn",
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        }

        client = BedrockClient(client=mock_boto3_client)
        client.converse("test", tool_config={"tools": []})

        call_kwargs = mock_boto3_client.converse.call_args[1]
        system_texts = [b["text"] for b in call_kwargs["system"]]
        assert any("Cleo" in t for t in system_texts)
