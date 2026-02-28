"""Tests for assistant.bedrock — Bedrock Converse API client."""

from unittest.mock import MagicMock

import pytest

from assistant.bedrock import BedrockClient, TextResult, ToolUseResult


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
