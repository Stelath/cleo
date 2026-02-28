"""Tests for data.vector.embedding — Amazon Nova embeddings (mocked boto3)."""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import services.data.vector.embedding as emb


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the module-level Bedrock client between tests."""
    emb._client = None
    yield
    emb._client = None


def _make_mock_response(dimension=1024, embedding_type="TEXT"):
    """Return a mock Bedrock invoke_model response with a random embedding."""
    vec = np.random.randn(dimension).astype(np.float32).tolist()
    body = io.BytesIO(
        json.dumps(
            {
                "embeddings": [
                    {
                        "embeddingType": embedding_type,
                        "embedding": vec,
                    }
                ]
            }
        ).encode()
    )
    return {"body": body}


@patch("services.data.vector.embedding.boto3")
def test_embed_text(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = _make_mock_response(1024)

    result = emb.embed_text("hello world")

    assert result.shape == (1024,)
    assert result.dtype == np.float32
    # Should be normalized
    assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    # Verify the API call
    call_args = mock_client.invoke_model.call_args
    assert call_args.kwargs["modelId"] == "amazon.nova-2-multimodal-embeddings-v1:0"
    body = json.loads(call_args.kwargs["body"])
    assert body["singleEmbeddingParams"]["text"]["value"] == "hello world"
    assert body["singleEmbeddingParams"]["text"]["truncationMode"] == "END"
    assert body["singleEmbeddingParams"]["embeddingDimension"] == 1024


@patch("services.data.vector.embedding.boto3")
def test_embed_image(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = _make_mock_response(1024)

    fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    result = emb.embed_image(fake_image)

    assert result.shape == (1024,)
    assert result.dtype == np.float32

    body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
    assert "image" in body["singleEmbeddingParams"]
    assert body["singleEmbeddingParams"]["image"]["format"] == "png"


@patch("services.data.vector.embedding.boto3")
def test_embed_video(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    audio_vec = np.random.randn(1024).astype(np.float32).tolist()
    video_vec = np.random.randn(1024).astype(np.float32).tolist()
    body = io.BytesIO(
        json.dumps(
            {
                "embeddings": [
                    {"embeddingType": "AUDIO", "embedding": audio_vec},
                    {"embeddingType": "VIDEO", "embedding": video_vec},
                ]
            }
        ).encode()
    )
    mock_client.invoke_model.return_value = {"body": body}

    fake_mp4 = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100
    result = emb.embed_video(fake_mp4)

    assert result.shape == (1024,)
    assert result.dtype == np.float32
    expected = np.asarray(video_vec, dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
    assert "video" in body["singleEmbeddingParams"]
    assert body["singleEmbeddingParams"]["video"]["format"] == "mp4"
    assert body["singleEmbeddingParams"]["video"]["embeddingMode"] == "AUDIO_VIDEO_SEPARATE"


@patch("services.data.vector.embedding.boto3")
def test_custom_dimension(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = _make_mock_response(512)

    result = emb.embed_text("test", dimension=512)
    assert result.shape == (512,)

    body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
    assert body["singleEmbeddingParams"]["embeddingDimension"] == 512


@patch("services.data.vector.embedding.boto3")
def test_client_reused(mock_boto3):
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client
    mock_client.invoke_model.return_value = _make_mock_response()

    emb.embed_text("first")
    # Reset the response body for second call
    mock_client.invoke_model.return_value = _make_mock_response()
    emb.embed_text("second")

    # boto3.client should only be called once (lazy singleton)
    mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")
