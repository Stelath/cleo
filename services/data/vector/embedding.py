"""Amazon Nova Multimodal Embeddings via AWS Bedrock.

Provides embed_video, embed_image, and embed_text functions that all map
into the same 1024-dimensional vector space, enabling cross-modal search.
"""

import base64
import json

import boto3
import numpy as np

from services.config import BEDROCK_MODEL_ID, BEDROCK_REGION, EMBEDDING_DIMENSION

_MODEL_ID = BEDROCK_MODEL_ID
_REGION = BEDROCK_REGION
_DEFAULT_DIMENSION = EMBEDDING_DIMENSION
_INDEX_PURPOSE = "GENERIC_INDEX"

# Module-level client — created once, reused across calls.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=_REGION)
    return _client


def _invoke(body: dict) -> np.ndarray:
    """Call Bedrock invoke_model and extract the embedding vector."""
    client = _get_client()
    response = client.invoke_model(
        modelId=_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())
    embeddings = result.get("embeddings")
    if embeddings:
        vec_data = embeddings[0].get("embedding")
    else:
        # Backward compatibility with older response shape.
        vec_data = result.get("embedding")

    if vec_data is None:
        raise ValueError("Bedrock response missing embedding vector")

    vec = np.asarray(vec_data, dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _invoke_for_video(body: dict) -> np.ndarray:
    """Call Bedrock invoke_model and extract the video embedding vector."""
    client = _get_client()
    response = client.invoke_model(
        modelId=_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    result = json.loads(response["body"].read())

    embeddings = result.get("embeddings")
    if embeddings:
        video_embedding = None
        for item in embeddings:
            if item.get("embeddingType") == "VIDEO":
                video_embedding = item.get("embedding")
                break
        if video_embedding is None:
            video_embedding = embeddings[0].get("embedding")
    else:
        # Backward compatibility with older response shape.
        video_embedding = result.get("embedding")

    if video_embedding is None:
        raise ValueError("Bedrock response missing video embedding vector")

    vec = np.asarray(video_embedding, dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _detect_image_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


def embed_video(
    mp4_bytes: bytes,
    dimension: int = _DEFAULT_DIMENSION,
    embedding_purpose: str = _INDEX_PURPOSE,
) -> np.ndarray:
    """Embed a video clip (MP4 bytes) into a vector via Amazon Nova."""
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": embedding_purpose,
            "embeddingDimension": dimension,
            "video": {
                "format": "mp4",
                "embeddingMode": "AUDIO_VIDEO_SEPARATE",
                "source": {"bytes": base64.b64encode(mp4_bytes).decode()},
            },
        },
    }
    return _invoke_for_video(body)


def embed_image(
    image_bytes: bytes,
    dimension: int = _DEFAULT_DIMENSION,
    embedding_purpose: str = _INDEX_PURPOSE,
) -> np.ndarray:
    """Embed an image (raw bytes, e.g. JPEG/PNG) into a vector via Amazon Nova."""
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": embedding_purpose,
            "embeddingDimension": dimension,
            "image": {
                "format": _detect_image_format(image_bytes),
                "source": {"bytes": base64.b64encode(image_bytes).decode()},
            },
        },
    }
    return _invoke(body)


def embed_text(
    text: str,
    dimension: int = _DEFAULT_DIMENSION,
    embedding_purpose: str = _INDEX_PURPOSE,
) -> np.ndarray:
    """Embed a text string into a vector via Amazon Nova."""
    body = {
        "schemaVersion": "nova-multimodal-embed-v1",
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": embedding_purpose,
            "embeddingDimension": dimension,
            "text": {
                "truncationMode": "END",
                "value": text,
            },
        },
    }
    return _invoke(body)
