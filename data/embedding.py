"""Amazon Nova Multimodal Embeddings via AWS Bedrock.

Provides embed_video, embed_image, and embed_text functions that all map
into the same 1024-dimensional vector space, enabling cross-modal search.
"""

import base64
import json

import boto3
import numpy as np

from core.config import BEDROCK_MODEL_ID, BEDROCK_REGION, EMBEDDING_DIMENSION

_MODEL_ID = BEDROCK_MODEL_ID
_REGION = BEDROCK_REGION
_DEFAULT_DIMENSION = EMBEDDING_DIMENSION

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
    vec = np.asarray(result["embedding"], dtype=np.float32)
    # Normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_video(mp4_bytes: bytes, dimension: int = _DEFAULT_DIMENSION) -> np.ndarray:
    """Embed a video clip (MP4 bytes) into a vector via Amazon Nova."""
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dimension,
            "video": {
                "format": "mp4",
                "source": {"bytes": base64.b64encode(mp4_bytes).decode()},
            },
        },
    }
    return _invoke(body)


def embed_image(image_bytes: bytes, dimension: int = _DEFAULT_DIMENSION) -> np.ndarray:
    """Embed an image (raw bytes, e.g. JPEG/PNG) into a vector via Amazon Nova."""
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dimension,
            "image": {
                "source": {"bytes": base64.b64encode(image_bytes).decode()},
            },
        },
    }
    return _invoke(body)


def embed_text(text: str, dimension: int = _DEFAULT_DIMENSION) -> np.ndarray:
    """Embed a text string into a vector via Amazon Nova."""
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dimension,
            "text": text,
        },
    }
    return _invoke(body)
