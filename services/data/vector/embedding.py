"""Embedding helpers for generic multimodal search and local face recognition."""

import base64
import json

import boto3
import cv2
import numpy as np

from services.config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    EMBEDDING_DIMENSION,
    FACE_EMBEDDING_MODEL,
)

_MODEL_ID = BEDROCK_MODEL_ID
_REGION = BEDROCK_REGION
_DEFAULT_DIMENSION = EMBEDDING_DIMENSION
_INDEX_PURPOSE = "GENERIC_INDEX"

# Module-level client — created once, reused across calls.
_client = None
_face_app = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=_REGION)
    return _client


def _get_face_analyzer():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis

        _face_app = FaceAnalysis(
            name=FACE_EMBEDDING_MODEL,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=-1, det_size=(320, 320))
    return _face_app


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


def _decode_image(image_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


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


def embed_face_image(image_bytes: bytes) -> np.ndarray:
    """Embed a face image using a local InsightFace model."""
    img = _decode_image(image_bytes)
    analyzer = _get_face_analyzer()
    faces = analyzer.get(img)
    if not faces:
        raise ValueError("InsightFace could not detect a face in the provided image")

    best_face = max(
        faces,
        key=lambda face: (
            float(getattr(face, "det_score", 0.0)),
            float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        ),
    )
    embedding = getattr(best_face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(best_face, "embedding", None)
    if embedding is None:
        raise ValueError("InsightFace result did not contain an embedding")

    vec = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec
