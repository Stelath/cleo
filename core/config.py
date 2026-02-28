"""Centralized configuration constants for the Cleo platform."""

SENSOR_PORT = 50051
TRANSCRIPTION_PORT = 50052
DATA_PORT = 50053
ASSISTANT_PORT = 50054  # Reserved for future use

SENSOR_ADDRESS = f"localhost:{SENSOR_PORT}"
TRANSCRIPTION_ADDRESS = f"localhost:{TRANSCRIPTION_PORT}"
DATA_ADDRESS = f"localhost:{DATA_PORT}"
ASSISTANT_ADDRESS = f"localhost:{ASSISTANT_PORT}"

EMBEDDING_DIMENSION = 1024
BEDROCK_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
BEDROCK_REGION = "us-east-1"

VIDEO_STORAGE_DIR = "data/videos"
