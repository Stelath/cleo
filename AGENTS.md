# AGENTS.md

## Commands

Use `uv` (Astral's Python package manager) for dependency management. Use `uv run` to execute commands — it automatically resolves the project's virtual environment.

```bash
# Install/sync dependencies
uv sync                  # install from uv.lock
uv sync --extra test     # include pytest + pytest-mock

# Run all tests
uv run pytest -v

# Run a single test file
uv run pytest tests/data/vector/test_faiss_db.py -v

# Run a single test by name
uv run pytest -k "test_add_and_search" -v

# Run tests in a directory
uv run pytest tests/services/ -v

# Regenerate gRPC stubs after editing .proto files
uv run bash protos/generate.sh

# Build viture-sensors (Rust/PyO3 driver) — needed after Rust changes
cd packages/viture-luma-interop-layer/viture-sensors && uv run maturin develop --release && cd -

# Start the platform
uv run python -m services.main
```

**Important:** Always use `uv run` to execute commands — it ensures the correct Python (3.10–3.12) and project dependencies are used.

## Architecture

Cleo is an AI-powered AR glasses platform for VITURE Luma Ultra glasses. The runtime is a multi-process service graph (`services/main.py`):

- **Sensor Service** (subprocess, port 50051) — owns camera/mic capture, maintains in-memory buffers, and exposes `CaptureFrame`, `RecordAudio`, `StreamCamera`, `StreamAudio`, `StreamIMU`
- **Transcription Service** (subprocess, port 50052) — runs Parakeet ASR and persistently subscribes to sensor audio; writes final transcripts to DataService and triggers AssistantService on wake phrase
- **Data Service** (subprocess, port 50053) — owns SQLite + FAISS + video storage/search RPCs
- **Assistant Service** (subprocess, port 50054) — routes transcribed commands to tools via Bedrock tool-use

Key data layer: `services/data/vector/faiss_db.py` is a thread-safe FAISS IndexFlatIP wrapper (cosine similarity on normalized vectors).

## Proto / gRPC

Proto definitions are in `protos/`. Generated Python stubs go to `generated/` via `protos/generate.sh`. The script fixes a known `grpc_tools` relative import bug (rewrites `import X_pb2` → `from generated import X_pb2`). Never edit files in `generated/` directly.

## Testing

pytest is configured in `pyproject.toml` with `pythonpath = ["."]` so all project modules are importable without installing the package.

Test fixtures are split by domain:
- `tests/conftest.py` — general fixtures (`faiss_db`, `random_embedding`)
- `tests/integration/conftest.py` — hardware fixtures (`requires_hardware`, `usb_camera`, `audio_recorder`, `sensor_server`)
- `tests/services/conftest.py` — mock sensor data (`mock_camera_frame`, `mock_audio_chunk`)
- `tests/transcription/conftest.py` — mock transcription data (`mock_transcription_result`)

Hardware integration tests in `tests/integration/` are gated by `VITURE_HARDWARE=1` env var and skipped by default. Run them with:
```bash
VITURE_HARDWARE=1 uv run pytest tests/integration/ -v
```

## Dependencies

Python >=3.10,<3.13. Managed with `uv` and locked in `uv.lock`. The `viture-sensors` package is a Rust/PyO3 extension built with maturin (requires Rust toolchain). PyAudio requires system `portaudio` (macOS: `brew install portaudio`).
