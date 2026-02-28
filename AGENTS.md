# AGENTS.md

## Commands

Use Bazel as the primary setup/build/run/test entrypoint. Use `uv` for direct Python commands. Use `pnpm` for frontend JavaScript/TypeScript dependency management.

```bash
# One-time bootstrap (installs Bazel/Bazelisk if missing, then runs setup)
bash scripts/bootstrap_bazel.sh

# Setup / install
bazel run //:setup

# Backend
bazel run //:backend_build
bazel run //:backend_run
bazel run //:backend_test

# Frontend
bazel run //:frontend_install
bazel run //:frontend_build
bazel run //:frontend_test
bazel run //:frontend_run
bazel run //:frontend_run -- --web

# Full app (backend + frontend)
bazel run //:run_full_app
bazel run //:run_full_app -- --web

# Direct Python commands (when not using Bazel wrappers)
uv sync --extra test
uv run pytest -v
uv run bash protos/generate.sh
uv run python -m services.main

# Build viture-sensors (Rust/PyO3 driver) — needed after Rust changes
uv run maturin develop --release --manifest-path packages/viture-luma-interop-layer/viture-sensors/Cargo.toml

# Adding frontend dependencies (use pnpm, never npm)
cd frontend/viture-luma-display && pnpm add <package>
```

**Important:** Prefer Bazel targets for orchestration. For direct Python execution, always use `uv run` to ensure the correct Python (3.10–3.12) and project dependencies are used. For frontend dependency management, always use `pnpm` — never `npm` or `yarn`.

## Architecture

Cleo is an AI-powered AR glasses platform for VITURE Luma Ultra glasses. The runtime is a multi-process service graph (`services/main.py`):

- **Sensor Service** (`services/media/sensor_service.py`, subprocess, port 50051) — owns camera/mic capture, maintains in-memory buffers, and exposes `CaptureFrame`, `RecordAudio`, `StreamCamera`, `StreamAudio`, `StreamIMU`
- **Transcription Service** (subprocess, port 50052) — runs Amazon Transcribe ASR and persistently subscribes to sensor audio; writes final transcripts to DataService and triggers AssistantService on wake phrase
- **Data Service** (subprocess, port 50053) — owns SQLite + FAISS + video storage/search RPCs
- **Assistant Service** (subprocess, port 50054) — routes transcribed commands to tools via Bedrock tool-use

Key data layer: `services/data/vector/faiss_db.py` is a thread-safe FAISS IndexFlatIP wrapper (cosine similarity on normalized vectors).

Media fan-out helper: `BroadcastHub` now lives at `services/media/broadcast.py`.

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

Python >=3.10,<3.13. Managed with `uv` and locked in `uv.lock`. The `viture-sensors` package is a Rust/PyO3 extension built with maturin (requires Rust toolchain). PyAudio requires system `portaudio` (macOS: `brew install portaudio`). `pyzbar` requires the `zbar` shared library (macOS: `brew install zbar`).

Frontend JavaScript/TypeScript dependencies are managed with `pnpm` and locked in `pnpm-lock.yaml`. Always use `pnpm add` / `pnpm remove` — never `npm install` or `yarn add`.
