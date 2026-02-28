# Getting Started with Cleo

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required by all packages |
| Rust / Cargo | latest stable | Needed to build `viture-sensors` via maturin |
| Homebrew | latest (macOS) | Used to install `portaudio` |

## 1. Install `uv`

[uv](https://github.com/astral-sh/uv) is used for Python dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and set up the environment

```bash
git clone <repo-url> cleo && cd cleo
uv venv .venv
source .venv/bin/activate
uv sync
```

## 3. Install portaudio (macOS)

PyAudio requires the `portaudio` system library:

```bash
brew install portaudio
```

## 4. Build viture-sensors

The `viture-sensors` package is a Rust/PyO3 extension built with [maturin](https://www.maturin.rs/):

```bash
uv pip install maturin
cd packages/viture-luma-interop-layer/viture-sensors
maturin develop --release
cd ../../..
```

## 5. Generate gRPC stubs

The proto definitions live in `protos/`. Generate the Python stubs into `generated/`:

```bash
bash protos/generate.sh
```

> **Note:** gRPC stubs must be generated before running the platform or tests.

## Quick setup (one command)

`install.sh` automates steps 1-5:

```bash
bash install.sh
```

## Running the platform

```bash
source .venv/bin/activate
python -m core.main
```

This starts the sensor service, transcription service, frame processor, and audio-transcription bridge.

## Running tests

Install test dependencies and run the suite:

```bash
uv sync --extra test
pytest
```

Verbose output:

```bash
pytest -v
```

Run a subset by directory:

```bash
pytest tests/data/
pytest tests/transcription/
```

## Project structure

```
cleo/
├── core/                  # Orchestrator and frame processing
│   ├── main.py            #   Entry point — starts all services and threads
│   └── frame_processor.py #   Streams camera frames, embeds, stores in FAISS
├── services/              # gRPC service implementations
│   └── sensor_service.py  #   Camera + audio hardware (VITURE Luma Ultra)
├── transcription/         # ASR / speech-to-text
│   └── parakeet.py        #   NVIDIA Parakeet via NeMo gRPC service
├── data/
│   └── vector/
│       ├── faiss_db.py    #   Thread-safe FAISS wrapper (cosine similarity)
│       └── embedding.py   #   Embedding utilities (placeholder)
├── generated/             # Auto-generated gRPC stubs (do not edit)
├── protos/                # Protobuf definitions
│   ├── sensor.proto
│   ├── transcription.proto
│   └── generate.sh
├── packages/
│   └── viture-luma-interop-layer/
│       └── viture-sensors/ # Rust/PyO3 hardware driver
├── tests/                 # pytest test suite
├── pyproject.toml
└── install.sh             # One-command setup script
```
