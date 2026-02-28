#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== Cleo Platform Install ==="

# ── 1. Ensure uv is available ──
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Create venv and install Python dependencies ──
echo ""
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
source .venv/bin/activate
uv sync

# ── 3. Install portaudio (PyAudio dependency) on macOS ──
if [[ "$(uname)" == "Darwin" ]]; then
    if ! brew list portaudio &>/dev/null 2>&1; then
        echo ""
        echo "Installing portaudio via Homebrew (required by PyAudio)..."
        brew install portaudio
    else
        echo "portaudio already installed."
    fi
fi

# ── 4. Build and install viture-sensors (Rust/PyO3 package) ──
echo ""
echo "Building viture-sensors with maturin..."
uv pip install maturin
VITURE_DIR="$PROJECT_ROOT/packages/viture-luma-interop-layer/viture-sensors"
if [[ -d "$VITURE_DIR" ]]; then
    cd "$VITURE_DIR"
    maturin develop --release
    cd "$PROJECT_ROOT"
else
    echo "WARNING: viture-sensors directory not found at $VITURE_DIR"
fi

# ── 5. Generate gRPC stubs from proto files ──
echo ""
echo "Generating gRPC stubs..."
source .venv/bin/activate
bash protos/generate.sh

# ── 6. Verify key imports ──
echo ""
echo "Verifying key imports..."
python -c "
import grpc
print(f'  grpc: {grpc.__version__}')

import faiss
print(f'  faiss: OK (ntotal test: {faiss.IndexFlatIP(128).ntotal})')

import numpy
print(f'  numpy: {numpy.__version__}')

import structlog
print(f'  structlog: {structlog.__version__}')

from generated import sensor_pb2, sensor_pb2_grpc
print('  generated.sensor_pb2: OK')

from generated import transcription_pb2, transcription_pb2_grpc
print('  generated.transcription_pb2: OK')

from data.vector.faiss_db import FaissDB
db = FaissDB(dimension=128)
db.add(numpy.random.randn(128).astype('float32'), {'test': True})
assert db.size == 1
print('  FaissDB: OK')
"

echo ""
echo "=== Install complete ==="
echo "Activate the environment with: source .venv/bin/activate"
echo "Start the platform with:       python -m core.main"
