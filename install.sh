#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

run_with_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command -v sudo &>/dev/null; then
        sudo "$@"
    else
        echo "ERROR: sudo is required to run: $*"
        exit 1
    fi
}

activate_venv() {
    if [[ -f ".venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    elif [[ -f ".venv/Scripts/activate" ]]; then
        # shellcheck disable=SC1091
        source .venv/Scripts/activate
    else
        echo "ERROR: Could not find virtual environment activation script."
        exit 1
    fi
}

is_windows_platform() {
    local kernel_name
    kernel_name="$(uname -s)"
    [[ "$kernel_name" == MINGW* || "$kernel_name" == MSYS* || "$kernel_name" == CYGWIN* || "${OS:-}" == "Windows_NT" ]]
}

install_pyaudio_system_deps() {
    local kernel_name
    kernel_name="$(uname -s)"

    case "$kernel_name" in
        Darwin)
            if command -v brew &>/dev/null; then
                if brew list portaudio &>/dev/null 2>&1; then
                    echo "portaudio already installed (Homebrew)."
                else
                    echo ""
                    echo "Installing portaudio via Homebrew (required by PyAudio)..."
                    brew install portaudio
                fi
            else
                echo "WARNING: Homebrew not found. Install portaudio manually if PyAudio build fails."
            fi
            ;;
        Linux)
            local distro_id="" distro_like=""
            if [[ -f /etc/os-release ]]; then
                # shellcheck disable=SC1091
                source /etc/os-release
                distro_id="${ID:-}"
                distro_like="${ID_LIKE:-}"
            fi

            if [[ "$distro_id" == "ubuntu" || "$distro_id" == "debian" || "$distro_like" == *"ubuntu"* || "$distro_like" == *"debian"* ]]; then
                if command -v apt-get &>/dev/null; then
                    echo ""
                    echo "Installing PortAudio build dependencies via apt (required by PyAudio)..."
                    run_with_sudo apt-get update
                    run_with_sudo apt-get install -y portaudio19-dev python3-dev build-essential
                else
                    echo "WARNING: apt-get not found. Install portaudio19-dev manually for PyAudio."
                fi
            else
                echo "WARNING: Linux distro '$distro_id' is not auto-configured."
                echo "         Install PortAudio dev headers manually before running uv sync."
            fi
            ;;
        MINGW* | MSYS* | CYGWIN*)
            echo "Windows detected. PyAudio uses prebuilt wheels on Windows; no PortAudio system package is required."
            ;;
        *)
            if is_windows_platform; then
                echo "Windows detected. PyAudio uses prebuilt wheels on Windows; no PortAudio system package is required."
            else
                echo "WARNING: Unsupported OS '$kernel_name'."
                echo "         Install PortAudio manually if PyAudio fails to build."
            fi
            ;;
    esac
}

echo "=== Cleo Platform Install ==="

# ── 1. Ensure uv is available ──
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── 2. Install PyAudio system dependencies ──
echo ""
echo "Checking platform-specific PyAudio prerequisites..."
install_pyaudio_system_deps

# ── 3. Create venv and install Python dependencies ──
echo ""
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
activate_venv
uv sync

# ── 4. Ensure PyAudio wheel on Windows ──
if is_windows_platform; then
    echo ""
    echo "Ensuring PyAudio is installed from a prebuilt wheel on Windows..."
    uv pip install --only-binary=:all: --no-deps "PyAudio==0.2.14"
fi

# ── 5. Build and install viture-sensors (Rust/PyO3 package) ──
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

# ── 6. Generate gRPC stubs from proto files ──
echo ""
echo "Generating gRPC stubs..."
activate_venv
bash protos/generate.sh

# ── 7. Verify key imports ──
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
if [[ -f ".venv/bin/activate" ]]; then
    echo "Activate the environment with: source .venv/bin/activate"
else
    echo "Activate the environment with: source .venv/Scripts/activate"
fi
echo "Start the platform with:       python -m core.main"
