#!/usr/bin/env bash
set -euo pipefail

COMMON_SH=""
if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" && -f "${BUILD_WORKSPACE_DIRECTORY}/tools/bazel/common.sh" ]]; then
    COMMON_SH="${BUILD_WORKSPACE_DIRECTORY}/tools/bazel/common.sh"
elif [[ -n "${RUNFILES_DIR:-}" && -f "${RUNFILES_DIR}/_main/tools/bazel/common.sh" ]]; then
    COMMON_SH="${RUNFILES_DIR}/_main/tools/bazel/common.sh"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/common.sh" ]]; then
        COMMON_SH="$SCRIPT_DIR/common.sh"
    fi
fi

if [[ -z "$COMMON_SH" ]]; then
    echo "ERROR: Could not locate tools/bazel/common.sh" >&2
    exit 1
fi

# shellcheck source=tools/bazel/common.sh
source "$COMMON_SH"

cd_workspace_root

APT_UPDATED=0

run_with_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif command_exists sudo; then
        sudo "$@"
    else
        echo "ERROR: sudo is required to run: $*"
        exit 1
    fi
}

apt_install() {
    if ! command_exists apt-get; then
        echo "ERROR: apt-get not found. Install required packages manually."
        exit 1
    fi

    if [[ "$APT_UPDATED" -eq 0 ]]; then
        run_with_sudo apt-get update
        APT_UPDATED=1
    fi

    run_with_sudo apt-get install -y "$@"
}

brew_install_if_missing() {
    local formula
    formula="$1"

    if brew list "$formula" &>/dev/null 2>&1; then
        echo "$formula already installed (Homebrew)."
        return
    fi

    echo "Installing $formula via Homebrew..."
    brew install "$formula"
}

is_windows_platform() {
    local kernel_name
    kernel_name="$(uname -s)"
    [[ "$kernel_name" == MINGW* || "$kernel_name" == MSYS* || "$kernel_name" == CYGWIN* || "${OS:-}" == "Windows_NT" ]]
}

is_linux() {
    [[ "$(uname -s)" == "Linux" ]]
}

is_macos() {
    [[ "$(uname -s)" == "Darwin" ]]
}

ensure_uv() {
    if command_exists uv; then
        echo "uv: $(uv --version)"
        return
    fi

    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! command_exists uv; then
        echo "ERROR: uv installation succeeded but 'uv' is not on PATH."
        echo "       Add '$HOME/.local/bin' to PATH and rerun 'bazel run //:setup'."
        exit 1
    fi

    echo "uv: $(uv --version)"
}

ensure_rust_toolchain() {
    if command_exists rustup; then
        echo "rustup: $(rustup --version)"
    elif command_exists cargo; then
        echo "cargo: $(cargo --version)"
    elif is_windows_platform; then
        echo "WARNING: Rust toolchain not found on Windows."
        echo "         Install rustup manually from https://rustup.rs"
        return
    else
        echo "Installing Rust toolchain via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    fi

    export PATH="$HOME/.cargo/bin:$PATH"

    if command_exists rustup; then
        rustup toolchain install stable --profile minimal
        rustup default stable
    fi

    if command_exists cargo; then
        echo "cargo: $(cargo --version)"
    fi
}

ensure_node() {
    if command_exists node; then
        echo "node: $(node --version)"
        return
    fi

    if is_macos; then
        if command_exists brew; then
            brew_install_if_missing node
        else
            echo "ERROR: Node.js is required for frontend builds."
            echo "       Install Homebrew (or Node manually) and rerun 'bazel run //:setup'."
            exit 1
        fi
    elif is_linux; then
        echo "Installing Node.js via apt..."
        apt_install nodejs npm
    elif is_windows_platform; then
        echo "ERROR: Node.js is required for frontend builds on Windows."
        echo "       Install Node.js from https://nodejs.org and rerun 'bazel run //:setup'."
        exit 1
    else
        echo "ERROR: Unsupported OS for automatic Node.js install."
        exit 1
    fi

    if ! command_exists node; then
        echo "ERROR: Failed to install Node.js."
        exit 1
    fi

    echo "node: $(node --version)"
}

ensure_pnpm() {
    if command_exists pnpm; then
        echo "pnpm: $(pnpm --version)"
        return
    fi

    if command_exists corepack; then
        echo "Enabling pnpm via corepack..."
        corepack enable
        corepack prepare pnpm@latest --activate
        hash -r
    elif command_exists npm; then
        echo "Installing pnpm via npm..."
        if is_windows_platform; then
            npm install -g pnpm
        else
            run_with_sudo npm install -g pnpm
        fi
        hash -r
    else
        echo "ERROR: Could not install pnpm (neither corepack nor npm found)."
        exit 1
    fi

    if ! command_exists pnpm; then
        echo "ERROR: Failed to install pnpm."
        exit 1
    fi

    echo "pnpm: $(pnpm --version)"
}

install_pyaudio_system_deps() {
    local kernel_name
    kernel_name="$(uname -s)"

    case "$kernel_name" in
        Darwin)
            if command_exists brew; then
                brew_install_if_missing portaudio
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
                echo "Installing PortAudio build dependencies via apt (required by PyAudio)..."
                apt_install portaudio19-dev python3-dev build-essential pkg-config
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

install_protobuf_system_deps() {
    if command_exists protoc; then
        echo "protoc: $(protoc --version)"
        return
    fi

    if is_macos; then
        if command_exists brew; then
            brew_install_if_missing protobuf
        else
            echo "WARNING: Homebrew not found. Install protoc manually for tonic/grpc builds."
            return
        fi
    elif is_linux; then
        if command_exists apt-get; then
            echo "Installing protobuf compiler via apt..."
            apt_install protobuf-compiler
        else
            echo "WARNING: apt-get not found. Install protoc manually for tonic/grpc builds."
            return
        fi
    fi

    if command_exists protoc; then
        echo "protoc: $(protoc --version)"
    fi
}

install_rust_build_system_deps() {
    if is_linux; then
        if command_exists apt-get; then
            echo "Installing Rust/bindgen dependencies via apt..."
            apt_install clang libclang-dev
        else
            echo "WARNING: apt-get not found. Install clang/libclang manually for bindgen builds."
        fi
    elif is_macos; then
        if ! command_exists clang; then
            echo "WARNING: clang not found. Install Xcode Command Line Tools with: xcode-select --install"
        fi
    fi
}

install_tauri_system_deps() {
    if ! is_linux; then
        return
    fi

    if ! command_exists apt-get || ! command_exists apt-cache; then
        echo "WARNING: apt not found. Install Linux Tauri GUI dependencies manually."
        return
    fi

    local webkit_pkg="libwebkit2gtk-4.1-dev"
    local indicator_pkg="libayatana-appindicator3-dev"

    if ! apt-cache show "$webkit_pkg" &>/dev/null; then
        webkit_pkg="libwebkit2gtk-4.0-dev"
    fi

    if ! apt-cache show "$indicator_pkg" &>/dev/null; then
        indicator_pkg="libappindicator3-dev"
    fi

    echo "Installing Linux GUI dependencies for Tauri frontend..."
    apt_install libgtk-3-dev "$webkit_pkg" "$indicator_pkg" librsvg2-dev patchelf
}

install_python_dependencies() {
    echo ""
    echo "Creating virtual environment and installing Python dependencies..."
    uv venv .venv
    uv sync --extra test

    if is_windows_platform; then
        echo ""
        echo "Ensuring PyAudio is installed from a prebuilt wheel on Windows..."
        uv pip install --only-binary=:all: --no-deps "PyAudio==0.2.14"
    fi
}

install_frontend_dependencies() {
    local frontend_dir
    frontend_dir="frontend/viture-luma-display"

    if [[ ! -d "$frontend_dir" ]]; then
        echo "WARNING: Frontend directory not found at $frontend_dir"
        return
    fi

    echo ""
    echo "Installing frontend dependencies with pnpm..."
    pnpm_install "$frontend_dir"
}

verify_key_imports() {
    echo ""
    echo "Verifying key imports..."
    uv run python -c "
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

from services.data.vector.faiss_db import FaissDB
db = FaissDB(dimension=128)
db.add(numpy.random.randn(128).astype('float32'), {'test': True})
assert db.size == 1
print('  FaissDB: OK')
"
}

echo "=== Cleo Platform Setup (Bazel) ==="

echo ""
echo "Ensuring toolchain prerequisites..."
ensure_uv
install_pyaudio_system_deps
install_protobuf_system_deps
install_rust_build_system_deps
install_tauri_system_deps
ensure_rust_toolchain
ensure_node
ensure_pnpm

install_python_dependencies
install_frontend_dependencies
generate_protos
build_viture_sensors
verify_key_imports

echo ""
echo "=== Setup complete ==="
echo "Run backend with:        bazel run //:backend_run"
echo "Run frontend with:       bazel run //:frontend_run"
echo "Run full app with:       bazel run //:run_full_app"
