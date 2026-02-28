#!/usr/bin/env bash

workspace_root() {
    if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" && -d "${BUILD_WORKSPACE_DIRECTORY}" ]]; then
        printf "%s\n" "${BUILD_WORKSPACE_DIRECTORY}"
        return
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "$script_dir/../.." && pwd)
}

cd_workspace_root() {
    cd "$(workspace_root)"
}

load_dotenv() {
    local env_file
    env_file="$(workspace_root)/.env"
    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$env_file"
        set +a
    fi
}

command_exists() {
    command -v "$1" &>/dev/null
}

ensure_command() {
    local name
    name="$1"
    if ! command_exists "$name"; then
        echo "ERROR: Required command '$name' not found." >&2
        echo "Run 'bash scripts/bootstrap_bazel.sh' first." >&2
        exit 1
    fi
}

pnpm_install() {
    local frontend_dir
    frontend_dir="$1"

    if [[ -f "$frontend_dir/pnpm-lock.yaml" ]]; then
        pnpm --dir "$frontend_dir" install --frozen-lockfile
    else
        pnpm --dir "$frontend_dir" install
    fi
}

ensure_uv_env() {
    ensure_command uv

    if [[ ! -d ".venv" ]]; then
        echo "Creating local virtual environment..."
        uv venv .venv
    fi
}

sync_uv_env() {
    ensure_uv_env
    uv sync --extra test
}

ensure_uv_runtime_deps() {
    ensure_uv_env

    if uv run python -c "import grpc, numpy, structlog, importlib.metadata as md; md.version('pyzbar')" &>/dev/null; then
        return
    fi

    echo "Python runtime dependencies missing; syncing with uv..."
    uv sync --extra test
}

ensure_frontend_deps() {
    local frontend_dir
    frontend_dir="${1:-frontend/viture-luma-display}"

    ensure_command node
    ensure_command pnpm

    if [[ ! -f "$frontend_dir/package.json" ]]; then
        echo "ERROR: Frontend package.json not found at '$frontend_dir'." >&2
        exit 1
    fi

}

generate_protos() {
    uv run bash protos/generate.sh
}

ensure_maturin() {
    if uv run maturin --version &>/dev/null; then
        return
    fi

    uv pip install maturin
}

build_viture_sensors() {
    local viture_manifest
    viture_manifest="packages/viture-luma-interop-layer/viture-sensors/Cargo.toml"

    if [[ ! -f "$viture_manifest" ]]; then
        echo "WARNING: Skipping viture-sensors build; missing manifest: $viture_manifest"
        return
    fi

    ensure_maturin
    uv run maturin develop --release --manifest-path "$viture_manifest"
}

ensure_viture_sensors() {
    if uv run python -c "import viture_sensors" &>/dev/null; then
        return
    fi

    echo "viture_sensors import failed; building extension with maturin..."
    build_viture_sensors
}
