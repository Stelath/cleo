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

MODE="tauri"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --web)
            MODE="web"
            ;;
        --tauri)
            MODE="tauri"
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

cd_workspace_root
ensure_uv_runtime_deps
generate_protos
ensure_viture_sensors

FRONTEND_DIR="frontend/viture-luma-display"
WEBSITE_DIR="website"
ensure_frontend_deps "$FRONTEND_DIR"
pnpm_install "$FRONTEND_DIR"
ensure_frontend_deps "$WEBSITE_DIR"
pnpm_install "$WEBSITE_DIR"

BACKEND_PID=""
WEBSITE_PID=""
BACKEND_STOP_TIMEOUT_SECONDS=12
WEBSITE_STOP_TIMEOUT_SECONDS=12

stop_process() {
    local pid
    local label
    local timeout_seconds

    pid="$1"
    label="$2"
    timeout_seconds="$3"

    if [[ -z "$pid" ]] || ! kill -0 "$pid" &>/dev/null; then
        return
    fi

    echo "Stopping ${label} process (${pid})..."
    kill "$pid" &>/dev/null || true

    local elapsed
    elapsed=0
    while kill -0 "$pid" &>/dev/null && [[ "$elapsed" -lt "$timeout_seconds" ]]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done

    if kill -0 "$pid" &>/dev/null; then
        echo "${label} did not exit after ${timeout_seconds}s; force-killing..."
        kill -9 "$pid" &>/dev/null || true
    fi

    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local exit_code
    exit_code=$?
    trap - EXIT INT TERM

    stop_process "$WEBSITE_PID" "website" "$WEBSITE_STOP_TIMEOUT_SECONDS"
    stop_process "$BACKEND_PID" "backend" "$BACKEND_STOP_TIMEOUT_SECONDS"

    exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Starting backend services..."
uv run python -m services.main &
BACKEND_PID=$!

sleep 2
if ! kill -0 "$BACKEND_PID" &>/dev/null; then
    echo "ERROR: Backend exited before frontend launch." >&2
    wait "$BACKEND_PID"
fi

echo "Starting website..."
pnpm --dir "$WEBSITE_DIR" run dev &
WEBSITE_PID=$!

sleep 2
if ! kill -0 "$WEBSITE_PID" &>/dev/null; then
    echo "ERROR: Website exited before frontend launch." >&2
    wait "$WEBSITE_PID"
fi

if [[ "$MODE" == "web" ]]; then
    echo "Starting frontend (web mode)..."
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        pnpm --dir "$FRONTEND_DIR" run dev -- "${EXTRA_ARGS[@]}"
    else
        pnpm --dir "$FRONTEND_DIR" run dev
    fi
else
    ensure_command cargo
    echo "Starting frontend (Tauri mode)..."
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        pnpm --dir "$FRONTEND_DIR" run tauri dev -- "${EXTRA_ARGS[@]}"
    else
        pnpm --dir "$FRONTEND_DIR" run tauri dev
    fi
fi
