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
ensure_uv_runtime_deps

WEBSITE_DIR="website"
ensure_frontend_deps "$WEBSITE_DIR"
pnpm_install "$WEBSITE_DIR"

API_PID=""
API_STOP_TIMEOUT_SECONDS=12
STARTED_API=0

find_website_api_pids() {
    if command -v lsof &>/dev/null; then
        lsof -tiTCP:8008 -sTCP:LISTEN 2>/dev/null || true
        return
    fi
    if command -v fuser &>/dev/null; then
        fuser 8008/tcp 2>/dev/null || true
        return
    fi
}

stop_existing_website_api() {
    local pids
    pids="$(find_website_api_pids)"
    if [[ -z "$pids" ]]; then
        return
    fi

    echo "Stopping existing website API listener(s) on port 8008: $pids"
    for pid in $pids; do
        stop_process "$pid" "website API" "$API_STOP_TIMEOUT_SECONDS"
    done
}

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

    if [[ "$STARTED_API" -eq 1 ]]; then
        stop_process "$API_PID" "website API" "$API_STOP_TIMEOUT_SECONDS"
    fi

    exit "$exit_code"
}
trap cleanup EXIT INT TERM

stop_existing_website_api

echo "Starting website API..."
uv run python -m services.website_api &
API_PID=$!
STARTED_API=1

sleep 1
if ! kill -0 "$API_PID" &>/dev/null; then
    echo "ERROR: Website API exited before the website launched." >&2
    wait "$API_PID"
fi

echo "Starting website in web mode (vite dev server)..."
if [[ $# -gt 0 ]]; then
    pnpm --dir "$WEBSITE_DIR" run dev -- "$@"
    exit $?
fi
pnpm --dir "$WEBSITE_DIR" run dev
