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

FRONTEND_DIR="frontend/viture-luma-display"
ensure_command cargo
ensure_frontend_deps "$FRONTEND_DIR"
pnpm_install "$FRONTEND_DIR"

echo "Building web frontend assets..."
pnpm --dir "$FRONTEND_DIR" run build

echo "Building Tauri Rust crate..."
cargo build --manifest-path "$FRONTEND_DIR/src-tauri/Cargo.toml"
