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
generate_protos
ensure_viture_sensors

exec uv run python -m services.main "$@"
