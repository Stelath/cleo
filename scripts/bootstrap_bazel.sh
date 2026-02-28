#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_SETUP=1

usage() {
    cat <<'EOF'
Usage: bash scripts/bootstrap_bazel.sh [--no-setup]

Options:
  --no-setup   Install Bazel/Bazelisk only (skip bazel run //:setup)
  --help       Show this help message
EOF
}

command_exists() {
    command -v "$1" &>/dev/null
}

ensure_local_bin_on_path() {
    mkdir -p "$HOME/.local/bin"
    export PATH="$HOME/.local/bin:$PATH"
}

platform_key() {
    case "$(uname -s)" in
        Darwin)
            echo "darwin"
            ;;
        Linux)
            echo "linux"
            ;;
        *)
            echo "unsupported"
            ;;
    esac
}

arch_key() {
    case "$(uname -m)" in
        x86_64 | amd64)
            echo "amd64"
            ;;
        arm64 | aarch64)
            echo "arm64"
            ;;
        *)
            echo "unsupported"
            ;;
    esac
}

install_bazelisk_binary() {
    local os_key arch url out
    os_key="$(platform_key)"
    arch="$(arch_key)"

    if [[ "$os_key" == "unsupported" || "$arch" == "unsupported" ]]; then
        echo "ERROR: Unsupported platform for automatic Bazel install." >&2
        echo "       Install Bazelisk manually: https://github.com/bazelbuild/bazelisk" >&2
        exit 1
    fi

    url="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-${os_key}-${arch}"
    out="$HOME/.local/bin/bazel"

    echo "Downloading Bazelisk from $url"
    curl -fsSL "$url" -o "$out"
    chmod +x "$out"
}

ensure_bazel() {
    ensure_local_bin_on_path

    if command_exists bazel; then
        echo "bazel: $(bazel --version)"
        return
    fi

    if command_exists bazelisk; then
        ln -sf "$(command -v bazelisk)" "$HOME/.local/bin/bazel"
    fi

    if command_exists bazel; then
        echo "bazel: $(bazel --version)"
        return
    fi

    if [[ "$(platform_key)" == "darwin" ]] && command_exists brew; then
        echo "Installing Bazelisk with Homebrew..."
        brew install bazelisk
        if command_exists bazelisk && ! command_exists bazel; then
            ln -sf "$(command -v bazelisk)" "$HOME/.local/bin/bazel"
        fi
    fi

    if ! command_exists bazel; then
        install_bazelisk_binary
    fi

    if ! command_exists bazel; then
        echo "ERROR: Unable to install Bazel automatically." >&2
        exit 1
    fi

    echo "bazel: $(bazel --version)"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-setup)
            RUN_SETUP=0
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

echo "=== Bazel Bootstrap ==="
ensure_bazel

if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" && "$RUN_SETUP" -eq 1 ]]; then
    echo "Detected Bazel-run environment; skipping nested 'bazel run //:setup'."
    RUN_SETUP=0
fi

if [[ "$RUN_SETUP" -eq 1 ]]; then
    echo ""
    echo "Running project setup via Bazel..."
    bazel run //:setup
    echo ""
    echo "Done. Try: bazel run //:run_full_app"
else
    echo ""
    echo "Bazel installed. Next step: bazel run //:setup"
fi
