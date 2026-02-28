#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "==> Creating development virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

if [[ "$(uname -s)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    if ! brew ls --versions portaudio >/dev/null 2>&1; then
      echo "==> Installing portaudio (required by PyAudio)"
      brew install portaudio
    fi
  else
    echo "==> Homebrew not found; if PyAudio build fails install portaudio manually."
  fi
fi

echo "==> Installing Python build/runtime dependencies"
python -m pip install maturin
python -m pip install -r "${ROOT_DIR}/demo/requirements.txt"

echo "==> Building and installing viture-sensors extension (editable)"
(
  cd "${ROOT_DIR}/viture-sensors"
  maturin develop
)

echo "==> Verifying install"
python -c "import viture_sensors as vs; print('Loaded:', vs.Device)"

cat <<'EOF'

Setup complete.

Next steps:
  source .venv/bin/activate
  python demo/main.py

EOF
