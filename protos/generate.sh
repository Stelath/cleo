#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$SCRIPT_DIR"
OUT_DIR="$PROJECT_ROOT/generated"

mkdir -p "$OUT_DIR"

echo "Generating gRPC stubs from proto files..."

python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR"/sensor.proto \
  "$PROTO_DIR"/transcription.proto \
  "$PROTO_DIR"/data.proto \
  "$PROTO_DIR"/assistant.proto \
  "$PROTO_DIR"/tool.proto \
  "$PROTO_DIR"/frontend.proto

# Fix relative imports in generated _grpc.py files (known grpc_tools bug)
# The generated files use "import sensor_pb2" instead of "from . import sensor_pb2"
for grpc_file in "$OUT_DIR"/*_pb2_grpc.py; do
  if [[ -f "$grpc_file" ]]; then
    # Replace "import X_pb2" with "from generated import X_pb2"
    sed -i.bak 's/^import \(.*\)_pb2 as/from generated import \1_pb2 as/' "$grpc_file"
    rm -f "${grpc_file}.bak"
  fi
done

# Ensure __init__.py exists
touch "$OUT_DIR/__init__.py"

echo "Generated stubs in $OUT_DIR"
