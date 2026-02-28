# Getting Started with Cleo

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10-3.12 | Backend/runtime and tests |
| Rust / Cargo | latest stable | `viture-sensors` + Tauri Rust side |
| Node.js | LTS | Frontend toolchain |
| pnpm | latest | Frontend package manager |
| Bazel / Bazelisk | latest | Unified build/run/test entrypoint |

## Recommended: Bazel bootstrap (one command)

From repository root:

```bash
bash scripts/bootstrap_bazel.sh
```

This script:

1. Installs Bazel (via Bazelisk) if missing
2. Runs `bazel run //:setup`
3. Triggers full project setup through Bazel setup scripts (`uv`, `pnpm`, Rust, stubs, `viture-sensors`)

## Bazel command quick reference

Use these as the main developer entrypoints:

```bash
# setup / install everything
bazel run //:setup

# backend
bazel run //:backend_build
bazel run //:backend_run
bazel run //:backend_test

# frontend
bazel run //:frontend_install
bazel run //:frontend_build
bazel run //:frontend_test
bazel run //:frontend_run

# full stack
bazel run //:run_full_app
```

Optional web-only frontend mode:

```bash
bazel run //:frontend_run -- --web
bazel run //:run_full_app -- --web
```

## Non-Bazel equivalents

If you need to run commands directly:

```bash
# backend
uv run python -m services.main
uv run pytest -v

# frontend
pnpm --dir frontend/viture-luma-display run tauri dev
pnpm --dir frontend/viture-luma-display run build
pnpm --dir frontend/viture-luma-display run test
```

## Hardware and integration test switches

Hardware integration tests are opt-in:

```bash
VITURE_HARDWARE=1 uv run pytest tests/integration/ -v
```

Bedrock video tests are opt-in:

```bash
BEDROCK_TESTS=1 uv run pytest tests/video/ -v
```

Transcription E2E tests are opt-in:

```bash
RUN_TRANSCRIPTION_E2E=1 uv run pytest tests/transcription/ -v
```

## Project structure

```text
cleo/
├── services/               # Runtime + service implementations
├── apps/                   # Tool app services invoked by assistant
├── frontend/
│   └── viture-luma-display # Tauri + React frontend
├── packages/
│   └── viture-luma-interop-layer/
│       └── viture-sensors/ # Rust/PyO3 hardware driver
├── generated/              # Auto-generated gRPC stubs (do not edit)
├── protos/                 # Protobuf definitions
├── tests/                  # pytest test suite
├── BUILD.bazel             # Bazel targets
├── MODULE.bazel            # Bazel module configuration
├── scripts/bootstrap_bazel.sh
└── tools/bazel/            # Bazel orchestration scripts
```
