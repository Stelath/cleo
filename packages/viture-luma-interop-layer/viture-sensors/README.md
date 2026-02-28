# viture-sensors

`viture-sensors` is a Rust + PyO3 integration layer that exposes VITURE XR device functionality to Python.

This package lives in `viture-sensors` at repository root.

## Current Status

- macOS arm64-first implementation with SDK linkage against:
  - `viture-sdk/aarch64/libglasses.dylib`
  - `viture-sdk/aarch64/libcarina_vio.dylib`
- SDK lifecycle + callback bridge implemented in Rust
- Python classes exposed:
  - `Device`
  - `IMU` / `IMUReading`
  - `Camera`
  - `DepthCamera`
  - `Mic`
- Rust examples and hardware-gated tests included

## Package Layout

```text
.
├── Cargo.toml
├── build.rs
├── pyproject.toml
├── src/
│   ├── sdk/
│   ├── camera/
│   ├── microphone/
│   └── python/
├── python/viture_sensors/
├── examples/
├── tests/
└── viture-sdk/
```

## Requirements

- macOS arm64 (Sequoia+ recommended by SDK docs)
- Rust stable toolchain
- Python 3.9+
- VITURE SDK headers and dylibs already vendored in `viture-sdk/`

## Build (Rust)

```bash
cd viture-sensors
cargo check
cargo test
```

## Build (Python extension)

This project is configured for `maturin`.

```bash
cd viture-sensors
python3 -m pip install maturin
maturin develop
python3 -c "import viture_sensors as vs; print(vs.Device)"
```

## Quick Start (Python)

```python
import viture_sensors as vs

device = vs.Device()     # auto-detects first valid product id
device.connect()

reading = device.imu.read()
print(reading.accel, reading.gyro, reading.quaternion)

frame = device.camera.capture()
depth = device.depth.capture("left")
audio = device.mic.record(duration_ms=500, sample_rate=16000)

device.disconnect()
```

## Rust Examples

```bash
cd viture-sensors
cargo run --example imu_dump
cargo run --example camera_preview
cargo run --example mic_record
```

## Testing Strategy

Hardware integration tests are gated by `VITURE_HARDWARE=1`.

```bash
cd viture-sensors
VITURE_HARDWARE=1 cargo test
```

Python integration tests live under `python/tests/` and are also hardware-gated.

## Linux udev

A starter rules file is provided at `99-viture.rules`. Replace placeholder VID/PID values before installing:

```bash
sudo cp viture-sensors/99-viture.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## Notes

- Cross-platform wheel scaffolding is present in CI, but only macOS arm64 runtime is currently first-class in this repository state.
- When using production hardware workflows, validate expected product IDs and model-specific brightness/volume ranges.
