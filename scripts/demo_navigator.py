"""Navigator full-pipeline demo: mock sensor serves static images, navigator analyzes them via Bedrock VLM.

Starts a mock sensor service (serves demo images as camera frames), DataService,
navigator tool, and assistant. Sends a navigate command and prints the VLM
guidance for each frame.

Requires valid AWS credentials with Bedrock access.

Usage:
    uv run python scripts/generate_demo_images.py   # one-time
    uv run python scripts/demo_navigator.py
"""

import multiprocessing
import os
import queue
import time
from concurrent import futures
from pathlib import Path

import cv2
import grpc
import numpy as np

from generated import sensor_pb2, sensor_pb2_grpc, assistant_pb2, assistant_pb2_grpc

DEMO_IMAGES_DIR = Path(__file__).parent / "demo_images"

# ── Mock sensor service ──────────────────────────────────────────────────────


class MockSensorServicer(sensor_pb2_grpc.SensorServiceServicer):
    """Serves static images as camera frames on StreamCamera."""

    def __init__(self, image_paths: list[Path]):
        self._frames: list[sensor_pb2.CameraFrame] = []
        for p in image_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {p}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            self._frames.append(
                sensor_pb2.CameraFrame(
                    data=img_rgb.tobytes(),
                    width=w,
                    height=h,
                    timestamp=time.time(),
                )
            )
        print(f"  MockSensor loaded {len(self._frames)} images", flush=True)

    def StreamCamera(self, request, context):
        fps = request.fps if request.fps > 0 else 1.0
        interval = 1.0 / fps
        idx = 0
        while context.is_active():
            frame = self._frames[idx % len(self._frames)]
            frame.timestamp = time.time()
            yield frame
            idx += 1
            time.sleep(interval)

    def CaptureFrame(self, request, context):
        frame = self._frames[0]
        frame.timestamp = time.time()
        return frame


def _run_mock_sensor(port: int, image_paths: list[str]):
    paths = [Path(p) for p in image_paths]
    servicer = MockSensorServicer(paths)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_send_message_length", 32 * 1024 * 1024)],
    )
    sensor_pb2_grpc.add_SensorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"  MockSensor started on :{port}", flush=True)
    server.wait_for_termination()


# ── Service launchers ────────────────────────────────────────────────────────


def _run_data_service():
    from services.data.service import serve
    serve()


def _run_navigator():
    from apps.navigator import serve
    serve()


def _run_assistant():
    from services.assistant.service import serve
    serve()


def _wait(addr, timeout=15):
    ch = grpc.insecure_channel(addr)
    grpc.channel_ready_future(ch).result(timeout=timeout)
    ch.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    from services.config import (
        ASSISTANT_PORT,
        DATA_PORT,
        NAVIGATOR_PORT,
        SENSOR_PORT,
    )

    # Collect demo images
    image_paths = sorted(DEMO_IMAGES_DIR.glob("*.jpg")) + sorted(DEMO_IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print("No images found in scripts/demo_images/. Run:", flush=True)
        print("  uv run python scripts/generate_demo_images.py", flush=True)
        os._exit(1)

    print(f"\nDemo images: {[p.name for p in image_paths]}", flush=True)

    # Disable stale app registrations
    from generated import data_pb2, data_pb2_grpc

    procs = []

    # 1. Mock sensor
    sensor_proc = multiprocessing.Process(
        target=_run_mock_sensor,
        args=(SENSOR_PORT, [str(p) for p in image_paths]),
        daemon=True,
    )
    sensor_proc.start()
    procs.append(sensor_proc)

    # 2. Data service
    data_proc = multiprocessing.Process(target=_run_data_service, daemon=True)
    data_proc.start()
    procs.append(data_proc)

    print("\nWaiting for services...", flush=True)
    _wait(f"localhost:{SENSOR_PORT}")
    print(f"  :{SENSOR_PORT} mock sensor ready", flush=True)
    _wait(f"localhost:{DATA_PORT}")
    print(f"  :{DATA_PORT} data ready", flush=True)

    # Disable stale registrations before starting tools
    data_channel = grpc.insecure_channel(f"localhost:{DATA_PORT}")
    data_stub = data_pb2_grpc.DataServiceStub(data_channel)
    for stale in ["object_recognition", "navigation_assist"]:
        data_stub.SetAppEnabled(data_pb2.SetAppEnabledRequest(name=stale, enabled=False))
    data_channel.close()

    # 3. Navigator + assistant
    for fn in [_run_navigator, _run_assistant]:
        p = multiprocessing.Process(target=fn, daemon=True)
        p.start()
        procs.append(p)

    _wait(f"localhost:{NAVIGATOR_PORT}")
    print(f"  :{NAVIGATOR_PORT} navigator ready", flush=True)
    _wait(f"localhost:{ASSISTANT_PORT}")
    print(f"  :{ASSISTANT_PORT} assistant ready", flush=True)

    # Give registration a moment
    time.sleep(2)

    # ── Send navigate command via assistant ───────────────────────────────
    channel = grpc.insecure_channel(f"localhost:{ASSISTANT_PORT}")
    stub = assistant_pb2_grpc.AssistantServiceStub(channel)

    print("\n=== Starting navigator via assistant ===", flush=True)
    try:
        resp = stub.ProcessCommand(
            assistant_pb2.CommandRequest(text="help me navigate, tell me about obstacles and hazards"),
            timeout=30,
        )
        print(f"  tool:     {resp.tool_name or '(none)'}", flush=True)
        print(f"  response: {resp.response_text}", flush=True)
    except grpc.RpcError as e:
        print(f"  FAIL: {e.code().name}: {e.details()}", flush=True)

    # ── Wait for navigator to process frames and print guidance ──────────
    print("\n=== Waiting for navigator guidance (watching for ~15s) ===", flush=True)
    print("  (Navigator streams frames from mock sensor → Bedrock VLM → guidance)\n", flush=True)

    # Poll navigator guidance by reading structlog output — or just wait and
    # let the navigator logs show the guidance. The navigator logs
    # "navigator.guidance" with the first 100 chars of each guidance.
    time.sleep(15)

    # ── Stop navigator ───────────────────────────────────────────────────
    print("\n=== Stopping navigator ===", flush=True)
    try:
        resp = stub.ProcessCommand(
            assistant_pb2.CommandRequest(text="stop navigating"),
            timeout=30,
        )
        print(f"  tool:     {resp.tool_name or '(none)'}", flush=True)
        print(f"  response: {resp.response_text}", flush=True)
    except grpc.RpcError as e:
        print(f"  FAIL: {e.code().name}: {e.details()}", flush=True)

    channel.close()

    # ── Cleanup ──────────────────────────────────────────────────────────
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=5)

    print("\n--- Demo complete ---", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
