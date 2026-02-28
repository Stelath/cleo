"""Visual demo of the typed FrontendService gRPC API.

Boots the FrontendService in a background thread and fires a choreographed
sequence of typed RPCs so you can watch them render in the Tauri HUD.

Usage:
    # Start the Tauri app first (it connects to gRPC on port 50055), then:
    uv run python scripts/test_frontend_visual.py

    # Or run standalone (prints to log, useful to verify the service boots):
    uv run python scripts/test_frontend_visual.py --no-wait
"""

from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from concurrent import futures

import grpc
import structlog

from generated import frontend_pb2, frontend_pb2_grpc
from services.config import FRONTEND_PORT
from services.frontend_service import FrontendServiceServicer

log = structlog.get_logger()


def _make_tiny_png() -> bytes:
    """Generate a minimal valid 1x1 red PNG (67 bytes) for the image test."""
    import zlib

    signature = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    raw_row = b"\x00\xff\x00\x00"  # filter=none, R=255 G=0 B=0
    idat_data = zlib.compress(raw_row)

    return signature + _chunk(b"IHDR", ihdr_data) + _chunk(b"IDAT", idat_data) + _chunk(b"IEND", b"")


def _stream_image_chunks(
    image_data: bytes,
    *,
    image_id: str,
    mime_type: str,
    position: str,
):
    chunk_size = 64
    chunk_index = 0
    total = len(image_data)
    if total == 0:
        yield frontend_pb2.ImageChunk(
            data=b"",
            image_id=image_id,
            chunk_index=0,
            is_last=True,
            mime_type=mime_type,
            position=position,
        )
        return

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield frontend_pb2.ImageChunk(
            data=image_data[start:end],
            image_id=image_id,
            chunk_index=chunk_index,
            is_last=end >= total,
            mime_type=mime_type,
            position=position,
        )
        chunk_index += 1


def _start_server(servicer: FrontendServiceServicer, port: int) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
    )
    frontend_pb2_grpc.add_FrontendServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def _run_demo(stub: frontend_pb2_grpc.FrontendServiceStub) -> None:
    """Send a choreographed sequence of typed RPCs."""
    delay = 1.5  # seconds between steps so the human can watch

    # ── 1. Welcome notification ──
    print("\n[1/8] ShowNotification — welcome toast")
    stub.ShowNotification(frontend_pb2.NotificationRequest(
        title="Cleo HUD",
        message="Visual demo started — watch the HUD!",
        style="info",
        duration_ms=3000,
    ))
    time.sleep(delay)

    # ── 2. Progress bar ──
    print("[2/8] ShowProgress — animating 0 → 100%")
    for pct in (0.0, 0.2, 0.5, 0.8, 1.0):
        stub.ShowProgress(frontend_pb2.ProgressRequest(
            label=f"Loading… {int(pct * 100)}%",
            value=pct,
            visible=True,
        ))
        time.sleep(0.35)
    time.sleep(0.5)

    # Hide progress
    stub.ShowProgress(frontend_pb2.ProgressRequest(visible=False))
    time.sleep(0.5)

    # ── 3. Text overlay ──
    print("[3/8] ShowText — center text")
    stub.ShowText(frontend_pb2.TextRequest(
        text="Hello from the typed gRPC API!",
        position="center",
    ))
    time.sleep(delay)

    # ── 4. Success notification ──
    print("[4/8] ShowNotification — success toast")
    stub.ShowNotification(frontend_pb2.NotificationRequest(
        title="Progress complete",
        message="All items loaded successfully.",
        style="success",
        duration_ms=2500,
    ))
    time.sleep(delay)

    # ── 5. Image ──
    print("[5/8] StreamImage — tiny red square")
    stub.StreamImage(
        _stream_image_chunks(
            _make_tiny_png(),
            image_id="demo-image-1",
            mime_type="image/png",
            position="center",
        )
    )
    time.sleep(delay)

    # ── 6. Card stack ──
    print("[6/8] ShowCard — two info cards")
    stub.ShowCard(frontend_pb2.CardRequest(
        cards=[
            frontend_pb2.Card(
                title="Cleo Platform",
                subtitle="AR Glasses OS",
                description="Multi-service architecture with gRPC backbone.",
                meta=[
                    frontend_pb2.KeyValue(key="Version", value="0.1.0"),
                    frontend_pb2.KeyValue(key="Services", value="6"),
                ],
            ),
            frontend_pb2.Card(
                title="Frontend Service",
                subtitle="Typed gRPC API",
                description="ShowNotification, StreamImage, ShowProgress, ShowText, ShowCard, Clear.",
                links=[
                    frontend_pb2.Link(label="GitHub", url="https://github.com/Stelath/cleo"),
                ],
            ),
        ],
        position="right",
        duration_ms=5000,
    ))
    time.sleep(delay)

    # ── 7. Warning notification ──
    print("[7/8] ShowNotification — warning toast")
    stub.ShowNotification(frontend_pb2.NotificationRequest(
        title="Heads up",
        message="Clearing the HUD in 2 seconds…",
        style="warning",
        duration_ms=2000,
    ))
    time.sleep(2.5)

    # ── 8. Clear everything ──
    print("[8/8] Clear — reset HUD")
    stub.Clear(frontend_pb2.ClearRequest())
    time.sleep(1.0)

    # Final toast
    stub.ShowNotification(frontend_pb2.NotificationRequest(
        title="Demo finished",
        message="All typed RPCs exercised successfully.",
        style="success",
        duration_ms=4000,
    ))
    print("\nDone! All RPCs sent.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual demo of typed FrontendService RPCs.")
    parser.add_argument(
        "--port", type=int, default=FRONTEND_PORT,
        help=f"gRPC port for FrontendService (default: {FRONTEND_PORT})",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Exit immediately after the demo sequence (don't keep server alive).",
    )
    args = parser.parse_args()

    print(f"Starting FrontendService on port {args.port}…")
    servicer = FrontendServiceServicer()
    server = _start_server(servicer, args.port)
    print(f"FrontendService ready on :{args.port}")
    print("Connect the Tauri HUD app, then press Enter to start the demo…")

    if not args.no_wait:
        try:
            input()
        except EOFError:
            pass

    address = f"localhost:{args.port}"
    channel = grpc.insecure_channel(address)
    stub = frontend_pb2_grpc.FrontendServiceStub(channel)

    try:
        _run_demo(stub)
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc}", file=sys.stderr)
        return 1
    finally:
        channel.close()

    if not args.no_wait:
        print("Server still running. Press Ctrl+C to exit.")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            pass

    servicer.shutdown()
    server.stop(grace=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
