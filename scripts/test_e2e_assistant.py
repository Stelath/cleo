"""End-to-end smoke test: starts DataService, tool services, and assistant, then sends commands via gRPC.

Requires valid AWS credentials with Bedrock access.

Usage:
    uv run python scripts/test_e2e_assistant.py
"""

import multiprocessing
import os
import time

import grpc

from generated import assistant_pb2, assistant_pb2_grpc


def _run_data_service():
    from services.data.service import serve
    serve()


def _run_color_blind():
    from apps.color_blind import serve
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


def main():
    procs = []

    # Start DataService first — tools register with it on boot
    data_proc = multiprocessing.Process(target=_run_data_service, daemon=True)
    data_proc.start()
    procs.append(data_proc)

    from services.config import (
        ASSISTANT_PORT,
        COLOR_BLIND_PORT,
        DATA_PORT,
        NAVIGATOR_PORT,
    )
    print("Waiting for DataService...", flush=True)
    _wait(f"localhost:{DATA_PORT}")
    print(f"  :{DATA_PORT} ready", flush=True)

    # Disable stale app registrations from previous runs
    from generated import data_pb2, data_pb2_grpc

    data_channel = grpc.insecure_channel(f"localhost:{DATA_PORT}")
    data_stub = data_pb2_grpc.DataServiceStub(data_channel)
    for stale in ["object_recognition", "navigation_assist"]:
        data_stub.SetAppEnabled(data_pb2.SetAppEnabledRequest(name=stale, enabled=False))
    print("Disabled stale app registrations", flush=True)

    # Start tool services and assistant
    for fn in [_run_color_blind, _run_navigator, _run_assistant]:
        p = multiprocessing.Process(target=fn, daemon=True)
        p.start()
        procs.append(p)

    print("Waiting for services...", flush=True)
    for port in [COLOR_BLIND_PORT, NAVIGATOR_PORT, ASSISTANT_PORT]:
        _wait(f"localhost:{port}")
        print(f"  :{port} ready", flush=True)

    # Give registration threads a moment to complete
    time.sleep(3)

    list_resp = data_stub.ListApps(data_pb2.ListAppsRequest(enabled_only=True))
    registered_names = {a.name for a in list_resp.apps}
    print(f"\nRegistered apps: {registered_names}", flush=True)
    expected = {"color_blindness_assist", "navigator"}
    if not expected.issubset(registered_names):
        print(f"  FAIL: expected {expected}, got {registered_names}", flush=True)
    else:
        print("  All 2 tools registered successfully", flush=True)

    stale_still_present = registered_names & {"object_recognition", "navigation_assist"}
    if stale_still_present:
        print(f"  FAIL: stale apps still enabled: {stale_still_present}", flush=True)

    data_channel.close()

    # Run assistant commands
    channel = grpc.insecure_channel(f"localhost:{ASSISTANT_PORT}")
    stub = assistant_pb2_grpc.AssistantServiceStub(channel)

    # Navigator's "start" action returns immediately (sensor connects in
    # background), so we can verify Bedrock routes to it without the sensor.
    # Color-blind tool needs the sensor synchronously, so we skip it here.
    tests = [
        ("help me navigate around obstacles", "navigator"),
        ("what is the capital of France", None),
    ]

    all_passed = True
    try:
        for text, expected_tool in tests:
            print(f"\n--- Command: \"{text}\" ---", flush=True)
            try:
                resp = stub.ProcessCommand(
                    assistant_pb2.CommandRequest(text=text), timeout=30
                )
                actual_tool = resp.tool_name or None
                print(f"  success:  {resp.success}", flush=True)
                print(f"  tool:     {actual_tool or '(none)'}", flush=True)
                print(f"  response: {resp.response_text}", flush=True)
                if not resp.success:
                    all_passed = False
                if expected_tool and actual_tool != expected_tool:
                    print(f"  FAIL: expected tool '{expected_tool}', got '{actual_tool}'", flush=True)
                    all_passed = False
            except grpc.RpcError as e:
                print(f"  FAIL: gRPC error: {e.code().name}: {e.details()}", flush=True)
                all_passed = False
    finally:
        channel.close()
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)

    print("\n--- Done ---", flush=True)
    # Use os._exit to avoid multiprocessing atexit handler blocking
    os._exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
