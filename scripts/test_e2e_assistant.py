"""End-to-end smoke test: starts DataService, tool services, and assistant, then sends commands via gRPC.

Requires valid AWS credentials with Bedrock access.

Usage:
    uv run python scripts/test_e2e_assistant.py
"""

import multiprocessing
import sys

import grpc

from generated import assistant_pb2, assistant_pb2_grpc


def _run_data_service():
    from services.data.service import serve
    serve()


def _run_color_blind():
    from apps.color_blind import serve
    serve()


def _run_object_recognition():
    from apps.object_recognition import serve
    serve()


def _run_navigation_assist():
    from apps.navigation_assist import serve
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
        NAVIGATION_ASSIST_PORT,
        OBJECT_RECOGNITION_PORT,
    )
    print("Waiting for DataService...")
    _wait(f"localhost:{DATA_PORT}")
    print(f"  :{DATA_PORT} ready")

    # Start tool services and assistant
    for fn in [_run_color_blind, _run_object_recognition, _run_navigation_assist, _run_assistant]:
        p = multiprocessing.Process(target=fn, daemon=True)
        p.start()
        procs.append(p)

    print("Waiting for services...")
    for port in [COLOR_BLIND_PORT, OBJECT_RECOGNITION_PORT, NAVIGATION_ASSIST_PORT, ASSISTANT_PORT]:
        _wait(f"localhost:{port}")
        print(f"  :{port} ready")

    # Verify tool registration
    from generated import data_pb2, data_pb2_grpc

    data_channel = grpc.insecure_channel(f"localhost:{DATA_PORT}")
    data_stub = data_pb2_grpc.DataServiceStub(data_channel)

    # Give registration threads a moment to complete
    import time
    time.sleep(3)

    list_resp = data_stub.ListApps(data_pb2.ListAppsRequest(enabled_only=True))
    registered_names = {a.name for a in list_resp.apps}
    print(f"\nRegistered apps: {registered_names}")
    expected = {"color_blindness_assist", "object_recognition", "navigation_assist"}
    if not expected.issubset(registered_names):
        print(f"  WARNING: expected {expected}, got {registered_names}")
    else:
        print("  All 3 tools registered successfully")
    data_channel.close()

    # Run assistant commands
    channel = grpc.insecure_channel(f"localhost:{ASSISTANT_PORT}")
    stub = assistant_pb2_grpc.AssistantServiceStub(channel)

    tests = [
        "help me identify the colors on this sign",
        "what am I looking at right now",
        "how do I get to the nearest coffee shop",
        "what is the capital of France",
    ]

    all_passed = True
    for text in tests:
        print(f"\n--- Command: \"{text}\" ---")
        resp = stub.ProcessCommand(assistant_pb2.CommandRequest(text=text), timeout=30)
        print(f"  success:  {resp.success}")
        print(f"  tool:     {resp.tool_name or '(none)'}")
        print(f"  response: {resp.response_text}")
        if not resp.success:
            all_passed = False

    channel.close()
    for p in procs:
        p.terminate()

    print("\n--- Done ---")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
