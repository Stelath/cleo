"""End-to-end smoke test: starts tool + assistant services, sends commands via gRPC.

Requires valid AWS credentials with Bedrock access.

Usage:
    uv run python scripts/test_e2e_assistant.py
"""

import multiprocessing
import sys

import grpc

from generated import assistant_pb2, assistant_pb2_grpc


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
    for fn in [_run_color_blind, _run_object_recognition, _run_navigation_assist, _run_assistant]:
        p = multiprocessing.Process(target=fn, daemon=True)
        p.start()
        procs.append(p)

    print("Waiting for services...")
    from services.config import (
        ASSISTANT_PORT,
        COLOR_BLIND_PORT,
        NAVIGATION_ASSIST_PORT,
        OBJECT_RECOGNITION_PORT,
    )
    for port in [COLOR_BLIND_PORT, OBJECT_RECOGNITION_PORT, NAVIGATION_ASSIST_PORT, ASSISTANT_PORT]:
        _wait(f"localhost:{port}")
        print(f"  :{port} ready")

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
