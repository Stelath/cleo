"""Live microphone transcription against the Parakeet gRPC service."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from concurrent import futures

import grpc
import pyaudio

from generated import transcription_pb2, transcription_pb2_grpc
from transcription import parakeet

_MAX_GRPC_MESSAGE_BYTES = 16 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the transcription service and stream microphone audio to it.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host.")
    parser.add_argument("--port", type=int, default=50052, help="Server bind port.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Microphone sample rate in Hz.",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=250,
        help="Chunk size to capture and stream in milliseconds.",
    )
    parser.add_argument(
        "--input-device-index",
        type=int,
        default=None,
        help="Optional PyAudio input device index.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone devices and exit.",
    )
    return parser.parse_args()


def _list_input_devices() -> int:
    audio = pyaudio.PyAudio()
    try:
        found = False
        for idx in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            found = True
            name = str(info.get("name", f"Device {idx}"))
            rate = int(info.get("defaultSampleRate", 0))
            print(f"{idx}: {name} (default_rate={rate})")
        if not found:
            print("No input devices found.")
    finally:
        audio.terminate()
    return 0


def _start_server(host: str, port: int) -> tuple[grpc.Server, str]:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ],
    )
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        parakeet.TranscriptionServiceServicer(),
        server,
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind transcription service on {host}:{port}")
    server.start()
    return server, f"{host}:{bound_port}"


def _microphone_requests(
    stop_event: threading.Event,
    sample_rate: int,
    chunk_ms: int,
    input_device_index: int | None,
):
    frames_per_buffer = max(1, int(sample_rate * chunk_ms / 1000))
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=frames_per_buffer,
    )

    try:
        while not stop_event.is_set():
            chunk = stream.read(frames_per_buffer, exception_on_overflow=False)
            yield transcription_pb2.AudioInput(
                audio_data=chunk,
                sample_rate=sample_rate,
                is_final=False,
            )
    finally:
        try:
            stream.stop_stream()
            stream.close()
        finally:
            audio.terminate()


def _run_client(
    address: str,
    stop_event: threading.Event,
    sample_rate: int,
    chunk_ms: int,
    input_device_index: int | None,
) -> None:
    with grpc.insecure_channel(
        address,
        options=[
            ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ],
    ) as channel:
        grpc.channel_ready_future(channel).result(timeout=30)
        stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
        responses = stub.TranscribeStream(
            _microphone_requests(
                stop_event=stop_event,
                sample_rate=sample_rate,
                chunk_ms=chunk_ms,
                input_device_index=input_device_index,
            )
        )

        for response in responses:
            label = "partial" if response.is_partial else "final"
            text = response.text.strip()
            if not text:
                continue
            print(f"[{label} {response.start_time:.2f}-{response.end_time:.2f}s] {text}")
            # Write to a file as well
            with open("transcription_output.txt", "a", encoding="utf-8") as f:
                f.write(f"[{label} {response.start_time:.2f}-{response.end_time:.2f}s] {text}\n")


def main() -> int:
    args = _parse_args()
    if args.list_devices:
        return _list_input_devices()

    stop_event = threading.Event()

    def _handle_signal(signum, frame) -> None:
        del frame
        print(f"\nStopping (signal {signum})...", file=sys.stderr)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("Loading Parakeet model and starting transcription service...", file=sys.stderr)
    server, address = _start_server(args.host, args.port)
    print(
        f"Listening on microphone and streaming to {address}. Press Ctrl-C to stop.",
        file=sys.stderr,
    )

    try:
        _run_client(
            address=address,
            stop_event=stop_event,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            input_device_index=args.input_device_index,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        server.stop(grace=2).wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
