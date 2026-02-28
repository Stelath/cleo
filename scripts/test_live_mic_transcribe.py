"""Live microphone transcription against the Parakeet gRPC service."""

from __future__ import annotations

import argparse
import math
import signal
import struct
import sys
import threading
from concurrent import futures
from pathlib import Path
from typing import Iterator

import grpc
import pyaudio

from generated import transcription_pb2, transcription_pb2_grpc
from services.transcription import service as transcription_service

_MAX_GRPC_MESSAGE_BYTES = 16 * 1024 * 1024
_SILENCE_THRESHOLD_RMS = 0.01
_FINAL_SILENCE_MS = 1000


def _chunk_rms(chunk_data: bytes) -> float:
    if not chunk_data:
        return 0.0

    sample_count = len(chunk_data) // 4
    if sample_count == 0:
        return 0.0

    total = 0.0
    for index in range(sample_count):
        sample = struct.unpack_from("<f", chunk_data, index * 4)[0]
        total += sample * sample
    return math.sqrt(total / sample_count)


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


class TriggerEventState:
    """Shared service-side wake-word detection state for the live mic harness."""

    def __init__(self):
        self._detected_utterances: set[str] = set()
        self._lock = threading.Lock()

    def mark_detected(self, result: transcription_pb2.TranscriptionResult) -> None:
        if not result.utterance_id:
            return
        with self._lock:
            self._detected_utterances.add(result.utterance_id)

    def was_detected(self, utterance_id: str) -> bool:
        if not utterance_id:
            return False
        with self._lock:
            return utterance_id in self._detected_utterances


def _start_server(host: str, port: int) -> tuple[grpc.Server, str, TriggerEventState]:
    trigger_state = TriggerEventState()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
        ],
    )
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        transcription_service.TranscriptionServiceServicer(
            on_trigger_detected=trigger_state.mark_detected,
        ),
        server,
    )
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind transcription service on {host}:{port}")
    server.start()
    return server, f"{host}:{bound_port}", trigger_state


class MicrophoneInput:
    """Keeps the microphone stream open while yielding one utterance per request stream."""

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_ms: int,
        input_device_index: int | None,
    ):
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._frames_per_buffer = max(1, int(sample_rate * chunk_ms / 1000))
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=self._frames_per_buffer,
        )
        self._stream_index = 0

    def close(self) -> None:
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._audio.terminate()

    def next_utterance_requests(
        self,
        stop_event: threading.Event,
    ) -> Iterator[transcription_pb2.AudioInput]:
        silence_ms = 0.0
        speech_active = False
        self._stream_index += 1
        stream_id = f"live-mic-{self._stream_index}"

        while not stop_event.is_set():
            chunk = self._stream.read(
                self._frames_per_buffer,
                exception_on_overflow=False,
            )
            rms = _chunk_rms(chunk)

            if rms >= _SILENCE_THRESHOLD_RMS:
                speech_active = True
                silence_ms = 0.0
            elif speech_active:
                silence_ms += self._chunk_ms
            else:
                continue

            is_final = speech_active and silence_ms >= _FINAL_SILENCE_MS
            yield transcription_pb2.AudioInput(
                audio_data=chunk,
                sample_rate=self._sample_rate,
                is_final=is_final,
                stream_id=stream_id,
            )

            if is_final:
                return

        if speech_active:
            yield transcription_pb2.AudioInput(
                audio_data=b"",
                sample_rate=self._sample_rate,
                is_final=True,
                stream_id=stream_id,
            )

def _run_client(
    address: str,
    stop_event: threading.Event,
    sample_rate: int,
    chunk_ms: int,
    input_device_index: int | None,
    trigger_state: TriggerEventState,
) -> None:
    mic = MicrophoneInput(
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        input_device_index=input_device_index,
    )

    try:
        with grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_BYTES),
            ],
        ) as channel:
            grpc.channel_ready_future(channel).result(timeout=30)
            stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)

            while not stop_event.is_set():
                responses = stub.TranscribeStream(
                    mic.next_utterance_requests(stop_event=stop_event)
                )

                saw_audio = False
                for response in responses:
                    text = response.text.strip()
                    if not text:
                        continue

                    saw_audio = True
                    label = "partial" if response.is_partial else "final"
                    wake_marker = (
                        " wake-word=DETECTED"
                        if trigger_state.was_detected(response.utterance_id)
                        else ""
                    )
                    line = (
                        f"[{label} {response.start_time:.2f}-{response.end_time:.2f}s "
                        f"utterance={response.utterance_id or '-'}{wake_marker}] {text}"
                    )
                    print(line)

                    output_path = Path("transcription_output.txt")
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(f"{line}\n")

                if not saw_audio and stop_event.is_set():
                    break
    finally:
        mic.close()


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

    print("Starting transcription service...", file=sys.stderr)
    server, address, trigger_state = _start_server(args.host, args.port)
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
            trigger_state=trigger_state,
        )
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        server.stop(grace=2).wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
