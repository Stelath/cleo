"""End-to-end tests for the Parakeet transcription gRPC service."""

from __future__ import annotations

import re
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import pytest

from generated import transcription_pb2, transcription_pb2_grpc

from transcription import parakeet


def _load_mp3_as_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    return audio, sample_rate


def _stream_requests(audio: np.ndarray, sample_rate: int):
    chunk_samples = max(1, sample_rate // 4)

    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        yield transcription_pb2.AudioInput(
            audio_data=audio[start:end].tobytes(),
            sample_rate=sample_rate,
            is_final=end == len(audio),
        )


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


@pytest.mark.integration
@pytest.mark.transcription_e2e
def test_transcribe_stream_end_to_end_with_mp3():
    """Start the real service, stream an MP3 in chunks, and verify the transcript."""
    print("Test called")
    audio_path = Path(__file__).with_name("hello_my_name_is_ethan.mp3")
    audio, sample_rate = _load_mp3_as_mono_float32(audio_path)

    print(f"Loaded audio from {audio_path} with shape {audio.shape} and sample rate {sample_rate}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        parakeet.TranscriptionServiceServicer(),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = transcription_pb2_grpc.TranscriptionServiceStub(channel)
            responses = list(stub.TranscribeStream(_stream_requests(audio, sample_rate), timeout=120))
    finally:
        server.stop(grace=2).wait()

    assert len(audio) > 0
    assert sample_rate > 0
    assert responses

    normalized_transcript = _normalize_text(" ".join(response.text for response in responses))
    assert "hello my name is ethan" in normalized_transcript
    assert responses[-1].is_partial is False
    assert responses[-1].end_time == pytest.approx(len(audio) / sample_rate, rel=1e-3)

    print("Normalized transcript:", normalized_transcript)
