"""Unit tests for transcription service timestamp handling."""

import numpy as np

from generated import sensor_pb2
from generated import transcription_pb2
from services.transcription.service import AmazonTranscribeBackend, SensorTranscriptionPipeline


def test_stream_start_epoch_uses_chunk_timestamp():
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()
    request = transcription_pb2.AudioInput(
        audio_data=audio,
        sample_rate=4,
        timestamp=100.0,
    )

    stream_start = AmazonTranscribeBackend._stream_start_epoch(request)

    assert stream_start == 99.5


def test_stream_start_epoch_falls_back_to_current_time(monkeypatch):
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()
    request = transcription_pb2.AudioInput(
        audio_data=audio,
        sample_rate=4,
    )
    monkeypatch.setattr("services.transcription.service.time.time", lambda: 500.0)

    stream_start = AmazonTranscribeBackend._stream_start_epoch(request)

    assert stream_start == 499.5


def test_sensor_audio_requests_forward_chunk_timestamps():
    servicer = object()
    pipeline = SensorTranscriptionPipeline(servicer)
    audio = np.array([0.0, 0.5], dtype=np.float32).tobytes()

    class Stub:
        @staticmethod
        def StreamAudio(_request):
            yield sensor_pb2.AudioChunk(
                data=audio,
                sample_rate=4,
                num_samples=2,
                timestamp=123.4,
            )

    requests = list(pipeline._sensor_audio_requests(Stub()))

    assert requests[0].timestamp == 123.4
    assert requests[1].timestamp > 0.0
    assert requests[1].is_final is True
