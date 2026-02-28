"""Fixtures for transcription service unit tests."""

import pytest

from generated import transcription_pb2


@pytest.fixture
def mock_transcription_result():
    """Return a TranscriptionResult with sample data."""
    return transcription_pb2.TranscriptionResult(
        text="hello world",
        confidence=0.95,
        start_time=0.0,
        end_time=1.5,
        is_partial=False,
    )
