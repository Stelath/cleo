"""Tests for transcription.parakeet.TranscriptionServiceServicer."""


def test_transcribe_unary():
    """Mock the NeMo model, call Transcribe(), and verify the result text is correct."""
    pass


def test_transcribe_stream_accumulation():
    """Send audio chunks below the accumulation threshold and verify no results are
    yielded until the threshold is reached."""
    pass


def test_transcribe_stream_is_final_flushes():
    """Send a chunk with is_final=True and verify it triggers an immediate flush
    regardless of accumulated duration."""
    pass
