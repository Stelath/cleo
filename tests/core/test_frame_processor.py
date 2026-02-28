"""Tests for core.frame_processor."""


def test_embed_frame_returns_none():
    """The stubbed embed_frame() should return None (no embedding API wired)."""
    pass


def test_process_chunk_skips_when_embed_none(faiss_db):
    """When embed_frame returns None, FaissDB size stays at zero."""
    pass


def test_process_chunk_adds_to_faiss(faiss_db):
    """When embed_frame is patched to return a vector, _process_chunk stores it in FAISS."""
    pass
