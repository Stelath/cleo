"""Tests for data.service.DataServiceServicer and data.sql.db.CleoSQLite."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.data.sql.db import CleoSQLite


# ── SQLite tests ──


@pytest.fixture
def sqlite_db(tmp_path):
    db = CleoSQLite(db_path=str(tmp_path / "test.db"))
    yield db
    db.close()


def test_insert_and_query_transcription(sqlite_db):
    row_id = sqlite_db.insert_transcription(
        text="hello world", confidence=0.95, start_time=1.0, end_time=2.0
    )
    assert row_id >= 1

    rows, total = sqlite_db.query_transcriptions(limit=10)
    assert total == 1
    assert rows[0]["text"] == "hello world"
    assert rows[0]["confidence"] == pytest.approx(0.95)


def test_transcription_pagination(sqlite_db):
    for i in range(10):
        sqlite_db.insert_transcription(text=f"entry {i}")

    rows, total = sqlite_db.query_transcriptions(limit=3, offset=0)
    assert total == 10
    assert len(rows) == 3

    rows2, _ = sqlite_db.query_transcriptions(limit=3, offset=3)
    assert len(rows2) == 3
    # Pages should not overlap
    ids1 = {r["id"] for r in rows}
    ids2 = {r["id"] for r in rows2}
    assert ids1.isdisjoint(ids2)


def test_insert_and_get_video_clip(sqlite_db):
    clip_id = sqlite_db.insert_video_clip(
        clip_path="/tmp/clip.mp4",
        start_timestamp=100.0,
        end_timestamp=110.0,
        num_frames=20,
        embedding_dimension=1024,
    )
    assert clip_id >= 1

    meta = sqlite_db.get_clip_metadata(clip_id)
    assert meta is not None
    assert meta["clip_path"] == "/tmp/clip.mp4"
    assert meta["num_frames"] == 20


def test_update_clip_faiss_id(sqlite_db):
    clip_id = sqlite_db.insert_video_clip(clip_path="/tmp/clip.mp4")
    sqlite_db.update_clip_faiss_id(clip_id, faiss_id=42)

    meta = sqlite_db.get_clip_by_faiss_id(42)
    assert meta is not None
    assert meta["id"] == clip_id


def test_get_nonexistent_clip(sqlite_db):
    assert sqlite_db.get_clip_metadata(9999) is None
    assert sqlite_db.get_clip_by_faiss_id(9999) is None


# ── DataService gRPC servicer tests ──


@pytest.fixture
def data_servicer(tmp_path):
    """Create a DataServiceServicer with temp paths and mocked embeddings."""
    with patch("services.data.service.embed_video") as mock_embed_video, \
         patch("services.data.service.embed_text") as mock_embed_text, \
         patch("services.data.service.embed_image") as mock_embed_image:

        # All embedding functions return a random normalized 1024-d vector
        def _fake_embed(*args, **kwargs):
            v = np.random.randn(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        mock_embed_video.side_effect = _fake_embed
        mock_embed_text.side_effect = _fake_embed
        mock_embed_image.side_effect = _fake_embed

        from services.data.service import DataServiceServicer

        servicer = DataServiceServicer(
            db_path=str(tmp_path / "test.db"),
            index_path=str(tmp_path / "test.index"),
            video_dir=str(tmp_path / "videos"),
        )
        yield servicer
        servicer.shutdown()


def _mock_context():
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx


def test_store_transcription_rpc(data_servicer):
    from generated import data_pb2

    req = data_pb2.StoreTranscriptionRequest(
        text="test transcription", confidence=0.9, start_time=1.0, end_time=2.0
    )
    resp = data_servicer.StoreTranscription(req, _mock_context())
    assert resp.id >= 1


def test_store_video_clip_rpc(data_servicer, tmp_path):
    from generated import data_pb2

    fake_mp4 = b"\x00" * 100
    req = data_pb2.StoreVideoClipRequest(
        mp4_data=fake_mp4,
        start_timestamp=100.0,
        end_timestamp=110.0,
        num_frames=20,
    )
    resp = data_servicer.StoreVideoClip(req, _mock_context())
    assert resp.clip_id >= 1
    assert resp.faiss_id >= 0

    # Verify file written to disk
    video_dir = tmp_path / "videos"
    clips = list(video_dir.glob("*.mp4"))
    assert len(clips) == 1


def test_search_rpc(data_servicer):
    from generated import data_pb2

    # Store a clip first
    store_req = data_pb2.StoreVideoClipRequest(
        mp4_data=b"\x00" * 100,
        start_timestamp=100.0,
        end_timestamp=110.0,
        num_frames=20,
    )
    data_servicer.StoreVideoClip(store_req, _mock_context())

    # Search by text
    search_req = data_pb2.SearchRequest(text="test query", top_k=5)
    resp = data_servicer.Search(search_req, _mock_context())
    assert len(resp.results) >= 1
    assert resp.results[0].clip_id >= 1


def test_get_transcription_log_rpc(data_servicer):
    from generated import data_pb2

    # Store some transcriptions
    for i in range(5):
        req = data_pb2.StoreTranscriptionRequest(text=f"text {i}")
        data_servicer.StoreTranscription(req, _mock_context())

    log_req = data_pb2.TranscriptionLogRequest(limit=3)
    resp = data_servicer.GetTranscriptionLog(log_req, _mock_context())
    assert len(resp.entries) == 3
    assert resp.total_count == 5


def test_get_video_clip_rpc(data_servicer):
    from generated import data_pb2

    # Store a clip
    fake_mp4 = b"fake_video_data_here"
    store_req = data_pb2.StoreVideoClipRequest(
        mp4_data=fake_mp4, start_timestamp=1.0, end_timestamp=11.0, num_frames=20
    )
    store_resp = data_servicer.StoreVideoClip(store_req, _mock_context())

    # Retrieve it
    get_req = data_pb2.GetVideoClipRequest(clip_id=store_resp.clip_id)
    resp = data_servicer.GetVideoClip(get_req, _mock_context())
    assert resp.mp4_data == fake_mp4
    assert resp.num_frames == 20


def test_get_nonexistent_clip_rpc(data_servicer):
    from generated import data_pb2

    ctx = _mock_context()
    req = data_pb2.GetVideoClipRequest(clip_id=9999)
    data_servicer.GetVideoClip(req, ctx)
    ctx.set_code.assert_called()

def test_set_and_get_user_preference(data_servicer):
    from generated import data_pb2

    ctx = _mock_context()
    
    # Set a preference
    set_req = data_pb2.SetUserPreferenceRequest(key="color_blindness_type", value="protanopia")
    set_resp = data_servicer.SetUserPreference(set_req, ctx)
    assert set_resp.success is True

    # Get the preference back
    get_req = data_pb2.GetUserPreferenceRequest(key="color_blindness_type")
    get_resp = data_servicer.GetUserPreference(get_req, ctx)
    assert get_resp.found is True
    assert get_resp.value == "protanopia"

def test_get_missing_user_preference_returns_empty(data_servicer):
    from generated import data_pb2

    ctx = _mock_context()
    
    # Get a preference that hasn't been set
    get_req = data_pb2.GetUserPreferenceRequest(key="missing_key")
    get_resp = data_servicer.GetUserPreference(get_req, ctx)
    
    assert get_resp.found is False
    assert get_resp.value == ""
