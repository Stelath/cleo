"""Tests for data.service.DataServiceServicer and data.sql.db.CleoSQLite."""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generated import data_pb2
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


def test_query_transcriptions_in_range(sqlite_db):
    sqlite_db.insert_transcription(text="before", start_time=10.0, end_time=11.0)
    sqlite_db.insert_transcription(text="inside", start_time=20.0, end_time=21.0)
    sqlite_db.insert_transcription(text="after", start_time=30.0, end_time=31.0)

    rows = sqlite_db.query_transcriptions_in_range(19.5, 21.5)
    assert [row["text"] for row in rows] == ["inside"]


def test_query_video_clips_in_range(sqlite_db):
    sqlite_db.insert_video_clip(clip_path="/tmp/a.mp4", start_timestamp=10.0, end_timestamp=11.0)
    sqlite_db.insert_video_clip(clip_path="/tmp/b.mp4", start_timestamp=20.0, end_timestamp=21.0)

    rows = sqlite_db.query_video_clips_in_range(19.5, 21.5)
    assert len(rows) == 1
    assert rows[0]["clip_path"] == "/tmp/b.mp4"


def test_insert_and_query_note_summary(sqlite_db):
    row_id = sqlite_db.insert_note_summary(
        summary_text="summary",
        start_timestamp=100.0,
        end_timestamp=110.0,
    )
    assert row_id >= 1

    rows, total = sqlite_db.query_note_summaries(limit=10)
    assert total == 1
    assert rows[0]["summary_text"] == "summary"


def test_insert_and_query_food_macros(sqlite_db):
    row_id = sqlite_db.insert_food_macros(
        product_name="Protein Bar",
        brand="Cleo",
        barcode="1234567890123",
        basis="per serving",
        calories_kcal=220.0,
        protein_g=20.0,
        fat_g=7.0,
        carbs_g=18.0,
        serving_size="1 bar",
        serving_quantity=50.0,
        recorded_at=123.0,
    )
    assert row_id >= 1

    rows, total = sqlite_db.query_food_macros(limit=10)
    assert total == 1
    assert rows[0]["product_name"] == "Protein Bar"
    assert rows[0]["protein_g"] == pytest.approx(20.0)


def test_insert_and_get_tracked_item(sqlite_db):
    item_id, created = sqlite_db.insert_tracked_item(
        title="phone",
        normalized_title="phone",
        embedding_json="[0.1, 0.2]",
        reference_image_path="/tmp/phone.jpg",
        registered_at=100.0,
    )
    assert created is True
    assert item_id >= 1

    row = sqlite_db.get_tracked_item_by_normalized_title("phone")
    assert row is not None
    assert row["id"] == item_id
    assert row["title"] == "phone"


def test_insert_tracked_item_returns_existing(sqlite_db):
    first_id, created = sqlite_db.insert_tracked_item(
        title="phone",
        normalized_title="phone",
        embedding_json="[0.1, 0.2]",
        reference_image_path="/tmp/phone.jpg",
        registered_at=100.0,
    )
    assert created is True

    second_id, created = sqlite_db.insert_tracked_item(
        title="phone",
        normalized_title="phone",
        embedding_json="[0.2, 0.3]",
        reference_image_path="/tmp/phone2.jpg",
        registered_at=101.0,
    )
    assert created is False
    assert second_id == first_id


# ── DataService gRPC servicer tests ──


@pytest.fixture
def data_servicer(tmp_path):
    """Create a DataServiceServicer with temp paths and mocked embeddings."""
    with patch("services.data.service.embed_video") as mock_embed_video, \
         patch("services.data.service.embed_text") as mock_embed_text, \
         patch("services.data.service.embed_image") as mock_embed_image, \
         patch("services.data.service.embed_face_image") as mock_embed_face_image:

        # Generic embeddings are 1024-d; face embeddings are 512-d.
        def _fake_embed(*args, **kwargs):
            v = np.random.randn(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        def _fake_face_embed(*args, **kwargs):
            v = np.random.randn(512).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        mock_embed_video.side_effect = _fake_embed
        mock_embed_text.side_effect = _fake_embed
        mock_embed_image.side_effect = _fake_embed
        mock_embed_face_image.side_effect = _fake_face_embed

        from services.data.service import DataServiceServicer

        servicer = DataServiceServicer(
            db_path=str(tmp_path / "test.db"),
            index_path=str(tmp_path / "test.index"),
            video_dir=str(tmp_path / "videos"),
            faces_index_path=str(tmp_path / "faces.index"),
            face_dir=str(tmp_path / "faces"),
            tracked_item_dir=str(tmp_path / "tracked_items"),
        )
        yield servicer
        servicer.shutdown()


def _mock_context():
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx


def _iter_media_chunks(data: bytes, media_id: str, chunk_size: int = 64):
    if not data:
        yield data_pb2.MediaChunk(data=b"", media_id=media_id, chunk_index=0, is_last=True)
        return

    chunk_index = 0
    total = len(data)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield data_pb2.MediaChunk(
            data=data[start:end],
            media_id=media_id,
            chunk_index=chunk_index,
            is_last=end >= total,
        )
        chunk_index += 1


def _store_clip_request_iter(
    *,
    upload_id: str,
    mp4_data: bytes,
    start_timestamp: float,
    end_timestamp: float,
    num_frames: int,
    embed_data: bytes = b"",
):
    yield data_pb2.StoreVideoClipChunk(
        metadata=data_pb2.StoreVideoClipMetadata(
            upload_id=upload_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            num_frames=num_frames,
        )
    )

    for chunk in _iter_media_chunks(mp4_data, media_id=f"{upload_id}:mp4"):
        yield data_pb2.StoreVideoClipChunk(mp4_chunk=chunk)

    if embed_data:
        for chunk in _iter_media_chunks(embed_data, media_id=f"{upload_id}:embed"):
            yield data_pb2.StoreVideoClipChunk(embed_chunk=chunk)


def test_store_transcription_rpc(data_servicer):
    from generated import data_pb2

    req = data_pb2.StoreTranscriptionRequest(
        text="test transcription", confidence=0.9, start_time=1.0, end_time=2.0
    )
    resp = data_servicer.StoreTranscription(req, _mock_context())
    assert resp.id >= 1


def test_store_video_clip_rpc(data_servicer, tmp_path):
    fake_mp4 = b"\x00" * 100
    req_iter = _store_clip_request_iter(
        upload_id="store-video-1",
        mp4_data=fake_mp4,
        start_timestamp=100.0,
        end_timestamp=110.0,
        num_frames=20,
    )
    resp = data_servicer.StoreVideoClip(req_iter, _mock_context())
    assert resp.clip_id >= 1
    assert resp.faiss_id >= 0

    # Verify file written to disk
    video_dir = tmp_path / "videos"
    clips = list(video_dir.glob("*.mp4"))
    assert len(clips) == 1


def test_search_rpc(data_servicer):
    # Store a clip first
    store_req_iter = _store_clip_request_iter(
        upload_id="store-video-2",
        mp4_data=b"\x00" * 100,
        start_timestamp=100.0,
        end_timestamp=110.0,
        num_frames=20,
    )
    data_servicer.StoreVideoClip(store_req_iter, _mock_context())

    # Search by text
    search_req = data_pb2.SearchRequest(text="test query", top_k=5)
    resp = data_servicer.Search(search_req, _mock_context())
    assert len(resp.results) >= 1
    assert resp.results[0].clip_id >= 1


def test_search_video_clips_by_text_rpc(data_servicer):
    store_req_iter = _store_clip_request_iter(
        upload_id="store-video-text-search",
        mp4_data=b"\x01" * 100,
        start_timestamp=120.0,
        end_timestamp=126.0,
        num_frames=12,
    )
    data_servicer.StoreVideoClip(store_req_iter, _mock_context())

    resp = data_servicer.SearchVideoClipsByText(
        data_pb2.SearchVideoClipsByTextRequest(query_text="find the clip", top_k=3),
        _mock_context(),
    )
    assert len(resp.results) >= 1
    assert resp.results[0].clip_id >= 1


def test_search_video_clips_by_text_rpc_filters_by_time_range(data_servicer):
    data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="store-video-filter-a",
            mp4_data=b"\x02" * 100,
            start_timestamp=100.0,
            end_timestamp=105.0,
            num_frames=10,
        ),
        _mock_context(),
    )
    data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="store-video-filter-b",
            mp4_data=b"\x03" * 100,
            start_timestamp=200.0,
            end_timestamp=205.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    resp = data_servicer.SearchVideoClipsByText(
        data_pb2.SearchVideoClipsByTextRequest(
            query_text="only the later clip",
            top_k=5,
            start_timestamp=190.0,
            end_timestamp=210.0,
        ),
        _mock_context(),
    )
    assert len(resp.results) >= 1
    assert all(result.start_timestamp >= 190.0 for result in resp.results)
    assert all(result.end_timestamp <= 210.0 for result in resp.results)


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
    # Store a clip
    fake_mp4 = b"fake_video_data_here"
    store_req_iter = _store_clip_request_iter(
        upload_id="store-video-3",
        mp4_data=fake_mp4,
        start_timestamp=1.0,
        end_timestamp=11.0,
        num_frames=20,
    )
    store_resp = data_servicer.StoreVideoClip(store_req_iter, _mock_context())

    # Retrieve it
    get_req = data_pb2.GetVideoClipRequest(clip_id=store_resp.clip_id)
    stream = data_servicer.GetVideoClip(get_req, _mock_context())
    chunks = list(stream)
    assert chunks
    assert b"".join(chunk.data for chunk in chunks) == fake_mp4
    assert chunks[-1].num_frames == 20


def test_store_food_macros_rpc(data_servicer):
    from generated import data_pb2

    req = data_pb2.StoreFoodMacrosRequest(
        product_name="Greek Yogurt",
        brand="Cleo",
        barcode="111222333444",
        basis="per serving",
        calories_kcal=120.0,
        protein_g=15.0,
        fat_g=0.0,
        carbs_g=8.0,
        serving_size="170 g",
        serving_quantity=170.0,
        recorded_at=100.0,
    )
    resp = data_servicer.StoreFoodMacros(req, _mock_context())
    assert resp.id >= 1

    rows, total = data_servicer._sqlite.query_food_macros(limit=10)
    assert total == 1
    assert rows[0]["barcode"] == "111222333444"


def test_get_food_macros_rpc(data_servicer):
    from generated import data_pb2

    data_servicer.StoreFoodMacros(
        data_pb2.StoreFoodMacrosRequest(
            product_name="Sparkling Water",
            brand="Cleo",
            basis="per can",
            calories_kcal=0.0,
            protein_g=0.0,
            fat_g=0.0,
            carbs_g=0.0,
            recorded_at=200.0,
        ),
        _mock_context(),
    )

    resp = data_servicer.GetFoodMacros(
        data_pb2.GetFoodMacrosRequest(limit=10),
        _mock_context(),
    )
    assert resp.total_count == 1
    assert resp.entries[0].product_name == "Sparkling Water"
    assert resp.entries[0].calories_kcal == pytest.approx(0.0)
    assert resp.entries[0].HasField("calories_kcal")


def test_get_transcriptions_in_range_rpc(data_servicer):
    from generated import data_pb2

    data_servicer.StoreTranscription(
        data_pb2.StoreTranscriptionRequest(
            text="captured",
            start_time=20.0,
            end_time=21.0,
        ),
        _mock_context(),
    )

    resp = data_servicer.GetTranscriptionsInRange(
        data_pb2.TimeRangeRequest(start_timestamp=19.0, end_timestamp=22.0),
        _mock_context(),
    )
    assert [entry.text for entry in resp.entries] == ["captured"]


def test_get_video_clips_in_range_rpc(data_servicer):
    data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="store-video-4",
            mp4_data=b"\x00" * 100,
            start_timestamp=20.0,
            end_timestamp=21.0,
            num_frames=3,
        ),
        _mock_context(),
    )

    resp = data_servicer.GetVideoClipsInRange(
        data_pb2.TimeRangeRequest(start_timestamp=19.0, end_timestamp=22.0),
        _mock_context(),
    )
    assert len(resp.clips) == 1
    assert resp.clips[0].num_frames == 3


def test_store_and_get_note_summaries_rpc(data_servicer):
    from generated import data_pb2

    store_resp = data_servicer.StoreNoteSummary(
        data_pb2.StoreNoteSummaryRequest(
            summary_text="meeting summary",
            start_timestamp=100.0,
            end_timestamp=110.0,
        ),
        _mock_context(),
    )
    assert store_resp.id >= 1

    resp = data_servicer.GetNoteSummaries(
        data_pb2.NoteSummariesRequest(limit=10),
        _mock_context(),
    )
    assert resp.total_count == 1
    assert resp.entries[0].summary_text == "meeting summary"


def test_get_nonexistent_clip_rpc(data_servicer):
    from generated import data_pb2

    ctx = _mock_context()
    req = data_pb2.GetVideoClipRequest(clip_id=9999)
    list(data_servicer.GetVideoClip(req, ctx))
    ctx.set_code.assert_called()


def test_store_tracked_item_rpc(data_servicer):
    image_data = b"\xff\xd8\xff\xe0" + b"\x00" * 64

    first = data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=image_data,
            registered_at=1000.0,
        ),
        _mock_context(),
    )
    assert first.created is True
    assert first.item_id >= 1
    assert first.normalized_title == "phone"

    second = data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="Phone",
            image_data=image_data,
            registered_at=1001.0,
        ),
        _mock_context(),
    )
    assert second.created is False
    assert second.item_id == first.item_id


def test_find_latest_tracked_item_occurrence_rpc(data_servicer):
    now = time.time()
    clip_a = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-a",
            mp4_data=b"\x01" * 100,
            start_timestamp=now - 3600.0,
            end_timestamp=now - 3590.0,
            num_frames=10,
        ),
        _mock_context(),
    )
    clip_b = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-b",
            mp4_data=b"\x02" * 100,
            start_timestamp=now - 1200.0,
            end_timestamp=now - 1190.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x03" * 64,
            registered_at=now - 100.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_a.faiss_id, 0.91, {}),
            (clip_b.faiss_id, 0.85, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="phone", top_k=5),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is True
    assert resp.latest_result.clip_id == clip_b.clip_id


def test_find_latest_tracked_item_occurrence_ranks_top_k_by_recency(data_servicer):
    now = time.time()
    clip_old_high = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-old-high-score",
            mp4_data=b"\x0a" * 100,
            start_timestamp=now - 4000.0,
            end_timestamp=now - 3985.0,
            num_frames=10,
        ),
        _mock_context(),
    )
    clip_recent_mid = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-recent-mid-score",
            mp4_data=b"\x0b" * 100,
            start_timestamp=now - 600.0,
            end_timestamp=now - 585.0,
            num_frames=10,
        ),
        _mock_context(),
    )
    clip_newest_low = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-newest-low-score",
            mp4_data=b"\x0c" * 100,
            start_timestamp=now - 200.0,
            end_timestamp=now - 185.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x03" * 64,
            registered_at=now - 120.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_old_high.faiss_id, 0.95, {}),
            (clip_recent_mid.faiss_id, 0.76, {}),
            (clip_newest_low.faiss_id, 0.30, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="phone", top_k=2),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is True
    assert resp.latest_result.clip_id == clip_recent_mid.clip_id


def test_find_latest_tracked_item_occurrence_skips_registration_clip(data_servicer):
    now = time.time()
    clip_outside = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-outside-window",
            mp4_data=b"\x11" * 100,
            start_timestamp=now - 1800.0,
            end_timestamp=now - 1790.0,
            num_frames=10,
        ),
        _mock_context(),
    )
    clip_registration = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-near-registration",
            mp4_data=b"\x22" * 100,
            start_timestamp=now - 130.0,
            end_timestamp=now - 115.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x04" * 64,
            registered_at=now - 120.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_registration.faiss_id, 0.97, {}),
            (clip_outside.faiss_id, 0.89, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="phone", top_k=5),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is True
    assert resp.latest_result.clip_id == clip_outside.clip_id


def test_find_latest_tracked_item_occurrence_falls_back_to_registration_clip(data_servicer):
    now = time.time()
    clip_registration = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-registration-only",
            mp4_data=b"\x33" * 100,
            start_timestamp=now - 130.0,
            end_timestamp=now - 115.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x05" * 64,
            registered_at=now - 120.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_registration.faiss_id, 0.94, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="phone", top_k=5),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is True
    assert resp.latest_result.clip_id == clip_registration.clip_id


def test_find_latest_tracked_item_occurrence_relaxes_default_threshold(data_servicer):
    now = time.time()
    clip_match = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-relaxed-threshold",
            mp4_data=b"\x44" * 100,
            start_timestamp=now - 3000.0,
            end_timestamp=now - 2985.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x06" * 64,
            registered_at=now - 120.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_match.faiss_id, 0.42, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="phone", top_k=5),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is True
    assert resp.latest_result.clip_id == clip_match.clip_id
    assert resp.min_score_used == pytest.approx(0.35)


def test_find_latest_tracked_item_occurrence_respects_explicit_min_score(data_servicer):
    now = time.time()
    clip_match = data_servicer.StoreVideoClip(
        _store_clip_request_iter(
            upload_id="tracked-item-explicit-threshold",
            mp4_data=b"\x55" * 100,
            start_timestamp=now - 3000.0,
            end_timestamp=now - 2985.0,
            num_frames=10,
        ),
        _mock_context(),
    )

    data_servicer.StoreTrackedItem(
        data_pb2.StoreTrackedItemRequest(
            title="phone",
            image_data=b"\xff\xd8\xff\xe0" + b"\x07" * 64,
            registered_at=now - 120.0,
        ),
        _mock_context(),
    )

    data_servicer._faiss.search = MagicMock(
        return_value=[
            (clip_match.faiss_id, 0.42, {}),
        ]
    )

    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(
            title="phone",
            top_k=5,
            min_score=0.5,
        ),
        _mock_context(),
    )
    assert resp.found_item is True
    assert resp.found_occurrence is False
    assert resp.min_score_used == pytest.approx(0.5)


def test_find_latest_tracked_item_occurrence_missing_item(data_servicer):
    resp = data_servicer.FindLatestTrackedItemOccurrence(
        data_pb2.FindLatestTrackedItemOccurrenceRequest(title="wallet"),
        _mock_context(),
    )
    assert resp.found_item is False
    assert resp.found_occurrence is False


# ── Face SQLite tests ──


def test_insert_and_get_face(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=99.5,
        first_seen=1000.0,
    )
    assert face_id >= 1

    # Set faiss_id so we can look it up
    sqlite_db.update_face_faiss_id(face_id, faiss_id=0)
    face = sqlite_db.get_face_by_faiss_id(0)
    assert face is not None
    assert face["thumbnail_path"] == "/tmp/face.jpg"
    assert face["confidence"] == pytest.approx(99.5)
    assert face["first_seen"] == pytest.approx(1000.0)
    assert face["seen_count"] == 1


def test_update_face_seen(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=99.0,
        first_seen=1000.0,
    )
    sqlite_db.update_face_faiss_id(face_id, faiss_id=0)

    sqlite_db.update_face_seen(face_id, last_seen=2000.0)

    face = sqlite_db.get_face_by_faiss_id(0)
    assert face["seen_count"] == 2
    assert face["last_seen"] == pytest.approx(2000.0)
    assert face["first_seen"] == pytest.approx(1000.0)


def test_get_face_by_faiss_id(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        first_seen=1000.0,
    )
    sqlite_db.update_face_faiss_id(face_id, faiss_id=42)

    face = sqlite_db.get_face_by_faiss_id(42)
    assert face is not None
    assert face["id"] == face_id

    # Non-existent
    assert sqlite_db.get_face_by_faiss_id(9999) is None


def test_set_face_name_and_list_faces(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=98.0,
        first_seen=1000.0,
    )

    assert sqlite_db.set_face_metadata(face_id, "Ada", "Met at the office") is True

    face = sqlite_db.get_face(face_id)
    assert face is not None
    assert face["display_name"] == "Ada"
    assert face["display_note"] == "Met at the office"

    rows, total = sqlite_db.list_faces(limit=10, offset=0)
    assert total == 1
    assert rows[0]["id"] == face_id
    assert rows[0]["display_name"] == "Ada"
    assert rows[0]["display_note"] == "Met at the office"

    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_1.jpg", seen_at=1000.0)
    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_2.jpg", seen_at=2000.0)

    sightings = sqlite_db.list_face_sightings(face_id=face_id, limit=4)
    assert [row["image_path"] for row in sightings] == ["/tmp/face_2.jpg", "/tmp/face_1.jpg"]
    assert sqlite_db.get_face_sighting_by_index(face_id, 1)["image_path"] == "/tmp/face_1.jpg"


def test_existing_faces_are_backfilled_into_face_sightings(tmp_path):
    db_path = tmp_path / "test.db"
    db = CleoSQLite(db_path=str(db_path))
    face_id = db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=98.0,
        first_seen=1000.0,
    )
    db.close()

    migrated = CleoSQLite(db_path=str(db_path))
    sightings = migrated.list_face_sightings(face_id=face_id, limit=4)
    assert len(sightings) == 1
    assert sightings[0]["image_path"] == "/tmp/face.jpg"
    migrated.close()


def test_clear_faces_removes_rows_and_returns_image_paths(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=98.0,
        first_seen=1000.0,
    )
    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_1.jpg", seen_at=1000.0)
    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_2.jpg", seen_at=2000.0)

    faces_deleted, sightings_deleted, image_paths = sqlite_db.clear_faces()

    assert faces_deleted == 1
    assert sightings_deleted == 2
    assert image_paths == ["/tmp/face.jpg", "/tmp/face_1.jpg", "/tmp/face_2.jpg"]
    rows, total = sqlite_db.list_faces(limit=10, offset=0)
    assert total == 0
    assert rows == []


def test_delete_face_removes_one_face_and_returns_image_paths(sqlite_db):
    face_id = sqlite_db.insert_face(
        thumbnail_path="/tmp/face.jpg",
        confidence=98.0,
        first_seen=1000.0,
    )
    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_1.jpg", seen_at=1000.0)
    sqlite_db.insert_face_sighting(face_id=face_id, image_path="/tmp/face_2.jpg", seen_at=2000.0)

    deleted, sightings_deleted, image_paths = sqlite_db.delete_face(face_id)

    assert deleted is True
    assert sightings_deleted == 2
    assert image_paths == ["/tmp/face.jpg", "/tmp/face_1.jpg", "/tmp/face_2.jpg"]
    assert sqlite_db.get_face(face_id) is None
    assert sqlite_db.list_face_sightings(face_id, limit=4) == []


# ── Face RPC tests ──


def test_store_face_embedding_new_face_rpc(data_servicer, tmp_path):
    from generated import data_pb2

    fake_face_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    req = data_pb2.StoreFaceEmbeddingRequest(
        image_data=fake_face_jpeg,
        timestamp=1000.0,
        confidence=99.5,
    )
    resp = data_servicer.StoreFaceEmbedding(req, _mock_context())
    assert resp.is_new is True
    assert resp.face_id >= 1
    assert resp.faiss_id >= 0


def test_store_face_embedding_dedup_rpc(data_servicer, tmp_path):
    from generated import data_pb2

    # Use a fixed embedding vector so both calls produce the same vector
    fixed_vec = np.random.randn(512).astype(np.float32)
    fixed_vec /= np.linalg.norm(fixed_vec)

    with patch("services.data.service.embed_face_image", return_value=fixed_vec):
        fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        # First store — new face
        resp1 = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face, timestamp=1000.0, confidence=99.0
            ),
            _mock_context(),
        )
        assert resp1.is_new is True

        # Second store — same vector, should deduplicate
        resp2 = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face, timestamp=2000.0, confidence=99.0
            ),
            _mock_context(),
        )
        assert resp2.is_new is False
        assert resp2.matched_face_id == resp1.face_id

        face = data_servicer._sqlite.get_face(resp1.face_id)
        sightings = data_servicer._sqlite.list_face_sightings(resp1.face_id, limit=4)
        assert face is not None
        assert face["seen_count"] == 2
        assert len(sightings) == 2
        assert data_servicer._faces_faiss.size == 2
        assert face["faiss_id"] == resp2.faiss_id


def test_search_faces_rpc(data_servicer, tmp_path):
    from generated import data_pb2

    fixed_vec = np.random.randn(512).astype(np.float32)
    fixed_vec /= np.linalg.norm(fixed_vec)

    with patch("services.data.service.embed_face_image", return_value=fixed_vec):
        fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        # Store a face
        data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face, timestamp=1000.0, confidence=99.0
            ),
            _mock_context(),
        )

        # Search for it
        resp = data_servicer.SearchFaces(
            data_pb2.SearchFacesRequest(image_data=fake_face, top_k=5),
            _mock_context(),
        )
        assert len(resp.results) >= 1
        assert resp.results[0].face_id >= 1
        assert resp.results[0].seen_count >= 1


def test_search_faces_groups_multiple_exemplars_into_one_result(data_servicer):
    from generated import data_pb2

    fixed_vec = np.random.randn(512).astype(np.float32)
    fixed_vec /= np.linalg.norm(fixed_vec)

    with patch("services.data.service.embed_face_image", return_value=fixed_vec):
        fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        first = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face, timestamp=1000.0, confidence=99.0
            ),
            _mock_context(),
        )
        second = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face, timestamp=1010.0, confidence=99.0
            ),
            _mock_context(),
        )

        assert second.face_id == first.face_id

        resp = data_servicer.SearchFaces(
            data_pb2.SearchFacesRequest(image_data=fake_face, top_k=5),
            _mock_context(),
        )

        assert len(resp.results) == 1
        assert resp.results[0].face_id == first.face_id
        assert resp.results[0].score >= 1.0


def test_delete_face_rpc_removes_one_identity_and_preserves_others(data_servicer):
    from generated import data_pb2

    vec_a = np.random.randn(512).astype(np.float32)
    vec_a /= np.linalg.norm(vec_a)
    vec_b = np.random.randn(512).astype(np.float32)
    vec_b /= np.linalg.norm(vec_b)
    fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

    with patch(
        "services.data.service.embed_face_image",
        side_effect=[vec_a, vec_b],
    ):
        first = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face,
                timestamp=1000.0,
                confidence=99.0,
            ),
            _mock_context(),
        )
        second = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=fake_face,
                timestamp=1010.0,
                confidence=99.0,
            ),
            _mock_context(),
        )

    response = data_servicer.DeleteFace(
        data_pb2.DeleteFaceRequest(face_id=first.face_id),
        _mock_context(),
    )

    assert response.deleted is True
    assert data_servicer._sqlite.get_face(first.face_id) is None
    remaining = data_servicer._sqlite.get_face(second.face_id)
    assert remaining is not None

    with patch("services.data.service.embed_face_image", return_value=vec_b):
        search = data_servicer.SearchFaces(
            data_pb2.SearchFacesRequest(image_data=fake_face, top_k=5),
            _mock_context(),
        )
    assert len(search.results) == 1
    assert search.results[0].face_id == second.face_id


def test_store_face_embedding_concurrent_identical_requests_share_identity(data_servicer):
    from generated import data_pb2

    fixed_vec = np.random.randn(512).astype(np.float32)
    fixed_vec /= np.linalg.norm(fixed_vec)
    fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

    original_select = data_servicer._select_face_match

    def delayed_select(embedding):
        result = original_select(embedding)
        time.sleep(0.05)
        return result

    responses: list[data_pb2.StoreFaceEmbeddingResponse] = []
    errors: list[Exception] = []
    start_barrier = threading.Barrier(2)

    def worker(ts: float) -> None:
        try:
            start_barrier.wait(timeout=1.0)
            resp = data_servicer.StoreFaceEmbedding(
                data_pb2.StoreFaceEmbeddingRequest(
                    image_data=fake_face,
                    timestamp=ts,
                    confidence=99.0,
                ),
                _mock_context(),
            )
            responses.append(resp)
        except Exception as exc:  # pragma: no cover - test failure path
            errors.append(exc)

    with patch("services.data.service.embed_face_image", return_value=fixed_vec), patch.object(
        data_servicer, "_select_face_match", side_effect=delayed_select
    ):
        threads = [
            threading.Thread(target=worker, args=(1000.0,)),
            threading.Thread(target=worker, args=(1001.0,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

    assert not errors
    assert len(responses) == 2
    assert {resp.face_id for resp in responses} == {responses[0].face_id}
    rows, total = data_servicer._sqlite.list_faces(limit=10, offset=0)
    assert total == 1
    assert rows[0]["seen_count"] == 2


def test_list_faces_rpc(data_servicer):
    from generated import data_pb2

    fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    store_resp = data_servicer.StoreFaceEmbedding(
        data_pb2.StoreFaceEmbeddingRequest(
            image_data=fake_face,
            timestamp=1000.0,
            confidence=99.0,
        ),
        _mock_context(),
    )

    name_resp = data_servicer.SetFaceName(
        data_pb2.SetFaceNameRequest(
            face_id=store_resp.face_id,
            name="Grace",
            note="Product manager from Brooklyn",
        ),
        _mock_context(),
    )
    assert name_resp.updated is True
    assert name_resp.name == "Grace"
    assert name_resp.note == "Product manager from Brooklyn"

    resp = data_servicer.ListFaces(
        data_pb2.ListFacesRequest(limit=10, offset=0),
        _mock_context(),
    )
    assert resp.total_count == 1
    assert resp.entries[0].face_id == store_resp.face_id
    assert resp.entries[0].name == "Grace"
    assert resp.entries[0].note == "Product manager from Brooklyn"
    assert resp.entries[0].seen_count >= 1
    assert resp.entries[0].collage_image_count == 1


def test_get_face_image_rpc(data_servicer):
    from generated import data_pb2

    fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    store_resp = data_servicer.StoreFaceEmbedding(
        data_pb2.StoreFaceEmbeddingRequest(
            image_data=fake_face,
            timestamp=1000.0,
            confidence=99.0,
        ),
        _mock_context(),
    )

    resp = data_servicer.GetFaceImage(
        data_pb2.GetFaceImageRequest(face_id=store_resp.face_id),
        _mock_context(),
    )
    assert resp.image_data == fake_face
    assert resp.content_type == "image/jpeg"


def test_get_face_sighting_image_rpc(data_servicer):
    from generated import data_pb2

    with patch(
        "services.data.service.embed_face_image",
        return_value=np.ones(512, dtype=np.float32) / np.sqrt(512),
    ):
        first_face = b"\xff\xd8\xff\xe0" + b"\x01" * 100
        second_face = b"\xff\xd8\xff\xe0" + b"\x02" * 100

        store_resp = data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=first_face,
                timestamp=1000.0,
                confidence=99.0,
            ),
            _mock_context(),
        )
        data_servicer.StoreFaceEmbedding(
            data_pb2.StoreFaceEmbeddingRequest(
                image_data=second_face,
                timestamp=2000.0,
                confidence=99.0,
            ),
            _mock_context(),
        )

    latest = data_servicer.GetFaceImage(
        data_pb2.GetFaceImageRequest(face_id=store_resp.face_id, sighting_index=0),
        _mock_context(),
    )
    older = data_servicer.GetFaceImage(
        data_pb2.GetFaceImageRequest(face_id=store_resp.face_id, sighting_index=1),
        _mock_context(),
    )

    assert latest.image_data == second_face
    assert older.image_data == first_face


def test_clear_faces_rpc(data_servicer, tmp_path):
    from generated import data_pb2

    first_image = tmp_path / "faces" / "face_a.jpg"
    second_image = tmp_path / "faces" / "face_b.jpg"
    fake_face = b"\xff\xd8\xff\xe0" + b"\x00" * 100

    store_resp = data_servicer.StoreFaceEmbedding(
        data_pb2.StoreFaceEmbeddingRequest(
            image_data=fake_face,
            timestamp=1000.0,
            confidence=99.0,
        ),
        _mock_context(),
    )
    data_servicer._store_face_sighting_image(
        face_id=store_resp.face_id,
        image_data=fake_face,
        timestamp=2000.0,
    )
    first_image.write_bytes(fake_face)
    second_image.write_bytes(fake_face)
    data_servicer._sqlite.insert_face_sighting(
        face_id=store_resp.face_id,
        image_path=str(first_image),
        seen_at=3000.0,
    )
    data_servicer._sqlite.insert_face_sighting(
        face_id=store_resp.face_id,
        image_path=str(second_image),
        seen_at=4000.0,
    )

    resp = data_servicer.ClearFaces(data_pb2.ClearFacesRequest(), _mock_context())

    assert resp.faces_deleted == 1
    assert resp.sightings_deleted == 4
    assert data_servicer._faces_faiss.size == 0
    assert data_servicer._sqlite.list_faces(limit=10, offset=0) == ([], 0)
    assert not first_image.exists()
    assert not second_image.exists()
