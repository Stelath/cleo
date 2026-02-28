"""DataService gRPC servicer - owns SQLite, FAISS, Bedrock embeddings, and video storage."""

import json
import mimetypes
import re
import signal
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any

import grpc
import structlog

from services.config import (
    DATA_PORT,
    EMBEDDING_DIMENSION,
    FACE_CLUSTER_MIN_SIGHTINGS,
    FACE_EMBEDDING_DIMENSION,
    FACE_SIMILARITY_THRESHOLD,
    FACE_STORAGE_DIR,
    TRACKED_ITEM_STORAGE_DIR,
    VIDEO_STORAGE_DIR,
)
from services.data.vector.embedding import (
    embed_face_image,
    embed_image,
    embed_text,
    embed_video,
)
from services.data.sql.db import CleoSQLite
from services.data.vector.faiss_db import FaissDB
from generated import data_pb2, data_pb2_grpc

log = structlog.get_logger()

_MEDIA_CHUNK_BYTES = 256 * 1024
_FACE_MATCH_SEARCH_K = 8
_FACE_MATCH_SCORE_MARGIN = 0.03
_FACE_SEARCH_RESULT_MULTIPLIER = 3
_TRACKED_ITEM_SEARCH_TOP_K_DEFAULT = 128
_TRACKED_ITEM_MIN_SCORE_DEFAULT = 0.55
_TRACKED_ITEM_REGISTRATION_CLIP_MARGIN_SECONDS = 0.75
_TRACKED_ITEM_RELAXED_MIN_SCORES = (0.45, 0.35, 0.25, 0.15)
_TRACKED_ITEM_LOOKBACK_SECONDS = 24 * 60 * 60
_TRACKED_ITEM_SEARCH_OVERFETCH_MULTIPLIER = 12


def _normalize_item_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return " ".join(normalized.split())


class DataServiceServicer(data_pb2_grpc.DataServiceServicer):
    """Manages persistent storage: SQLite for metadata, FAISS for vectors, disk for video."""

    def __init__(
        self,
        db_path: str = "data/cleo.db",
        index_path: str = "data/vector/clips.index",
        video_dir: str = VIDEO_STORAGE_DIR,
        embedding_dim: int = EMBEDDING_DIMENSION,
        face_embedding_dim: int = FACE_EMBEDDING_DIMENSION,
        faces_index_path: str = "data/vector/faces.index",
        face_dir: str = FACE_STORAGE_DIR,
        tracked_item_dir: str = TRACKED_ITEM_STORAGE_DIR,
    ):
        self._sqlite = CleoSQLite(db_path=db_path)
        self._faiss = FaissDB(dimension=embedding_dim, index_path=index_path)
        self._faces_faiss = FaissDB(
            dimension=face_embedding_dim,
            index_path=faces_index_path,
        )
        self._video_dir = Path(video_dir)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._face_dir = Path(face_dir)
        self._face_dir.mkdir(parents=True, exist_ok=True)
        self._tracked_item_dir = Path(tracked_item_dir)
        self._tracked_item_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim
        self._face_embedding_dim = face_embedding_dim
        self._face_store_lock = threading.Lock()
        log.info(
            "data_service.init",
            db_path=db_path,
            index_path=index_path,
            video_dir=video_dir,
            tracked_item_dir=tracked_item_dir,
            faiss_size=self._faiss.size,
            faces_faiss_size=self._faces_faiss.size,
            face_embedding_dim=face_embedding_dim,
        )

    def shutdown(self):
        """Persist state and close connections."""
        self._faiss.save()
        self._faces_faiss.save()
        self._sqlite.close()
        log.info("data_service.shutdown")

    def _group_face_matches(
        self,
        embedding,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raw_k = max(top_k, 1)
        raw_results = self._faces_faiss.search(embedding, k=raw_k)
        grouped: dict[int, dict[str, Any]] = {}

        for faiss_id, score, meta in raw_results:
            face_id_value = meta.get("face_id") if isinstance(meta, dict) else None
            face_id = int(face_id_value) if face_id_value is not None else None
            if face_id is None:
                face = self._sqlite.get_face_by_faiss_id(faiss_id)
                if face is None:
                    continue
                face_id = int(face["id"])
            else:
                face = self._sqlite.get_face(face_id)
                if face is None:
                    continue

            entry = grouped.get(face_id)
            if entry is None:
                entry = {
                    "face": face,
                    "best_faiss_id": int(faiss_id),
                    "max_score": float(score),
                    "score_sum": float(score),
                    "vote_count": 1,
                }
                grouped[face_id] = entry
                continue

            entry["score_sum"] += float(score)
            entry["vote_count"] += 1
            if score > entry["max_score"]:
                entry["max_score"] = float(score)
                entry["best_faiss_id"] = int(faiss_id)

        ranked: list[dict[str, Any]] = []
        for face_id, entry in grouped.items():
            vote_count = int(entry["vote_count"])
            avg_score = float(entry["score_sum"]) / vote_count
            max_score = float(entry["max_score"])
            vote_bonus = min(0.02 * max(vote_count - 1, 0), 0.06)
            aggregate_score = (max_score * 0.7) + (avg_score * 0.3) + vote_bonus
            ranked.append(
                {
                    "face_id": face_id,
                    "face": entry["face"],
                    "best_faiss_id": int(entry["best_faiss_id"]),
                    "max_score": max_score,
                    "avg_score": avg_score,
                    "vote_count": vote_count,
                    "aggregate_score": aggregate_score,
                }
            )

        ranked.sort(
            key=lambda item: (
                float(item["aggregate_score"]),
                float(item["max_score"]),
                int(item["vote_count"]),
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _is_core_face(face: dict[str, Any]) -> bool:
        return int(face.get("seen_count", 0)) >= FACE_CLUSTER_MIN_SIGHTINGS

    def _select_ranked_face_match(self, ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not ranked:
            return None

        best = ranked[0]
        runner_up_score = float(ranked[1]["aggregate_score"]) if len(ranked) > 1 else None
        if (
            runner_up_score is not None
            and float(best["aggregate_score"]) - runner_up_score < _FACE_MATCH_SCORE_MARGIN
        ):
            log.info(
                "data_service.face_match_ambiguous",
                best_face_id=best["face_id"],
                best_score=best["aggregate_score"],
                runner_up_face_id=ranked[1]["face_id"],
                runner_up_score=runner_up_score,
            )
            return None

        return best

    def _select_face_match(self, embedding, *, core_only: bool = False) -> dict[str, Any] | None:
        ranked = self._group_face_matches(embedding, top_k=_FACE_MATCH_SEARCH_K)
        passing = [
            match
            for match in ranked
            if float(match["aggregate_score"]) >= FACE_SIMILARITY_THRESHOLD
        ]
        if not passing:
            return None

        core_matches = [match for match in passing if self._is_core_face(match["face"])]
        if core_only:
            return self._select_ranked_face_match(core_matches)

        if core_matches:
            return self._select_ranked_face_match(core_matches)

        return self._select_ranked_face_match(passing)

    # ── StoreTranscription ──

    def StoreTranscription(self, request, context):
        row_id = self._sqlite.insert_transcription(
            text=request.text,
            confidence=request.confidence if request.confidence else None,
            start_time=request.start_time if request.start_time else None,
            end_time=request.end_time if request.end_time else None,
        )
        log.info("data_service.transcription_stored", id=row_id, text=request.text[:80])
        return data_pb2.StoreTranscriptionResponse(id=row_id)

    # ── StoreVideoClip ──

    def StoreVideoClip(self, request_iterator, context):
        metadata_msg = None
        mp4_parts: list[bytes] = []
        embed_parts: list[bytes] = []
        mp4_media_id: str | None = None
        embed_media_id: str | None = None
        expected_mp4_chunk = 0
        expected_embed_chunk = 0
        mp4_complete = False
        embed_complete = False

        for item in request_iterator:
            payload = item.WhichOneof("payload")
            if payload == "metadata":
                metadata_msg = item.metadata
                continue

            if payload == "mp4_chunk":
                chunk = item.mp4_chunk
                if mp4_media_id is None:
                    mp4_media_id = chunk.media_id
                if chunk.media_id != mp4_media_id:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("StoreVideoClip mp4 media_id changed mid-stream")
                    return data_pb2.StoreVideoClipResponse()
                if chunk.chunk_index != expected_mp4_chunk:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"StoreVideoClip mp4 chunk gap: expected {expected_mp4_chunk}, got {chunk.chunk_index}"
                    )
                    return data_pb2.StoreVideoClipResponse()
                mp4_parts.append(chunk.data)
                expected_mp4_chunk += 1
                if chunk.is_last:
                    mp4_complete = True
                continue

            if payload == "embed_chunk":
                chunk = item.embed_chunk
                if embed_media_id is None:
                    embed_media_id = chunk.media_id
                if chunk.media_id != embed_media_id:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("StoreVideoClip embed media_id changed mid-stream")
                    return data_pb2.StoreVideoClipResponse()
                if chunk.chunk_index != expected_embed_chunk:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"StoreVideoClip embed chunk gap: expected {expected_embed_chunk}, got {chunk.chunk_index}"
                    )
                    return data_pb2.StoreVideoClipResponse()
                embed_parts.append(chunk.data)
                expected_embed_chunk += 1
                if chunk.is_last:
                    embed_complete = True
                continue

            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("StoreVideoClip request item must contain metadata, mp4_chunk, or embed_chunk")
            return data_pb2.StoreVideoClipResponse()

        if metadata_msg is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("StoreVideoClip metadata is required")
            return data_pb2.StoreVideoClipResponse()

        if not mp4_complete:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("StoreVideoClip stream ended before final mp4 chunk")
            return data_pb2.StoreVideoClipResponse()

        mp4_data = b"".join(mp4_parts)
        embed_data = b"".join(embed_parts) if embed_complete else b""

        ts = time.time()

        # 1. Save MP4 to disk
        clip_name = f"clip_{ts:.6f}.mp4"
        clip_path = self._video_dir / clip_name
        clip_path.write_bytes(mp4_data)

        # 2. Embed video via Nova
        try:
            embed_source = embed_data if embed_data else mp4_data
            embedding = embed_video(embed_source, dimension=self._embedding_dim)
        except Exception as e:
            log.error("data_service.embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {e}")
            return data_pb2.StoreVideoClipResponse()

        # 3. Insert metadata into SQLite
        clip_id = self._sqlite.insert_video_clip(
            clip_path=str(clip_path),
            start_timestamp=metadata_msg.start_timestamp,
            end_timestamp=metadata_msg.end_timestamp,
            num_frames=metadata_msg.num_frames,
            embedding_dimension=self._embedding_dim,
        )

        # 4. Store embedding in FAISS
        faiss_metadata = {
            "clip_id": clip_id,
            "start_timestamp": metadata_msg.start_timestamp,
            "end_timestamp": metadata_msg.end_timestamp,
        }
        faiss_id = self._faiss.add(embedding, faiss_metadata)

        # 5. Link FAISS ID back to SQLite
        self._sqlite.update_clip_faiss_id(clip_id, faiss_id)

        log.info(
            "data_service.clip_stored",
            clip_id=clip_id,
            faiss_id=faiss_id,
            path=str(clip_path),
            num_frames=metadata_msg.num_frames,
        )
        return data_pb2.StoreVideoClipResponse(clip_id=clip_id, faiss_id=faiss_id)

    # ── Search ──

    def Search(self, request, context):
        top_k = request.top_k if request.top_k > 0 else 5
        query_type = request.WhichOneof("query")
        query_embedding_purpose = "VIDEO_RETRIEVAL"

        try:
            if query_type == "text":
                query_vec = embed_text(
                    request.text,
                    dimension=self._embedding_dim,
                    embedding_purpose=query_embedding_purpose,
                )
            elif query_type == "image_data":
                query_vec = embed_image(
                    request.image_data,
                    dimension=self._embedding_dim,
                    embedding_purpose=query_embedding_purpose,
                )
            elif query_type == "video_data":
                query_vec = embed_video(
                    request.video_data,
                    dimension=self._embedding_dim,
                    embedding_purpose=query_embedding_purpose,
                )
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Search query must contain text, image_data, or video_data")
                return data_pb2.SearchResponse()
        except Exception as e:
            log.error("data_service.search_embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Query embedding failed: {e}")
            return data_pb2.SearchResponse()

        return self._search_video_clips(query_vec, top_k=top_k, query_type=query_type)

    def SearchVideoClipsByText(self, request, context):
        query_text = request.query_text.strip()
        if not query_text:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("SearchVideoClipsByText query_text is required")
            return data_pb2.SearchResponse()

        start_timestamp = request.start_timestamp if request.HasField("start_timestamp") else None
        end_timestamp = request.end_timestamp if request.HasField("end_timestamp") else None
        if (
            start_timestamp is not None
            and end_timestamp is not None
            and start_timestamp > end_timestamp
        ):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("SearchVideoClipsByText start_timestamp must be <= end_timestamp")
            return data_pb2.SearchResponse()

        top_k = request.top_k if request.top_k > 0 else 5
        try:
            query_vec = embed_text(
                query_text,
                dimension=self._embedding_dim,
                embedding_purpose="VIDEO_RETRIEVAL",
            )
        except Exception as e:
            log.error("data_service.text_search_embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Query embedding failed: {e}")
            return data_pb2.SearchResponse()

        return self._search_video_clips(
            query_vec,
            top_k=top_k,
            query_type="text",
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

    def StoreTrackedItem(self, request, context):
        title = request.title.strip()
        normalized_title = _normalize_item_title(title)
        if not normalized_title:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("StoreTrackedItem title is required")
            return data_pb2.StoreTrackedItemResponse()
        if not request.image_data:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("StoreTrackedItem image_data is required")
            return data_pb2.StoreTrackedItemResponse()

        existing = self._sqlite.get_tracked_item_by_normalized_title(normalized_title)
        if existing is not None:
            return data_pb2.StoreTrackedItemResponse(
                item_id=existing["id"],
                created=False,
                title=existing["title"],
                normalized_title=existing["normalized_title"],
            )

        try:
            embedding = embed_image(
                request.image_data,
                dimension=self._embedding_dim,
                embedding_purpose="VIDEO_RETRIEVAL",
            )
        except Exception as exc:
            log.error("data_service.tracked_item_embed_error", error=str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Tracked item embedding failed: {exc}")
            return data_pb2.StoreTrackedItemResponse()

        registered_at = request.registered_at if request.registered_at > 0 else time.time()
        safe_slug = (normalized_title.replace(" ", "_") or "item")[:64]
        image_path = self._tracked_item_dir / f"tracked_{safe_slug}_{registered_at:.6f}.jpg"
        image_path.write_bytes(request.image_data)

        item_id, created = self._sqlite.insert_tracked_item(
            title=title,
            normalized_title=normalized_title,
            embedding_json=json.dumps(embedding.tolist()),
            reference_image_path=str(image_path),
            registered_at=registered_at,
        )
        log.info(
            "data_service.tracked_item_stored",
            item_id=item_id,
            title=title,
            created=created,
            image_path=str(image_path),
        )
        return data_pb2.StoreTrackedItemResponse(
            item_id=item_id,
            created=created,
            title=title,
            normalized_title=normalized_title,
        )

    def FindLatestTrackedItemOccurrence(self, request, context):
        normalized_title = _normalize_item_title(request.title)
        if not normalized_title:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("FindLatestTrackedItemOccurrence title is required")
            return data_pb2.FindLatestTrackedItemOccurrenceResponse()

        has_explicit_min_score = request.HasField("min_score")
        min_score = request.min_score if has_explicit_min_score else _TRACKED_ITEM_MIN_SCORE_DEFAULT
        top_k = request.top_k if request.top_k > 0 else _TRACKED_ITEM_SEARCH_TOP_K_DEFAULT

        item = self._sqlite.get_tracked_item_by_normalized_title(normalized_title)
        if item is None:
            return data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=False,
                title=request.title.strip(),
                min_score_used=min_score,
            )

        try:
            query_vec = json.loads(item["embedding_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.error(
                "data_service.tracked_item_embedding_invalid",
                item_id=item["id"],
                error=str(exc),
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Tracked item embedding is corrupted")
            return data_pb2.FindLatestTrackedItemOccurrenceResponse()

        min_score_candidates = [min_score]
        if not has_explicit_min_score:
            min_score_candidates.extend(
                candidate
                for candidate in _TRACKED_ITEM_RELAXED_MIN_SCORES
                if candidate < min_score
            )

        resolved_min_score = min_score
        latest = None
        for candidate_min_score in min_score_candidates:
            latest = self._find_latest_occurrence_for_embedding(
                query_vec,
                top_k=top_k,
                min_score=candidate_min_score,
                exclude_contains_timestamp=item.get("registered_at"),
                exclude_margin_seconds=_TRACKED_ITEM_REGISTRATION_CLIP_MARGIN_SECONDS,
            )
            if latest is None:
                log.info(
                    "data_service.tracked_item_no_result_outside_registration_clip",
                    item_id=item["id"],
                    title=item["title"],
                    registered_at=item.get("registered_at"),
                    min_score=candidate_min_score,
                    top_k=top_k,
                )
                latest = self._find_latest_occurrence_for_embedding(
                    query_vec,
                    top_k=top_k,
                    min_score=candidate_min_score,
                )

            if latest is not None:
                resolved_min_score = candidate_min_score
                break

        if latest is None:
            return data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=True,
                item_id=item["id"],
                title=item["title"],
                found_occurrence=False,
                min_score_used=min_score_candidates[-1],
            )

        clip, score = latest
        return data_pb2.FindLatestTrackedItemOccurrenceResponse(
            found_item=True,
            item_id=item["id"],
            title=item["title"],
            found_occurrence=True,
            latest_result=data_pb2.SearchResult(
                clip_id=clip["id"],
                score=score,
                start_timestamp=clip["start_timestamp"] or 0.0,
                end_timestamp=clip["end_timestamp"] or 0.0,
                clip_path=clip["clip_path"],
                num_frames=clip["num_frames"] or 0,
            ),
            min_score_used=resolved_min_score,
        )

    def _find_latest_occurrence_for_embedding(
        self,
        query_vec,
        *,
        top_k: int,
        min_score: float,
        exclude_contains_timestamp: float | None = None,
        exclude_margin_seconds: float = 0.0,
    ) -> tuple[dict, float] | None:
        search_k = max(top_k, top_k * _TRACKED_ITEM_SEARCH_OVERFETCH_MULTIPLIER)
        if self._faiss.size > 0:
            search_k = min(search_k, int(self._faiss.size))
        faiss_results = self._faiss.search(query_vec, k=search_k)

        cutoff_ts = time.time() - _TRACKED_ITEM_LOOKBACK_SECONDS
        recent_candidates: list[tuple[float, float, dict]] = []
        for faiss_id, score, _meta in faiss_results:
            if score < min_score:
                continue
            clip = self._sqlite.get_clip_by_faiss_id(faiss_id)
            if clip is None:
                continue

            clip_start = clip.get("start_timestamp") or 0.0
            clip_end = clip.get("end_timestamp") or 0.0
            if clip_end <= 0.0:
                clip_end = clip_start
            if clip_start <= 0.0:
                clip_start = clip_end

            if (
                exclude_contains_timestamp is not None
                and clip_start - exclude_margin_seconds
                <= exclude_contains_timestamp
                <= clip_end + exclude_margin_seconds
            ):
                continue

            if clip_end < cutoff_ts:
                continue

            recent_candidates.append((float(score), float(clip_end), clip))

        if not recent_candidates:
            return None

        recent_candidates.sort(key=lambda row: row[0], reverse=True)
        top_recent_by_score = recent_candidates[:top_k]
        top_recent_by_score.sort(key=lambda row: row[1], reverse=True)

        chosen_score, _chosen_end, chosen_clip = top_recent_by_score[0]
        return chosen_clip, chosen_score

    def _search_video_clips(
        self,
        query_vec,
        *,
        top_k: int,
        query_type: str,
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ):
        # Over-fetch a bit when filtering so we still have a reasonable chance of
        # returning up to top_k results after applying the time window.
        search_k = max(top_k, top_k * 4) if start_timestamp is not None or end_timestamp is not None else top_k
        faiss_results = self._faiss.search(query_vec, k=search_k)

        results = []
        for faiss_id, score, meta in faiss_results:
            clip = self._sqlite.get_clip_by_faiss_id(faiss_id)
            if clip is None:
                continue
            clip_start = clip["start_timestamp"] or 0.0
            clip_end = clip["end_timestamp"] or 0.0
            if start_timestamp is not None and clip_end < start_timestamp:
                continue
            if end_timestamp is not None and clip_start > end_timestamp:
                continue
            results.append(
                data_pb2.SearchResult(
                    clip_id=clip["id"],
                    score=score,
                    start_timestamp=clip_start,
                    end_timestamp=clip_end,
                    clip_path=clip["clip_path"],
                    num_frames=clip["num_frames"] or 0,
                )
            )
            if len(results) >= top_k:
                break

        log.info("data_service.search", query_type=query_type, num_results=len(results))
        return data_pb2.SearchResponse(results=results)

    # ── GetTranscriptionLog ──

    def GetTranscriptionLog(self, request, context):
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset if request.offset > 0 else 0

        rows, total = self._sqlite.query_transcriptions(limit=limit, offset=offset)

        entries = [
            data_pb2.TranscriptionLogEntry(
                id=r["id"],
                text=r["text"],
                confidence=r["confidence"] or 0.0,
                start_time=r["start_time"] or 0.0,
                end_time=r["end_time"] or 0.0,
                created_at=r["created_at"],
            )
            for r in rows
        ]
        return data_pb2.TranscriptionLogResponse(entries=entries, total_count=total)

    # ── RegisterApp ──

    def RegisterApp(self, request, context):
        # Validate input_schema_json is valid JSON
        schema_json = request.input_schema_json or "{}"
        try:
            json.loads(schema_json)
        except json.JSONDecodeError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"input_schema_json is not valid JSON: {e}")
            return data_pb2.RegisterAppResponse()

        row_id, created = self._sqlite.upsert_app(
            name=request.name,
            description=request.description,
            app_type=request.app_type or "on_demand",
            grpc_address=request.grpc_address,
            input_schema_json=schema_json,
        )
        log.info(
            "data_service.app_registered",
            name=request.name,
            id=row_id,
            created=created,
        )
        return data_pb2.RegisterAppResponse(id=row_id, created=created)

    # ── ListApps ──

    def ListApps(self, request, context):
        rows = self._sqlite.list_apps(
            enabled_only=request.enabled_only,
            app_type=request.app_type,
        )
        apps = [
            data_pb2.AppInfo(
                id=r["id"],
                name=r["name"],
                description=r["description"],
                app_type=r["app_type"],
                grpc_address=r["grpc_address"],
                input_schema_json=r["input_schema_json"],
                enabled=bool(r["enabled"]),
                registered_at=r["registered_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]
        return data_pb2.ListAppsResponse(apps=apps)

    # ── SetAppEnabled ──

    def SetAppEnabled(self, request, context):
        found = self._sqlite.set_app_enabled(request.name, request.enabled)
        if not found:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"App '{request.name}' not found")
            return data_pb2.SetAppEnabledResponse(success=False)
        log.info(
            "data_service.app_enabled_toggled",
            name=request.name,
            enabled=request.enabled,
        )
        return data_pb2.SetAppEnabledResponse(success=True)
    def GetTranscriptionsInRange(self, request, context):
        rows = self._sqlite.query_transcriptions_in_range(
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
        )
        return data_pb2.TranscriptionRangeResponse(
            entries=[
                data_pb2.TranscriptionLogEntry(
                    id=r["id"],
                    text=r["text"],
                    confidence=r["confidence"] or 0.0,
                    start_time=r["start_time"] or 0.0,
                    end_time=r["end_time"] or 0.0,
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        )

    # ── Preferences ──

    def GetPreference(self, request, context):
        value = self._sqlite.get_preference(request.key)
        if value is None:
            return data_pb2.GetPreferenceResponse(found=False)
        return data_pb2.GetPreferenceResponse(value=value, found=True)

    def SetPreference(self, request, context):
        self._sqlite.set_preference(request.key, request.value)
        log.info(
            "data_service.preference_set",
            key=request.key,
        )
        return data_pb2.SetPreferenceResponse(success=True)

    # ── GetVideoClip ──

    def GetVideoClip(self, request, context):
        clip = self._sqlite.get_clip_metadata(request.clip_id)
        if clip is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Clip {request.clip_id} not found")
            return

        clip_path = Path(clip["clip_path"])
        if not clip_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Clip file missing: {clip_path}")
            return

        mp4_data = clip_path.read_bytes()
        total = len(mp4_data)

        if total == 0:
            yield data_pb2.VideoClipChunk(
                data=b"",
                clip_id=clip["id"],
                chunk_index=0,
                is_last=True,
                start_timestamp=clip["start_timestamp"] or 0.0,
                end_timestamp=clip["end_timestamp"] or 0.0,
                num_frames=clip["num_frames"] or 0,
            )
            return

        chunk_index = 0
        for start in range(0, total, _MEDIA_CHUNK_BYTES):
            end = min(start + _MEDIA_CHUNK_BYTES, total)
            yield data_pb2.VideoClipChunk(
                data=mp4_data[start:end],
                clip_id=clip["id"],
                chunk_index=chunk_index,
                is_last=end >= total,
                start_timestamp=clip["start_timestamp"] or 0.0,
                end_timestamp=clip["end_timestamp"] or 0.0,
                num_frames=clip["num_frames"] or 0,
            )
            chunk_index += 1

    def GetVideoClipsInRange(self, request, context):
        rows = self._sqlite.query_video_clips_in_range(
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
        )
        return data_pb2.VideoClipRangeResponse(
            clips=[
                data_pb2.VideoClipMetadata(
                    clip_id=r["id"],
                    clip_path=r["clip_path"],
                    start_timestamp=r["start_timestamp"] or 0.0,
                    end_timestamp=r["end_timestamp"] or 0.0,
                    num_frames=r["num_frames"] or 0,
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        )

    def StoreNoteSummary(self, request, context):
        row_id = self._sqlite.insert_note_summary(
            summary_text=request.summary_text,
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
        )
        log.info(
            "data_service.note_summary_stored",
            id=row_id,
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
        )
        return data_pb2.StoreNoteSummaryResponse(id=row_id)

    def GetNoteSummaries(self, request, context):
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset if request.offset > 0 else 0
        rows, total = self._sqlite.query_note_summaries(limit=limit, offset=offset)
        return data_pb2.NoteSummariesResponse(
            entries=[
                data_pb2.NoteSummaryEntry(
                    id=r["id"],
                    summary_text=r["summary_text"],
                    start_timestamp=r["start_timestamp"],
                    end_timestamp=r["end_timestamp"],
                    created_at=r["created_at"],
                )
                for r in rows
            ],
            total_count=total,
        )

    def StoreFoodMacros(self, request, context):
        row_id = self._sqlite.insert_food_macros(
            product_name=request.product_name,
            brand=request.brand or None,
            barcode=request.barcode or None,
            basis=request.basis or "per serving",
            calories_kcal=request.calories_kcal,
            protein_g=request.protein_g,
            fat_g=request.fat_g,
            carbs_g=request.carbs_g,
            serving_size=request.serving_size or None,
            serving_quantity=request.serving_quantity or None,
            recorded_at=request.recorded_at or None,
        )
        log.info(
            "data_service.food_macros_stored",
            id=row_id,
            product_name=request.product_name,
            barcode=request.barcode or None,
        )
        return data_pb2.StoreFoodMacrosResponse(id=row_id)

    def GetFoodMacros(self, request, context):
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset if request.offset > 0 else 0
        rows, total = self._sqlite.query_food_macros(limit=limit, offset=offset)
        entries = []
        for row in rows:
            entry = data_pb2.FoodMacroEntry(
                id=row["id"],
                product_name=row["product_name"] or "",
                brand=row["brand"] or "",
                barcode=row["barcode"] or "",
                basis=row["basis"] or "",
                serving_size=row["serving_size"] or "",
                recorded_at=row["recorded_at"] or 0.0,
                created_at=row["created_at"],
            )
            if row["calories_kcal"] is not None:
                entry.calories_kcal = row["calories_kcal"]
            if row["protein_g"] is not None:
                entry.protein_g = row["protein_g"]
            if row["fat_g"] is not None:
                entry.fat_g = row["fat_g"]
            if row["carbs_g"] is not None:
                entry.carbs_g = row["carbs_g"]
            if row["serving_quantity"] is not None:
                entry.serving_quantity = row["serving_quantity"]
            entries.append(entry)

        return data_pb2.GetFoodMacrosResponse(
            entries=entries,
            total_count=total,
        )

    # ── StoreRecording ──

    def StoreRecording(self, request, context):
        clip = self._sqlite.get_clip_metadata(request.clip_id)
        if clip is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Clip {request.clip_id} not found")
            return data_pb2.StoreRecordingResponse()

        row_id = self._sqlite.insert_recording(
            clip_id=request.clip_id,
            started_at=request.started_at,
            ended_at=request.ended_at,
        )
        log.info(
            "data_service.recording_stored",
            id=row_id,
            clip_id=request.clip_id,
        )
        return data_pb2.StoreRecordingResponse(id=row_id, clip_id=request.clip_id)

    # ── GetRecordings ──

    def GetRecordings(self, request, context):
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset if request.offset > 0 else 0
        rows, total = self._sqlite.query_recordings(limit=limit, offset=offset)
        return data_pb2.GetRecordingsResponse(
            entries=[
                data_pb2.RecordingEntry(
                    id=r["id"],
                    clip_id=r["clip_id"],
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    created_at=r["created_at"],
                )
                for r in rows
            ],
            total_count=total,
        )

    # ── StoreFaceEmbedding ──

    def StoreFaceEmbedding(self, request, context):
        try:
            embedding = embed_face_image(request.image_data)
        except Exception as e:
            log.error("data_service.face_embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Face embedding failed: {e}")
            return data_pb2.StoreFaceEmbeddingResponse()

        with self._face_store_lock:
            # Match-and-store must be atomic across gRPC worker threads or identical
            # requests can race and create duplicate identities.
            try:
                matched = self._select_face_match(embedding)
            except Exception as e:
                log.error("data_service.face_match_error", error=str(e))
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Face match failed: {e}")
                return data_pb2.StoreFaceEmbeddingResponse()
            if matched is not None:
                matched_face_id = int(matched["face_id"])
                self._store_face_sighting_image(
                    face_id=matched_face_id,
                    image_data=request.image_data,
                    timestamp=request.timestamp,
                )
                self._sqlite.update_face_seen(matched_face_id, request.timestamp)

                exemplar_faiss_id = self._faces_faiss.add(
                    embedding,
                    {"face_id": matched_face_id, "timestamp": request.timestamp},
                )
                self._sqlite.update_face_faiss_id(matched_face_id, exemplar_faiss_id)
                log.info(
                    "data_service.face_dedup",
                    face_id=matched_face_id,
                    best_score=matched["max_score"],
                    aggregate_score=matched["aggregate_score"],
                    vote_count=matched["vote_count"],
                    exemplar_faiss_id=exemplar_faiss_id,
                )
                return data_pb2.StoreFaceEmbeddingResponse(
                    face_id=matched_face_id,
                    faiss_id=exemplar_faiss_id,
                    is_new=False,
                    matched_face_id=matched_face_id,
                )

            # New face - save thumbnail, insert SQLite, add to FAISS
            thumbnail_name = f"face_{request.timestamp:.6f}.jpg"
            thumbnail_path = self._face_dir / thumbnail_name
            thumbnail_path.write_bytes(request.image_data)

            face_id = self._sqlite.insert_face(
                thumbnail_path=str(thumbnail_path),
                confidence=request.confidence,
                first_seen=request.timestamp,
            )

            try:
                faiss_id = self._faces_faiss.add(
                    embedding, {"face_id": face_id, "timestamp": request.timestamp}
                )
            except Exception as e:
                log.error("data_service.face_store_index_error", error=str(e), face_id=face_id)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Face index update failed: {e}")
                return data_pb2.StoreFaceEmbeddingResponse()
            self._sqlite.update_face_faiss_id(face_id, faiss_id)
            self._sqlite.insert_face_sighting(
                face_id=face_id,
                image_path=str(thumbnail_path),
                seen_at=request.timestamp,
            )

            log.info(
                "data_service.face_stored",
                face_id=face_id,
                faiss_id=faiss_id,
                thumbnail=str(thumbnail_path),
            )
            return data_pb2.StoreFaceEmbeddingResponse(
                face_id=face_id, faiss_id=faiss_id, is_new=True
            )

    # ── SearchFaces ──

    def SearchFaces(self, request, context):
        top_k = request.top_k if request.top_k > 0 else 5

        try:
            query_vec = embed_face_image(request.image_data)
        except Exception as e:
            log.error("data_service.face_search_embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Face query embedding failed: {e}")
            return data_pb2.SearchFacesResponse()

        try:
            grouped_matches = self._group_face_matches(
                query_vec,
                top_k=max(top_k * _FACE_SEARCH_RESULT_MULTIPLIER, top_k),
            )
        except Exception as e:
            log.error("data_service.face_search_match_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Face search failed: {e}")
            return data_pb2.SearchFacesResponse()

        results = []
        core_matches = [match for match in grouped_matches if self._is_core_face(match["face"])]
        for match in core_matches[:top_k]:
            face = match["face"]
            results.append(
                data_pb2.FaceSearchResult(
                    face_id=face["id"],
                    score=float(match["aggregate_score"]),
                    first_seen=face["first_seen"],
                    last_seen=face["last_seen"],
                    seen_count=face["seen_count"],
                    thumbnail_path=face["thumbnail_path"],
                    name=face["display_name"] or "",
                    note=face["display_note"] or "",
                )
            )

        log.info("data_service.face_search", num_results=len(results))
        return data_pb2.SearchFacesResponse(results=results)

    def ListFaces(self, request, context):
        limit = request.limit if request.limit > 0 else 50
        offset = request.offset if request.offset >= 0 else 0
        rows, total = self._sqlite.list_faces(
            limit=limit,
            offset=offset,
            min_seen_count=FACE_CLUSTER_MIN_SIGHTINGS,
        )
        entries = [self._serialize_face_entry(row) for row in rows]
        return data_pb2.ListFacesResponse(entries=entries, total_count=total)

    def GetFaceImage(self, request, context):
        face = self._sqlite.get_face(request.face_id)
        if face is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Face {request.face_id} was not found.")
            return data_pb2.GetFaceImageResponse()

        if request.HasField("sighting_index"):
            sighting = self._sqlite.get_face_sighting_by_index(
                request.face_id,
                request.sighting_index,
            )
            if sighting is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(
                    f"Face sighting {request.sighting_index} for {request.face_id} was not found."
                )
                return data_pb2.GetFaceImageResponse()
            image_path = Path(sighting["image_path"])
        else:
            image_path = Path(face["thumbnail_path"])

        if not image_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Face image for {request.face_id} is missing.")
            return data_pb2.GetFaceImageResponse()

        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        return data_pb2.GetFaceImageResponse(
            image_data=image_path.read_bytes(),
            content_type=content_type,
        )

    def SetFaceName(self, request, context):
        display_name = request.name.strip()
        display_note = request.note.strip()
        updated = self._sqlite.set_face_metadata(
            request.face_id,
            display_name,
            display_note,
        )
        if not updated:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Face {request.face_id} was not found.")
            return data_pb2.SetFaceNameResponse()

        log.info(
            "data_service.face_name_updated",
            face_id=request.face_id,
            has_name=bool(display_name),
            has_note=bool(display_note),
        )
        return data_pb2.SetFaceNameResponse(
            updated=True,
            name=display_name,
            note=display_note,
        )

    def ClearFaces(self, request, context):
        faces_deleted, sightings_deleted, image_paths = self._sqlite.clear_faces()
        self._faces_faiss.clear(persist=True)
        self._delete_face_images(image_paths)
        log.info(
            "data_service.faces_cleared",
            faces_deleted=faces_deleted,
            sightings_deleted=sightings_deleted,
        )
        return data_pb2.ClearFacesResponse(
            faces_deleted=faces_deleted,
            sightings_deleted=sightings_deleted,
        )

    def DeleteFace(self, request, context):
        with self._face_store_lock:
            face = self._sqlite.get_face(request.face_id)
            if face is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Face {request.face_id} was not found.")
                return data_pb2.DeleteFaceResponse()

            id_map = self._faces_faiss.rebuild(
                lambda _idx, meta: int(meta.get("face_id", -1)) != request.face_id,
                persist=True,
            )
            deleted, sightings_deleted, image_paths = self._sqlite.delete_face(request.face_id)
            if not deleted:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Face {request.face_id} was not found.")
                return data_pb2.DeleteFaceResponse()

            for row in self._sqlite.list_all_faces():
                current_faiss_id = row["faiss_id"]
                new_faiss_id = id_map.get(current_faiss_id) if current_faiss_id is not None else None
                self._sqlite.update_face_faiss_id(row["id"], new_faiss_id)

        self._delete_face_images(image_paths)
        log.info(
            "data_service.face_deleted",
            face_id=request.face_id,
            sightings_deleted=sightings_deleted,
        )
        return data_pb2.DeleteFaceResponse(
            deleted=True,
            sightings_deleted=sightings_deleted,
        )

    def _serialize_face_entry(self, row: dict) -> data_pb2.FaceEntry:
        # Limit the preview collage to four most recent sightings.
        sighting_count = len(self._sqlite.list_face_sightings(row["id"], limit=4))
        entry = data_pb2.FaceEntry(
            face_id=row["id"],
            name=row["display_name"] or "",
            note=row["display_note"] or "",
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            seen_count=row["seen_count"],
            thumbnail_path=row["thumbnail_path"],
            collage_image_count=sighting_count,
        )
        if row["confidence"] is not None:
            entry.confidence = row["confidence"]
        return entry

    def _store_face_sighting_image(
        self,
        *,
        face_id: int,
        image_data: bytes,
        timestamp: float,
    ) -> None:
        image_path = self._face_dir / f"face_{face_id}_{timestamp:.6f}.jpg"
        image_path.write_bytes(image_data)
        self._sqlite.insert_face_sighting(
            face_id=face_id,
            image_path=str(image_path),
            seen_at=timestamp,
        )

    def _delete_face_images(self, image_paths: list[str]) -> None:
        for raw_path in image_paths:
            image_path = Path(raw_path)
            try:
                image_path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning(
                    "data_service.face_image_delete_failed",
                    path=str(image_path),
                    error=str(exc),
                )


def serve(port: int = DATA_PORT):
    """Start the DataService gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_send_message_length", 32 * 1024 * 1024),
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    servicer = DataServiceServicer()
    data_pb2_grpc.add_DataServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("data_service.started", port=port)

    def _shutdown(signum, frame):
        log.info("data_service.stopping", signal=signum)
        servicer.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
