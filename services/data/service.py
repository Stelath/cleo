"""DataService gRPC servicer — owns SQLite, FAISS, Bedrock embeddings, and video storage."""

import json
import signal
import time
from concurrent import futures
from pathlib import Path

import grpc
import structlog

from services.config import DATA_PORT, EMBEDDING_DIMENSION, VIDEO_STORAGE_DIR
from services.data.vector.embedding import embed_image, embed_text, embed_video
from services.data.sql.db import CleoSQLite
from services.data.vector.faiss_db import FaissDB
from generated import data_pb2, data_pb2_grpc

log = structlog.get_logger()

_MEDIA_CHUNK_BYTES = 256 * 1024


class DataServiceServicer(data_pb2_grpc.DataServiceServicer):
    """Manages persistent storage: SQLite for metadata, FAISS for vectors, disk for video."""

    def __init__(
        self,
        db_path: str = "data/cleo.db",
        index_path: str = "data/vector/clips.index",
        video_dir: str = VIDEO_STORAGE_DIR,
        embedding_dim: int = EMBEDDING_DIMENSION,
    ):
        self._sqlite = CleoSQLite(db_path=db_path)
        self._faiss = FaissDB(dimension=embedding_dim, index_path=index_path)
        self._video_dir = Path(video_dir)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_dim = embedding_dim
        log.info(
            "data_service.init",
            db_path=db_path,
            index_path=index_path,
            video_dir=video_dir,
            faiss_size=self._faiss.size,
        )

    def shutdown(self):
        """Persist state and close connections."""
        self._faiss.save()
        self._sqlite.close()
        log.info("data_service.shutdown")

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

        faiss_results = self._faiss.search(query_vec, k=top_k)

        results = []
        for faiss_id, score, meta in faiss_results:
            clip = self._sqlite.get_clip_by_faiss_id(faiss_id)
            if clip is None:
                continue
            results.append(
                data_pb2.SearchResult(
                    clip_id=clip["id"],
                    score=score,
                    start_timestamp=clip["start_timestamp"] or 0.0,
                    end_timestamp=clip["end_timestamp"] or 0.0,
                    clip_path=clip["clip_path"],
                    num_frames=clip["num_frames"] or 0,
                )
            )

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
