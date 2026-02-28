"""DataService gRPC servicer — owns SQLite, FAISS, Bedrock embeddings, and video storage."""

import signal
import time
from concurrent import futures
from pathlib import Path

import grpc
import structlog

from services.config import DATA_PORT, EMBEDDING_DIMENSION, VIDEO_STORAGE_DIR
from services.data.embedding import embed_image, embed_text, embed_video
from services.data.sql.db import CleoSQLite
from services.data.vector.faiss_db import FaissDB
from generated import data_pb2, data_pb2_grpc

log = structlog.get_logger()


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

    def StoreVideoClip(self, request, context):
        ts = time.time()

        # 1. Save MP4 to disk
        clip_name = f"clip_{ts:.6f}.mp4"
        clip_path = self._video_dir / clip_name
        clip_path.write_bytes(request.mp4_data)

        # 2. Embed video via Nova
        try:
            embedding = embed_video(request.mp4_data, dimension=self._embedding_dim)
        except Exception as e:
            log.error("data_service.embed_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Embedding failed: {e}")
            return data_pb2.StoreVideoClipResponse()

        # 3. Insert metadata into SQLite
        clip_id = self._sqlite.insert_video_clip(
            clip_path=str(clip_path),
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp,
            num_frames=request.num_frames,
            embedding_dimension=self._embedding_dim,
        )

        # 4. Store embedding in FAISS
        metadata = {
            "clip_id": clip_id,
            "start_timestamp": request.start_timestamp,
            "end_timestamp": request.end_timestamp,
        }
        faiss_id = self._faiss.add(embedding, metadata)

        # 5. Link FAISS ID back to SQLite
        self._sqlite.update_clip_faiss_id(clip_id, faiss_id)

        log.info(
            "data_service.clip_stored",
            clip_id=clip_id,
            faiss_id=faiss_id,
            path=str(clip_path),
            num_frames=request.num_frames,
        )
        return data_pb2.StoreVideoClipResponse(clip_id=clip_id, faiss_id=faiss_id)

    # ── Search ──

    def Search(self, request, context):
        top_k = request.top_k if request.top_k > 0 else 5
        query_type = request.WhichOneof("query")

        try:
            if query_type == "text":
                query_vec = embed_text(request.text, dimension=self._embedding_dim)
            elif query_type == "image_data":
                query_vec = embed_image(request.image_data, dimension=self._embedding_dim)
            elif query_type == "video_data":
                query_vec = embed_video(request.video_data, dimension=self._embedding_dim)
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

    # ── GetVideoClip ──

    def GetVideoClip(self, request, context):
        clip = self._sqlite.get_clip_metadata(request.clip_id)
        if clip is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Clip {request.clip_id} not found")
            return data_pb2.VideoClipResponse()

        clip_path = Path(clip["clip_path"])
        if not clip_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Clip file missing: {clip_path}")
            return data_pb2.VideoClipResponse()

        return data_pb2.VideoClipResponse(
            mp4_data=clip_path.read_bytes(),
            start_timestamp=clip["start_timestamp"] or 0.0,
            end_timestamp=clip["end_timestamp"] or 0.0,
            num_frames=clip["num_frames"] or 0,
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
