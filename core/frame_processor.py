"""Background thread that streams camera frames, buffers 10s chunks, and embeds them."""

import threading
import time

import grpc
import numpy as np
import structlog

from data.vector.faiss_db import FaissDB
from generated import sensor_pb2, sensor_pb2_grpc

log = structlog.get_logger()

_BUFFER_SECONDS = 10
_TARGET_FPS = 2
_EMBEDDING_DIM = 512  # Placeholder dimension; adjust when embedding API is wired up


def embed_frame(frame: np.ndarray) -> np.ndarray | None:
    """Embed a single camera frame into a vector.

    Stubbed out — returns None until an external embedding API is wired up.
    When implemented, should return a float32 numpy array of shape (EMBEDDING_DIM,).
    """
    # TODO: Call external embedding API (e.g., CLIP) and return the vector
    return None


class FrameProcessor(threading.Thread):
    """Daemon thread that streams camera frames and builds a FAISS index.

    Connects to SensorService.StreamCamera at 2 FPS, buffers frames for 10 seconds,
    selects the middle frame as representative, passes it to embed_frame(), and
    stores the result in a FaissDB instance.
    """

    def __init__(
        self,
        sensor_address: str = "localhost:50051",
        faiss_db: FaissDB | None = None,
        index_path: str = "data/vector/frames.index",
    ):
        super().__init__(daemon=True, name="FrameProcessor")
        self._sensor_address = sensor_address
        self._db = faiss_db or FaissDB(dimension=_EMBEDDING_DIM, index_path=index_path)
        self._stop_event = threading.Event()

    @property
    def db(self) -> FaissDB:
        return self._db

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        log.info("frame_processor.started", address=self._sensor_address)

        channel = grpc.insecure_channel(
            self._sensor_address,
            options=[
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ],
        )
        stub = sensor_pb2_grpc.SensorServiceStub(channel)
        request = sensor_pb2.StreamRequest(fps=_TARGET_FPS)

        while not self._stop_event.is_set():
            try:
                self._process_stream(stub, request)
            except grpc.RpcError as e:
                if self._stop_event.is_set():
                    break
                log.error("frame_processor.rpc_error", error=str(e))
                time.sleep(2)

        channel.close()
        log.info("frame_processor.stopped")

    def _process_stream(self, stub, request):
        """Stream frames, buffer 10s chunks, embed middle frame."""
        frame_buffer: list[tuple[np.ndarray, float]] = []
        chunk_start = time.monotonic()

        for frame_msg in stub.StreamCamera(request):
            if self._stop_event.is_set():
                return

            frame = np.frombuffer(frame_msg.data, dtype=np.uint8).reshape(
                frame_msg.height, frame_msg.width, 3
            )
            frame_buffer.append((frame, frame_msg.timestamp))

            elapsed = time.monotonic() - chunk_start
            if elapsed >= _BUFFER_SECONDS:
                self._process_chunk(frame_buffer)
                frame_buffer = []
                chunk_start = time.monotonic()

    def _process_chunk(self, frames: list[tuple[np.ndarray, float]]):
        """Select middle frame from buffer, embed it, and store in FAISS."""
        if not frames:
            return

        mid_idx = len(frames) // 2
        mid_frame, mid_timestamp = frames[mid_idx]

        log.info(
            "frame_processor.chunk",
            num_frames=len(frames),
            selected_idx=mid_idx,
            timestamp=mid_timestamp,
        )

        embedding = embed_frame(mid_frame)
        if embedding is not None:
            metadata = {
                "timestamp": mid_timestamp,
                "chunk_frames": len(frames),
            }
            idx = self._db.add(embedding, metadata)
            log.info("frame_processor.embedded", db_id=idx, db_size=self._db.size)
        else:
            log.debug("frame_processor.embed_skipped", msg="embed_frame() returned None")
