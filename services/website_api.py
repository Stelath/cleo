"""Small HTTP/JSON API for the website.

This BFF exposes the website read paths needed by the browser and proxies them
to the DataService over gRPC so the browser does not need direct gRPC access.
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import grpc
import structlog

from generated import data_pb2, data_pb2_grpc
from services.config import DATA_ADDRESS, WEBSITE_API_PORT

log = structlog.get_logger()

_DATA_READY_TIMEOUT_SECONDS = 1.0
_DATA_RECHECK_INTERVAL_SECONDS = 5.0


class _DataServiceUnavailableError(RuntimeError):
    """Raised when the DataService channel is not ready yet."""


class _DataServiceClient:
    """Track DataService channel readiness and retry connecting in the background."""

    def __init__(self, address: str):
        self._address = address
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._channel = grpc.insecure_channel(address)
        self._stub = data_pb2_grpc.DataServiceStub(self._channel)
        self._connected = False
        self._thread = threading.Thread(
            target=self._connection_loop,
            name="website-api-data-connection",
            daemon=True,
        )

    def start(self) -> None:
        """Start the background readiness loop."""
        self._thread.start()

    def close(self) -> None:
        """Stop the background readiness loop and close the channel."""
        self._stop_event.set()
        self._thread.join(timeout=_DATA_READY_TIMEOUT_SECONDS)
        self._channel.close()

    def get_stub(self) -> data_pb2_grpc.DataServiceStub:
        """Return the active stub when the DataService channel is ready."""
        with self._lock:
            if self._connected:
                return self._stub
        raise _DataServiceUnavailableError("Data service is still starting. Retrying shortly.")

    def _connection_loop(self) -> None:
        """Poll gRPC channel readiness until shutdown."""
        while not self._stop_event.is_set():
            try:
                grpc.channel_ready_future(self._channel).result(timeout=_DATA_READY_TIMEOUT_SECONDS)
            except grpc.FutureTimeoutError:
                self._set_connected(False)
            except Exception as exc:
                log.warning(
                    "website_api.data_service_probe_failed",
                    data_address=self._address,
                    error=str(exc),
                )
                self._set_connected(False)
            else:
                self._set_connected(True)

            self._stop_event.wait(_DATA_RECHECK_INTERVAL_SECONDS)

    def _set_connected(self, connected: bool) -> None:
        """Update connection state and log transitions once."""
        with self._lock:
            if self._connected == connected:
                if not connected:
                    log.debug("website_api.data_service_waiting", data_address=self._address)
                return
            self._connected = connected

        if connected:
            log.info("website_api.data_service_connected", data_address=self._address)
            return
        log.warning("website_api.data_service_waiting", data_address=self._address)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that can be restarted immediately on the same port."""

    allow_reuse_address = True


class _WebsiteApiHandler(BaseHTTPRequestHandler):
    """Serve the website JSON and media endpoints."""

    data_client: _DataServiceClient

    server_version = "CleoWebsiteAPI/0.2"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/faces":
                self._handle_list_faces(parsed.query)
                return
            if parsed.path == "/api/food-macros":
                self._handle_list_food_macros(parsed.query)
                return
            if parsed.path == "/api/notes":
                self._handle_list_notes(parsed.query)
                return
            if parsed.path == "/api/search":
                self._handle_search(parsed.query)
                return
            if parsed.path.startswith("/api/faces/") and parsed.path.endswith("/image"):
                self._handle_get_face_image(parsed.path)
                return
            if parsed.path.startswith("/api/faces/") and "/sightings/" in parsed.path:
                self._handle_get_face_sighting_image(parsed.path)
                return
            if parsed.path.startswith("/api/videos/"):
                self._handle_get_video_clip(parsed.path)
                return

            self._write_json(
                {"error": "not_found", "message": f"Unknown path: {parsed.path}"},
                status=HTTPStatus.NOT_FOUND,
            )
        except _DataServiceUnavailableError as exc:
            self._write_json(
                {"error": "data_service_unavailable", "message": str(exc)},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except grpc.RpcError as exc:
            self._write_grpc_error(exc, path=parsed.path)
        except Exception as exc:
            log.exception("website_api.request_failed", path=parsed.path, error=str(exc))
            self._write_json(
                {"error": "internal_error", "message": "Unable to load website data."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/faces/") and parsed.path.endswith("/name"):
                self._handle_set_face_name(parsed.path)
                return

            self._write_json(
                {"error": "not_found", "message": f"Unknown path: {parsed.path}"},
                status=HTTPStatus.NOT_FOUND,
            )
        except _DataServiceUnavailableError as exc:
            self._write_json(
                {"error": "data_service_unavailable", "message": str(exc)},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except grpc.RpcError as exc:
            self._write_grpc_error(exc, path=parsed.path)
        except ValueError as exc:
            self._write_json(
                {"error": "invalid_request", "message": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            log.exception("website_api.request_failed", path=parsed.path, error=str(exc))
            self._write_json(
                {"error": "internal_error", "message": "Unable to update website data."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        log.debug("website_api.request", message=format % args)

    def _handle_list_food_macros(self, query_string: str) -> None:
        params = parse_qs(query_string)
        limit = self._coerce_int(params.get("limit", ["50"])[0], default=50, minimum=1, maximum=200)
        offset = self._coerce_int(params.get("offset", ["0"])[0], default=0, minimum=0)
        response = self._get_data_stub().GetFoodMacros(
            data_pb2.GetFoodMacrosRequest(limit=limit, offset=offset)
        )
        self._write_json(
            {
                "entries": [self._serialize_food_macro(entry) for entry in response.entries],
                "totalCount": response.total_count,
                "limit": limit,
                "offset": offset,
            }
        )

    def _handle_list_faces(self, query_string: str) -> None:
        params = parse_qs(query_string)
        limit = self._coerce_int(params.get("limit", ["100"])[0], default=100, minimum=1, maximum=500)
        offset = self._coerce_int(params.get("offset", ["0"])[0], default=0, minimum=0)
        response = self._get_data_stub().ListFaces(
            data_pb2.ListFacesRequest(limit=limit, offset=offset)
        )
        self._write_json(
            {
                "entries": [self._serialize_face_entry(entry) for entry in response.entries],
                "totalCount": response.total_count,
                "limit": limit,
                "offset": offset,
            }
        )

    def _handle_list_notes(self, query_string: str) -> None:
        params = parse_qs(query_string)
        limit = self._coerce_int(params.get("limit", ["50"])[0], default=50, minimum=1, maximum=200)
        offset = self._coerce_int(params.get("offset", ["0"])[0], default=0, minimum=0)
        response = self._get_data_stub().GetNoteSummaries(
            data_pb2.NoteSummariesRequest(limit=limit, offset=offset)
        )
        self._write_json(
            {
                "entries": [self._serialize_note_summary(entry) for entry in response.entries],
                "totalCount": response.total_count,
                "limit": limit,
                "offset": offset,
            }
        )

    def _handle_search(self, query_string: str) -> None:
        params = parse_qs(query_string)
        query = ((params.get("q") or params.get("query") or [""])[0]).strip()
        limit = self._coerce_int(params.get("limit", ["5"])[0], default=5, minimum=1, maximum=20)
        start_timestamp = self._coerce_float(
            params.get("startTimestamp", [None])[0],
            minimum=0.0,
        )
        end_timestamp = self._coerce_float(
            params.get("endTimestamp", [None])[0],
            minimum=0.0,
        )
        if (
            start_timestamp is not None
            and end_timestamp is not None
            and start_timestamp > end_timestamp
        ):
            self._write_json(
                {
                    "error": "invalid_time_range",
                    "message": "startTimestamp must be less than or equal to endTimestamp.",
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if not query:
            if start_timestamp is None or end_timestamp is None:
                self._write_json(
                    {
                        "error": "invalid_query",
                        "message": "Provide a text query or both startTimestamp and endTimestamp.",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            range_response = self._get_data_stub().GetVideoClipsInRange(
                data_pb2.TimeRangeRequest(
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                )
            )
            entries = [self._serialize_video_clip_metadata(clip) for clip in range_response.clips]
        else:
            request = data_pb2.SearchVideoClipsByTextRequest(query_text=query, top_k=limit)
            if start_timestamp is not None:
                request.start_timestamp = start_timestamp
            if end_timestamp is not None:
                request.end_timestamp = end_timestamp

            response = self._get_data_stub().SearchVideoClipsByText(request)
            entries = [self._serialize_search_result(result) for result in response.results]

        self._write_json(
            {
                "query": query,
                "entries": entries,
                "limit": limit,
                "startTimestamp": start_timestamp,
                "endTimestamp": end_timestamp,
            }
        )

    def _handle_get_video_clip(self, path: str) -> None:
        clip_id = self._parse_clip_id(path)
        if clip_id is None:
            self._write_json(
                {"error": "invalid_clip_id", "message": f"Invalid clip path: {path}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        body_parts: list[bytes] = []
        metadata: data_pb2.VideoClipChunk | None = None
        for chunk in self._get_data_stub().GetVideoClip(data_pb2.GetVideoClipRequest(clip_id=clip_id)):
            if metadata is None:
                metadata = chunk
            body_parts.append(chunk.data)

        body = b"".join(body_parts)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if metadata is not None:
            self.send_header("X-Cleo-Clip-Id", str(metadata.clip_id))
            self.send_header("X-Cleo-Start-Timestamp", str(metadata.start_timestamp))
            self.send_header("X-Cleo-End-Timestamp", str(metadata.end_timestamp))
        self.end_headers()
        self.wfile.write(body)

    def _handle_get_face_image(self, path: str) -> None:
        face_id = self._parse_face_id(path, suffix="/image")
        if face_id is None:
            self._write_json(
                {"error": "invalid_face_id", "message": f"Invalid face image path: {path}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        response = self._get_data_stub().GetFaceImage(data_pb2.GetFaceImageRequest(face_id=face_id))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", response.content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(response.image_data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response.image_data)

    def _handle_get_face_sighting_image(self, path: str) -> None:
        parsed = self._parse_face_sighting_path(path)
        if parsed is None:
            self._write_json(
                {"error": "invalid_face_id", "message": f"Invalid face sighting path: {path}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        face_id, sighting_index = parsed
        request = data_pb2.GetFaceImageRequest(face_id=face_id, sighting_index=sighting_index)
        response = self._get_data_stub().GetFaceImage(request)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", response.content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(response.image_data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response.image_data)

    def _handle_set_face_name(self, path: str) -> None:
        face_id = self._parse_face_id(path, suffix="/name")
        if face_id is None:
            self._write_json(
                {"error": "invalid_face_id", "message": f"Invalid face name path: {path}"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        payload = self._read_json_body()
        raw_name = payload.get("name", "")
        raw_note = payload.get("note", "")
        if not isinstance(raw_name, str):
            raise ValueError("Field 'name' must be a string.")
        if not isinstance(raw_note, str):
            raise ValueError("Field 'note' must be a string.")

        response = self._get_data_stub().SetFaceName(
            data_pb2.SetFaceNameRequest(
                face_id=face_id,
                name=raw_name,
                note=raw_note,
            )
        )
        self._write_json(
            {
                "faceId": face_id,
                "updated": response.updated,
                "name": response.name,
                "note": response.note,
            }
        )

    def _write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _get_data_stub(cls) -> data_pb2_grpc.DataServiceStub:
        return cls.data_client.get_stub()

    def _write_grpc_error(self, exc: grpc.RpcError, *, path: str) -> None:
        code = exc.code()
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            status = HTTPStatus.BAD_REQUEST
            error = "invalid_request"
        elif code == grpc.StatusCode.NOT_FOUND:
            status = HTTPStatus.NOT_FOUND
            error = "not_found"
        elif code == grpc.StatusCode.UNAVAILABLE:
            status = HTTPStatus.SERVICE_UNAVAILABLE
            error = "data_service_unavailable"
        else:
            status = HTTPStatus.BAD_GATEWAY
            error = "data_service_error"

        details = exc.details() or "Data service request failed."
        log.warning(
            "website_api.grpc_error",
            path=path,
            grpc_code=str(code),
            details=details,
        )
        self._write_json(
            {"error": error, "message": details},
            status=status,
        )

    @staticmethod
    def _coerce_int(
        raw_value: str,
        *,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default

        if minimum is not None and value < minimum:
            return minimum
        if maximum is not None and value > maximum:
            return maximum
        return value

    @staticmethod
    def _coerce_float(
        raw_value: str | None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        if raw_value in (None, ""):
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        if minimum is not None and value < minimum:
            return minimum
        if maximum is not None and value > maximum:
            return maximum
        return value

    @staticmethod
    def _parse_clip_id(path: str) -> int | None:
        clip_id_text = path.removeprefix("/api/videos/").strip("/")
        if not clip_id_text.isdigit():
            return None
        return int(clip_id_text)

    @staticmethod
    def _parse_face_id(path: str, *, suffix: str) -> int | None:
        if not path.startswith("/api/faces/") or not path.endswith(suffix):
            return None
        face_id_text = path.removeprefix("/api/faces/").removesuffix(suffix).strip("/")
        if not face_id_text.isdigit():
            return None
        return int(face_id_text)

    @staticmethod
    def _parse_face_sighting_path(path: str) -> tuple[int, int] | None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 5 or parts[0] != "api" or parts[1] != "faces" or parts[3] != "sightings":
            return None
        face_id_text = parts[2]
        sighting_index_text = parts[4]
        if not face_id_text.isdigit() or not sighting_index_text.isdigit():
            return None
        return int(face_id_text), int(sighting_index_text)

    def _read_json_body(self) -> dict[str, object]:
        content_length = self._coerce_int(
            self.headers.get("Content-Length", "0"),
            default=0,
            minimum=0,
        )
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    @staticmethod
    def _serialize_food_macro(entry: data_pb2.FoodMacroEntry) -> dict[str, object]:
        return {
            "id": entry.id,
            "productName": entry.product_name,
            "brand": entry.brand,
            "barcode": entry.barcode,
            "basis": entry.basis,
            "caloriesKcal": entry.calories_kcal if entry.HasField("calories_kcal") else None,
            "proteinG": entry.protein_g if entry.HasField("protein_g") else None,
            "fatG": entry.fat_g if entry.HasField("fat_g") else None,
            "carbsG": entry.carbs_g if entry.HasField("carbs_g") else None,
            "servingSize": entry.serving_size,
            "servingQuantity": entry.serving_quantity if entry.HasField("serving_quantity") else None,
            "recordedAt": entry.recorded_at,
            "createdAt": entry.created_at,
        }

    @staticmethod
    def _serialize_note_summary(entry: data_pb2.NoteSummaryEntry) -> dict[str, object]:
        return {
            "id": entry.id,
            "summaryText": entry.summary_text,
            "startTimestamp": entry.start_timestamp,
            "endTimestamp": entry.end_timestamp,
            "createdAt": entry.created_at,
        }

    @staticmethod
    def _serialize_search_result(result: data_pb2.SearchResult) -> dict[str, object]:
        return {
            "clipId": result.clip_id,
            "score": result.score,
            "startTimestamp": result.start_timestamp,
            "endTimestamp": result.end_timestamp,
            "numFrames": result.num_frames,
            "videoUrl": f"/api/videos/{result.clip_id}",
        }

    @staticmethod
    def _serialize_video_clip_metadata(clip: data_pb2.VideoClipMetadata) -> dict[str, object]:
        return {
            "clipId": clip.clip_id,
            "score": 0.0,
            "startTimestamp": clip.start_timestamp,
            "endTimestamp": clip.end_timestamp,
            "numFrames": clip.num_frames,
            "videoUrl": f"/api/videos/{clip.clip_id}",
        }

    @staticmethod
    def _serialize_face_entry(entry: data_pb2.FaceEntry) -> dict[str, object]:
        return {
            "faceId": entry.face_id,
            "name": entry.name,
            "note": entry.note,
            "firstSeen": entry.first_seen,
            "lastSeen": entry.last_seen,
            "seenCount": entry.seen_count,
            "confidence": entry.confidence if entry.HasField("confidence") else None,
            "thumbnailPath": entry.thumbnail_path,
            "imageUrl": f"/api/faces/{entry.face_id}/image",
            "collageImageUrls": [
                f"/api/faces/{entry.face_id}/sightings/{index}"
                for index in range(entry.collage_image_count)
            ],
        }


def serve(port: int = WEBSITE_API_PORT, data_address: str = DATA_ADDRESS) -> None:
    """Start the website API server."""
    data_client = _DataServiceClient(data_address)
    data_client.start()
    _WebsiteApiHandler.data_client = data_client
    server = _ReusableThreadingHTTPServer(("127.0.0.1", port), _WebsiteApiHandler)
    server.daemon_threads = True
    log.info("website_api.starting", port=port, data_address=data_address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("website_api.shutdown_signal", signal="SIGINT")
    finally:
        server.server_close()
        data_client.close()
        log.info("website_api.stopped")


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
