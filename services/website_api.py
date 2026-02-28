"""Small HTTP/JSON API for the website.

This BFF exposes the website read paths needed by the browser. It reads
directly from SQLite so the browser does not need direct database or gRPC
access.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import structlog

from services.config import WEBSITE_API_PORT
from services.data.sql.db import CleoSQLite

log = structlog.get_logger()


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server that can be restarted immediately on the same port."""

    allow_reuse_address = True


class _WebsiteApiHandler(BaseHTTPRequestHandler):
    """Serve the website JSON endpoints."""

    sqlite: CleoSQLite

    server_version = "CleoWebsiteAPI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/food-macros":
                self._handle_list_food_macros(parsed.query)
                return
            if parsed.path == "/api/notes":
                self._handle_list_notes(parsed.query)
                return

            self._write_json(
                {"error": "not_found", "message": f"Unknown path: {parsed.path}"},
                status=HTTPStatus.NOT_FOUND,
            )
        except Exception as exc:
            log.exception("website_api.request_failed", path=parsed.path, error=str(exc))
            self._write_json(
                {"error": "internal_error", "message": "Unable to load website data."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        log.debug("website_api.request", message=format % args)

    def _handle_list_food_macros(self, query_string: str) -> None:
        params = parse_qs(query_string)
        limit = self._coerce_int(params.get("limit", ["50"])[0], default=50, minimum=1, maximum=200)
        offset = self._coerce_int(params.get("offset", ["0"])[0], default=0, minimum=0)
        rows, total = self.sqlite.query_food_macros(limit=limit, offset=offset)
        self._write_json(
            {
                "entries": [self._serialize_food_macro(row) for row in rows],
                "totalCount": total,
                "limit": limit,
                "offset": offset,
            }
        )

    def _handle_list_notes(self, query_string: str) -> None:
        params = parse_qs(query_string)
        limit = self._coerce_int(params.get("limit", ["50"])[0], default=50, minimum=1, maximum=200)
        offset = self._coerce_int(params.get("offset", ["0"])[0], default=0, minimum=0)
        rows, total = self.sqlite.query_note_summaries(limit=limit, offset=offset)
        self._write_json(
            {
                "entries": [self._serialize_note_summary(row) for row in rows],
                "totalCount": total,
                "limit": limit,
                "offset": offset,
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
    def _serialize_food_macro(row: dict[str, object]) -> dict[str, object]:
        return {
            "id": row["id"],
            "productName": row["product_name"] or "",
            "brand": row["brand"] or "",
            "barcode": row["barcode"] or "",
            "basis": row["basis"] or "",
            "caloriesKcal": row["calories_kcal"],
            "proteinG": row["protein_g"],
            "fatG": row["fat_g"],
            "carbsG": row["carbs_g"],
            "servingSize": row["serving_size"] or "",
            "servingQuantity": row["serving_quantity"],
            "recordedAt": row["recorded_at"],
            "createdAt": row["created_at"],
        }

    @staticmethod
    def _serialize_note_summary(row: dict[str, object]) -> dict[str, object]:
        return {
            "id": row["id"],
            "summaryText": row["summary_text"] or "",
            "startTimestamp": row["start_timestamp"],
            "endTimestamp": row["end_timestamp"],
            "createdAt": row["created_at"],
        }


def serve(port: int = WEBSITE_API_PORT, db_path: str = "data/cleo.db") -> None:
    """Start the website API server."""
    sqlite = CleoSQLite(db_path=db_path)
    _WebsiteApiHandler.sqlite = sqlite
    server = _ReusableThreadingHTTPServer(("127.0.0.1", port), _WebsiteApiHandler)
    server.daemon_threads = True
    log.info("website_api.starting", port=port, db_path=db_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("website_api.shutdown_signal", signal="SIGINT")
    finally:
        server.server_close()
        sqlite.close()
        log.info("website_api.stopped")


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
