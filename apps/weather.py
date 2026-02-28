"""Weather lookup tool service."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import grpc
import structlog

from apps.tool_base import ToolServiceBase, serve_tool
from generated import frontend_pb2, frontend_pb2_grpc
from services.config import FRONTEND_ADDRESS, WEATHER_PORT

log = structlog.get_logger()

_WTTR_BASE_URL = "https://wttr.in"
_WTTR_TIMEOUT_S = 8
_WTTR_USER_AGENT = "CleoWeather/0.1"


class WeatherServicer(ToolServiceBase):
    """Fetches current weather conditions from the internet."""

    @property
    def tool_name(self) -> str:
        return "weather"

    @property
    def tool_description(self) -> str:
        return (
            "Get current weather from the internet for the user's location or a named place. "
            "Use for questions about weather, temperature, rain, or forecast conditions."
        )

    @property
    def tool_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "Optional city, region, or location name. "
                        "If omitted, weather is fetched for the user's current area."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Optional natural-language weather request.",
                },
            },
        }

    def __init__(self, frontend_address: str = FRONTEND_ADDRESS):
        self._frontend_channel = grpc.insecure_channel(frontend_address)
        self._frontend = frontend_pb2_grpc.FrontendServiceStub(self._frontend_channel)

    def close(self) -> None:
        self._frontend_channel.close()

    def execute(self, params: dict) -> tuple[bool, str]:
        location = self._resolve_location(params)
        try:
            weather_text = self._fetch_weather(location)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("weather.lookup_failed", location=location, error=str(exc))
            return False, f"Could not fetch weather right now: {exc}"

        self._notify(weather_text)
        return True, weather_text

    def _resolve_location(self, params: dict) -> str:
        location = str(params.get("location", "")).strip()
        if location:
            return location

        query = str(params.get("query", "")).strip()
        if not query:
            return ""

        extracted = self._extract_location_from_query(query)
        return extracted or ""

    @staticmethod
    def _extract_location_from_query(query: str) -> str:
        stripped = query.strip().strip("?.!,")
        lowered = stripped.lower()
        for marker in (" in ", " at ", " for "):
            index = lowered.rfind(marker)
            if index == -1:
                continue
            candidate = stripped[index + len(marker) :].strip("?.!, ")
            if candidate:
                return candidate
        return ""

    def _fetch_weather(self, location: str) -> str:
        path = f"/{quote(location, safe='')}" if location else ""
        request = Request(
            f"{_WTTR_BASE_URL}{path}?format=j1",
            headers={
                "User-Agent": _WTTR_USER_AGENT,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=_WTTR_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))

        current = (payload.get("current_condition") or [{}])[0]
        nearest_area = (payload.get("nearest_area") or [{}])[0]

        area_name = ((nearest_area.get("areaName") or [{}])[0]).get("value", "")
        desc = ((current.get("weatherDesc") or [{}])[0]).get("value", "")
        temp_c = str(current.get("temp_C", "")).strip()
        feels_like_c = str(current.get("FeelsLikeC", "")).strip()
        humidity = str(current.get("humidity", "")).strip()
        wind_kph = str(current.get("windspeedKmph", "")).strip()

        location_label = location or area_name or "your area"

        conditions: list[str] = []
        if desc:
            conditions.append(desc)
        if temp_c:
            conditions.append(f"{temp_c}C")
        if feels_like_c:
            conditions.append(f"feels like {feels_like_c}C")

        extras: list[str] = []
        if humidity:
            extras.append(f"humidity {humidity}%")
        if wind_kph:
            extras.append(f"wind {wind_kph} km/h")

        details = ", ".join(conditions) if conditions else "conditions unavailable"
        if extras:
            return f"Weather for {location_label}: {details}. {', '.join(extras)}."
        return f"Weather for {location_label}: {details}."

    def _notify(self, weather_text: str) -> None:
        try:
            self._frontend.ShowNotification(
                frontend_pb2.NotificationRequest(
                    title="Weather",
                    message=weather_text,
                    style="info",
                    duration_ms=6000,
                ),
                timeout=2,
            )
        except grpc.RpcError as exc:
            log.warning("weather.notify_failed", error=str(exc))


def serve(port: int = WEATHER_PORT) -> None:
    servicer = WeatherServicer()
    try:
        serve_tool(servicer, port=port)
    finally:
        servicer.close()


if __name__ == "__main__":
    serve()
