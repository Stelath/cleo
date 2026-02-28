"""Tests for the Weather tool service."""

import json
from urllib.error import URLError

from apps.weather import WeatherServicer
from generated import tool_pb2


def _weather_payload(*, area: str = "San Francisco") -> dict:
    return {
        "current_condition": [
            {
                "temp_C": "12",
                "FeelsLikeC": "10",
                "humidity": "81",
                "windspeedKmph": "8",
                "weatherDesc": [{"value": "Light rain"}],
            }
        ],
        "nearest_area": [{"areaName": [{"value": area}]}],
    }


class TestWeatherServicer:
    def _make_servicer(self, mocker):
        servicer = WeatherServicer.__new__(WeatherServicer)
        servicer._frontend = mocker.MagicMock()
        servicer._frontend_channel = mocker.MagicMock()
        return servicer

    def _mock_urlopen_payload(self, mocker, payload: dict):
        mock_urlopen = mocker.patch("apps.weather.urlopen")
        mock_response = mocker.MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response
        return mock_urlopen

    def test_tool_metadata(self):
        servicer = WeatherServicer.__new__(WeatherServicer)
        assert servicer.tool_name == "weather"
        assert servicer.tool_type == "on_demand"
        assert "weather" in servicer.tool_description.lower()
        schema = servicer.tool_input_schema
        assert schema["type"] == "object"
        assert "location" in schema["properties"]
        assert "query" in schema["properties"]

    def test_execute_returns_weather_and_notifies_frontend(self, mock_grpc_context, mocker):
        servicer = self._make_servicer(mocker)
        self._mock_urlopen_payload(mocker, _weather_payload(area="San Francisco"))

        request = tool_pb2.ToolRequest(
            tool_name="weather",
            parameters_json=json.dumps({"location": "San Francisco"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        assert "Weather for San Francisco" in response.result_text
        assert "Light rain" in response.result_text
        servicer._frontend.ShowCard.assert_called_once()

        card_request = servicer._frontend.ShowCard.call_args.args[0]
        assert len(card_request.cards) == 1
        card = card_request.cards[0]
        assert "San Francisco" in card.title
        assert card_request.position == "right"
        assert card_request.duration_ms == 8000

        meta_keys = [kv.key for kv in card.meta]
        assert "Temperature" in meta_keys
        assert "Feels Like" in meta_keys
        assert "Humidity" in meta_keys
        assert "Wind" in meta_keys
        assert "Condition" in meta_keys
        assert "__card_type" in meta_keys

        card_type_kv = next(kv for kv in card.meta if kv.key == "__card_type")
        assert card_type_kv.value == "weather"

    def test_execute_extracts_location_from_query(self, mock_grpc_context, mocker):
        servicer = self._make_servicer(mocker)
        mock_urlopen = self._mock_urlopen_payload(mocker, _weather_payload(area="Tokyo"))

        request = tool_pb2.ToolRequest(
            tool_name="weather",
            parameters_json=json.dumps({"query": "what is the weather in Tokyo?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)

        assert response.success
        request_arg = mock_urlopen.call_args.args[0]
        assert request_arg.full_url.endswith("/Tokyo?format=j1")

    def test_execute_returns_failure_when_lookup_fails(self, mock_grpc_context, mocker):
        servicer = self._make_servicer(mocker)
        mocker.patch("apps.weather.urlopen", side_effect=URLError("offline"))

        request = tool_pb2.ToolRequest(tool_name="weather", parameters_json="{}")
        response = servicer.Execute(request, mock_grpc_context)

        assert not response.success
        assert "Could not fetch weather right now" in response.result_text
        servicer._frontend.ShowCard.assert_not_called()
