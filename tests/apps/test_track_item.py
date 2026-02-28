"""Tests for track item register and locate tools."""

import json
from io import BytesIO
from unittest.mock import MagicMock

from PIL import Image

from apps.track_item import TrackItemLocateServicer, TrackItemRegisterServicer
from generated import data_pb2, sensor_pb2, tool_pb2


def _capture_stream_from_rgb(rgb: tuple[int, int, int] = (255, 0, 0)):
    image = Image.new("RGB", (1, 1), rgb)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    jpeg = buf.getvalue()
    return iter(
        [
            sensor_pb2.CameraFrameChunk(
                data=jpeg,
                frame_id="test-frame",
                chunk_index=0,
                is_last=True,
                width=1,
                height=1,
                timestamp=1000.0,
                encoding=sensor_pb2.FRAME_ENCODING_JPEG,
                key_frame=True,
            )
        ]
    )


class TestTrackItemRegisterServicer:
    def _make_servicer(self):
        servicer = TrackItemRegisterServicer.__new__(TrackItemRegisterServicer)
        servicer._vision = MagicMock()
        servicer._sensor = MagicMock()
        servicer._data = MagicMock()
        servicer._frontend = MagicMock()
        return servicer

    def test_tool_metadata(self):
        servicer = TrackItemRegisterServicer.__new__(TrackItemRegisterServicer)
        assert servicer.tool_name == "track_item_register"
        assert servicer.tool_type == "on_demand"
        assert "register" in servicer.tool_description.lower()
        assert servicer.tool_input_schema["type"] == "object"

    def test_register_rejects_when_no_visible_trackable_item(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._sensor.CaptureFrame.return_value = _capture_stream_from_rgb((1, 2, 3))
        servicer._vision.verify_trackable_item.return_value = (False, "no object visible", None)

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_register",
                parameters_json=json.dumps({"title": "phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is False
        assert "No visible trackable item" in response.result_text
        servicer._data.StoreTrackedItem.assert_not_called()

    def test_register_stores_item_when_visible(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._sensor.CaptureFrame.return_value = _capture_stream_from_rgb((4, 5, 6))
        servicer._vision.verify_trackable_item.return_value = (True, "", None)
        servicer._data.StoreTrackedItem.return_value = data_pb2.StoreTrackedItemResponse(
            item_id=4,
            created=True,
            title="phone",
            normalized_title="phone",
        )

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_register",
                parameters_json=json.dumps({"title": "phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is True
        assert "Registered 'phone'" in response.result_text
        servicer._data.StoreTrackedItem.assert_called_once()
        servicer._frontend.ShowNotification.assert_called_once()

    def test_register_returns_existing_item_message(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._sensor.CaptureFrame.return_value = _capture_stream_from_rgb((7, 8, 9))
        servicer._vision.verify_trackable_item.return_value = (True, "", None)
        servicer._data.StoreTrackedItem.return_value = data_pb2.StoreTrackedItemResponse(
            item_id=4,
            created=False,
            title="phone",
            normalized_title="phone",
        )

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_register",
                parameters_json=json.dumps({"title": "phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is True
        assert "already registered" in response.result_text


class TestTrackItemLocateServicer:
    def _make_servicer(self):
        servicer = TrackItemLocateServicer.__new__(TrackItemLocateServicer)
        servicer._data = MagicMock()
        servicer._frontend = MagicMock()
        servicer._locate_wait_seconds = 0.0
        servicer._locate_poll_interval_seconds = 0.1
        return servicer

    def test_tool_metadata(self):
        servicer = TrackItemLocateServicer.__new__(TrackItemLocateServicer)
        assert servicer.tool_name == "track_item_locate"
        assert servicer.tool_type == "on_demand"
        assert "last seen" in servicer.tool_description
        assert servicer.tool_input_schema["type"] == "object"

    def test_locate_requires_item_name(self, mock_grpc_context):
        servicer = self._make_servicer()

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_locate",
                parameters_json=json.dumps({"query": ""}),
            ),
            mock_grpc_context,
        )

        assert response.success is False
        assert "which tracked item" in response.result_text

    def test_locate_handles_unregistered_item(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._data.FindLatestTrackedItemOccurrence.return_value = (
            data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=False,
                title="phone",
                min_score_used=0.55,
            )
        )

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_locate",
                parameters_json=json.dumps({"title": "phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is True
        assert "Register it first" in response.result_text

    def test_locate_returns_latest_occurrence(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._data.GetVideoClip.return_value = iter(
            [
                data_pb2.VideoClipChunk(
                    data=b"fake-mp4-bytes",
                    clip_id=7,
                    chunk_index=0,
                    is_last=True,
                )
            ]
        )
        servicer._data.FindLatestTrackedItemOccurrence.return_value = (
            data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=True,
                item_id=1,
                title="phone",
                found_occurrence=True,
                latest_result=data_pb2.SearchResult(
                    clip_id=7,
                    score=0.81,
                    start_timestamp=1000.0,
                    end_timestamp=1010.0,
                    clip_path="/tmp/clip.mp4",
                    num_frames=10,
                ),
                min_score_used=0.55,
            )
        )

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_locate",
                parameters_json=json.dumps({"query": "where did i leave my phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is True
        assert "clip #7" in response.result_text
        assert "Playing the clip now" in response.result_text
        servicer._frontend.ShowCard.assert_called_once()
        servicer._frontend.RenderHtml.assert_called_once()

    def test_locate_waits_for_new_clip_before_giving_up(self, mock_grpc_context):
        servicer = self._make_servicer()
        servicer._locate_wait_seconds = 5.0
        servicer._locate_poll_interval_seconds = 0.1
        servicer._data.GetVideoClip.return_value = iter(
            [
                data_pb2.VideoClipChunk(
                    data=b"fake-mp4-bytes",
                    clip_id=9,
                    chunk_index=0,
                    is_last=True,
                )
            ]
        )
        servicer._data.FindLatestTrackedItemOccurrence.side_effect = [
            data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=True,
                item_id=1,
                title="phone",
                found_occurrence=False,
                min_score_used=0.55,
            ),
            data_pb2.FindLatestTrackedItemOccurrenceResponse(
                found_item=True,
                item_id=1,
                title="phone",
                found_occurrence=True,
                latest_result=data_pb2.SearchResult(
                    clip_id=9,
                    score=0.78,
                    start_timestamp=1000.0,
                    end_timestamp=1010.0,
                    clip_path="/tmp/clip.mp4",
                    num_frames=10,
                ),
                min_score_used=0.55,
            ),
        ]

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="track_item_locate",
                parameters_json=json.dumps({"query": "where did i leave my phone"}),
            ),
            mock_grpc_context,
        )

        assert response.success is True
        assert "Playing the clip now" in response.result_text
        assert servicer._data.FindLatestTrackedItemOccurrence.call_count == 2
