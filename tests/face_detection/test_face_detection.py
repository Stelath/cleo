"""Tests for apps.face_detection — continuous face detection and tracking."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apps.face_detection import (
    FaceDetectionLoop,
    FaceDetectionServicer,
    _crop_face,
    _rgb_to_jpeg,
)


# ── Helpers ──


def _make_fake_frame(width=64, height=48):
    frame = MagicMock()
    frame.data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
    frame.width = width
    frame.height = height
    frame.timestamp = 1000.0
    return frame


def _mock_rekognition_response(n_faces=1, confidence=99.5):
    return {
        "FaceDetails": [
            {
                "BoundingBox": {
                    "Width": 0.2,
                    "Height": 0.3,
                    "Left": 0.4,
                    "Top": 0.3,
                },
                "Confidence": confidence,
            }
            for _ in range(n_faces)
        ]
    }


# ── _crop_face ──


class TestCropFace:
    def test_basic_crop(self):
        width, height = 64, 48
        frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
        bbox = {"Width": 0.2, "Height": 0.3, "Left": 0.4, "Top": 0.3}
        result = _crop_face(frame_data, width, height, bbox)
        # Should produce valid JPEG bytes
        assert result[:2] == b"\xff\xd8"

    def test_padding_expands_region(self):
        width, height = 100, 100
        frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
        bbox = {"Width": 0.2, "Height": 0.2, "Left": 0.4, "Top": 0.4}

        no_pad = _crop_face(frame_data, width, height, bbox, padding=0.0)
        with_pad = _crop_face(frame_data, width, height, bbox, padding=0.5)
        # Padded crop should be larger (more pixels → generally more JPEG bytes)
        assert len(with_pad) > len(no_pad)

    def test_clamps_to_image_bounds(self):
        width, height = 64, 48
        frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
        # bbox near top-left corner with large padding
        bbox = {"Width": 0.1, "Height": 0.1, "Left": 0.0, "Top": 0.0}
        result = _crop_face(frame_data, width, height, bbox, padding=1.0)
        # Should not crash, should return valid JPEG
        assert result[:2] == b"\xff\xd8"


# ── FaceDetectionLoop ──


class TestFaceDetectionLoop:
    def test_stop_event_terminates_thread(self):
        mock_rek = MagicMock()
        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )
        loop.start()
        loop.stop()
        loop.join(timeout=5.0)
        assert not loop.is_alive()

    def test_process_frame_calls_rekognition(self):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(0)

        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )

        frame = _make_fake_frame()
        loop._process_frame(frame)

        mock_rek.detect_faces.assert_called_once()
        call_kwargs = mock_rek.detect_faces.call_args
        image_bytes = call_kwargs[1]["Image"]["Bytes"] if "Image" in call_kwargs[1] else call_kwargs.kwargs["Image"]["Bytes"]
        assert image_bytes[:2] == b"\xff\xd8"

    @patch("apps.face_detection.grpc")
    def test_process_frame_stores_faces(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1)

        # Mock the data stub
        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_store_resp = MagicMock()
        mock_store_resp.face_id = 1
        mock_store_resp.is_new = True
        mock_stub_instance.StoreFaceEmbedding.return_value = mock_store_resp

        with patch("apps.face_detection.data_pb2_grpc.DataServiceStub", return_value=mock_stub_instance):
            loop = FaceDetectionLoop(
                sensor_address="localhost:99999",
                rekognition_client=mock_rek,
                data_address="localhost:99998",
            )

            frame = _make_fake_frame()
            loop._process_frame(frame)

            mock_stub_instance.StoreFaceEmbedding.assert_called_once()

    def test_low_confidence_faces_skipped(self):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1, confidence=50.0)

        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )

        frame = _make_fake_frame()
        loop._process_frame(frame)

        # No faces stored since confidence is below threshold
        assert len(loop.recent_faces) == 0

    @patch("apps.face_detection.grpc")
    def test_multiple_faces_in_frame(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(3)

        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_store_resp = MagicMock()
        mock_store_resp.face_id = 1
        mock_store_resp.is_new = True
        mock_stub_instance.StoreFaceEmbedding.return_value = mock_store_resp

        with patch("apps.face_detection.data_pb2_grpc.DataServiceStub", return_value=mock_stub_instance):
            loop = FaceDetectionLoop(
                sensor_address="localhost:99999",
                rekognition_client=mock_rek,
                data_address="localhost:99998",
            )

            frame = _make_fake_frame()
            loop._process_frame(frame)

            assert mock_stub_instance.StoreFaceEmbedding.call_count == 3

    def test_rekognition_error_does_not_crash(self):
        mock_rek = MagicMock()
        mock_rek.detect_faces.side_effect = RuntimeError("Rekognition timeout")

        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )

        frame = _make_fake_frame()
        # Should not raise
        loop._process_frame(frame)
        assert len(loop.recent_faces) == 0


# ── FaceDetectionServicer ──


class TestFaceDetectionServicer:
    def _make_servicer(self):
        """Create a FaceDetectionServicer with a mock sensor address so auto-start loop connects but properties are testable."""
        mock_rek = MagicMock()
        servicer = FaceDetectionServicer(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )
        return servicer

    def test_tool_name(self):
        servicer = self._make_servicer()
        assert servicer.tool_name == "face_detection"
        servicer._stop()

    def test_tool_type_is_active(self):
        servicer = self._make_servicer()
        assert servicer.tool_type == "active"
        servicer._stop()

    def test_auto_starts_on_init(self):
        servicer = self._make_servicer()
        assert servicer._loop is not None
        assert servicer._loop.is_alive()
        servicer._stop()

    def test_stop_and_restart(self):
        servicer = self._make_servicer()

        # Stop
        success, text = servicer.execute({"action": "stop"})
        assert success is True
        assert "stopped" in text.lower()

        # Restart
        success, text = servicer.execute({"action": "start"})
        assert success is True
        assert "started" in text.lower()

        # Cleanup
        servicer._stop()

    def test_stop_without_active_loop(self):
        servicer = self._make_servicer()
        # Stop the auto-started loop first
        servicer._stop()
        # Now _loop is None, stop again
        with servicer._lock:
            servicer._loop = None
        success, text = servicer.execute({"action": "stop"})
        assert success is False
        assert "not active" in text.lower()

    def test_double_start_returns_already_active(self):
        servicer = self._make_servicer()
        # Already auto-started, start again
        success, text = servicer.execute({"action": "start"})
        assert success is True
        assert "already" in text.lower()
        servicer._stop()

    def test_resolve_action_keywords(self):
        servicer = self._make_servicer()
        assert servicer._resolve_action({"query": "start detecting"}) == "start"
        assert servicer._resolve_action({"query": "resume detection"}) == "start"
        assert servicer._resolve_action({"query": "stop detecting"}) == "stop"
        assert servicer._resolve_action({"query": "pause face detection"}) == "stop"
        assert servicer._resolve_action({"query": "hello"}) is None
        servicer._stop()

    def test_tool_description_nonempty(self):
        servicer = self._make_servicer()
        assert len(servicer.tool_description) > 0
        servicer._stop()

    def test_tool_input_schema_has_type_object(self):
        servicer = self._make_servicer()
        assert servicer.tool_input_schema["type"] == "object"
        servicer._stop()
