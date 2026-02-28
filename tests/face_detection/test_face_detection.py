"""Tests for apps.face_detection — continuous face detection and tracking."""

import threading
from unittest.mock import MagicMock, patch

import cv2
import grpc
import numpy as np
import pytest

from apps.face_detection import (
    FaceDetectionLoop,
    FaceDetectionServicer,
    _NO_FACE_DETECTED_DETAIL,
    _crop_face,
)
from generated import sensor_pb2
from services.media.camera_transport import AssembledCameraFrame


# ── Helpers ──


def _make_fake_frame(width=64, height=48):
    """Create an assembled JPEG frame."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    return AssembledCameraFrame(
        frame_id="frame-1",
        data=encoded.tobytes(),
        width=width,
        height=height,
        timestamp=1000.0,
        encoding=sensor_pb2.FRAME_ENCODING_JPEG,
        key_frame=True,
    )


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


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


# ── _crop_face ──


class TestCropFace:
    def test_basic_crop(self):
        img_bgr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        bbox = {"Width": 0.2, "Height": 0.3, "Left": 0.4, "Top": 0.3}
        result = _crop_face(img_bgr, bbox)
        # Should produce valid JPEG bytes
        assert result[:2] == b"\xff\xd8"

    def test_padding_expands_region(self):
        img_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = {"Width": 0.2, "Height": 0.2, "Left": 0.4, "Top": 0.4}

        no_pad = _crop_face(img_bgr, bbox, padding=0.0)
        with_pad = _crop_face(img_bgr, bbox, padding=0.5)
        # Padded crop should be larger (more pixels → generally more JPEG bytes)
        assert len(with_pad) > len(no_pad)

    def test_clamps_to_image_bounds(self):
        img_bgr = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        # bbox near top-left corner with large padding
        bbox = {"Width": 0.1, "Height": 0.1, "Left": 0.0, "Top": 0.0}
        result = _crop_face(img_bgr, bbox, padding=1.0)
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
        image_bytes = mock_rek.detect_faces.call_args.kwargs["Image"]["Bytes"]
        assert image_bytes[:2] == b"\xff\xd8"

    def test_process_frame_converts_rgb24_to_jpeg_for_rekognition(self):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(0)

        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )

        rgb = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        frame = AssembledCameraFrame(
            frame_id="rgb24-test",
            data=rgb.tobytes(),
            width=64,
            height=48,
            timestamp=1000.0,
            encoding=sensor_pb2.FRAME_ENCODING_RGB24,
            key_frame=True,
        )

        loop._process_frame(frame)

        mock_rek.detect_faces.assert_called_once()
        image_bytes = mock_rek.detect_faces.call_args.kwargs["Image"]["Bytes"]
        assert image_bytes[:2] == b"\xff\xd8"

    @patch("apps.face_detection.grpc")
    def test_stream_loop_assembles_chunked_frames(self, mock_grpc_mod):
        mock_rek = MagicMock()
        loop = FaceDetectionLoop(
            sensor_address="localhost:99999",
            rekognition_client=mock_rek,
            data_address="localhost:99998",
        )

        frame = _make_fake_frame()
        data = bytes(frame.data)
        split = len(data) // 2

        chunks = [
            sensor_pb2.CameraFrameChunk(
                data=data[:split],
                frame_id=frame.frame_id,
                chunk_index=0,
                is_last=False,
                width=frame.width,
                height=frame.height,
                timestamp=frame.timestamp,
                encoding=sensor_pb2.FRAME_ENCODING_JPEG,
                key_frame=True,
            ),
            sensor_pb2.CameraFrameChunk(
                data=data[split:],
                frame_id=frame.frame_id,
                chunk_index=1,
                is_last=True,
                width=frame.width,
                height=frame.height,
                timestamp=frame.timestamp,
                encoding=sensor_pb2.FRAME_ENCODING_JPEG,
                key_frame=True,
            ),
        ]

        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_sensor_stub = MagicMock()
        mock_sensor_stub.StreamCamera.return_value = iter(chunks)

        with patch(
            "apps.face_detection.sensor_pb2_grpc.SensorServiceStub",
            return_value=mock_sensor_stub,
        ):
            with patch.object(loop, "_process_frame") as mock_process:
                loop._stream_loop()

        mock_process.assert_called_once()
        assembled = mock_process.call_args.args[0]
        assert assembled.frame_id == frame.frame_id
        assert assembled.data == data
        assert assembled.encoding == sensor_pb2.FRAME_ENCODING_JPEG

    @patch("apps.face_detection.grpc")
    def test_process_frame_stores_faces(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1)

        # Mock the data stub
        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_search_resp = MagicMock()
        mock_search_resp.results = []
        mock_stub_instance.SearchFaces.return_value = mock_search_resp
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

    @patch("apps.face_detection.grpc")
    def test_process_frame_skips_recent_matching_face(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1)

        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.face_id = 1
        mock_search_result.score = 0.99
        mock_search_resp = MagicMock()
        mock_search_resp.results = [mock_search_result]
        mock_stub_instance.SearchFaces.return_value = mock_search_resp

        with patch(
            "apps.face_detection.data_pb2_grpc.DataServiceStub",
            return_value=mock_stub_instance,
        ):
            loop = FaceDetectionLoop(
                sensor_address="localhost:99999",
                rekognition_client=mock_rek,
                data_address="localhost:99998",
            )
            loop._last_saved_by_face_id[1] = 950.0

            frame = _make_fake_frame()
            loop._process_frame(frame)

            mock_stub_instance.StoreFaceEmbedding.assert_not_called()

    @patch("apps.face_detection.grpc")
    def test_process_frame_does_not_skip_low_similarity_match(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1)

        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.face_id = 1
        mock_search_result.score = 0.40
        mock_search_resp = MagicMock()
        mock_search_resp.results = [mock_search_result]
        mock_stub_instance.SearchFaces.return_value = mock_search_resp
        mock_store_resp = MagicMock()
        mock_store_resp.face_id = 2
        mock_store_resp.is_new = True
        mock_stub_instance.StoreFaceEmbedding.return_value = mock_store_resp

        with patch(
            "apps.face_detection.data_pb2_grpc.DataServiceStub",
            return_value=mock_stub_instance,
        ):
            loop = FaceDetectionLoop(
                sensor_address="localhost:99999",
                rekognition_client=mock_rek,
                data_address="localhost:99998",
            )
            loop._last_saved_by_face_id[1] = 950.0

            frame = _make_fake_frame()
            loop._process_frame(frame)

            mock_stub_instance.StoreFaceEmbedding.assert_called_once()

    @patch("apps.face_detection.grpc")
    def test_process_frame_ignores_no_face_detected_rpc_error(self, mock_grpc_mod):
        mock_rek = MagicMock()
        mock_rek.detect_faces.return_value = _mock_rekognition_response(1)

        mock_channel = MagicMock()
        mock_grpc_mod.insecure_channel.return_value = mock_channel
        mock_stub_instance = MagicMock()
        mock_stub_instance.SearchFaces.side_effect = _FakeRpcError(
            grpc.StatusCode.INTERNAL,
            f"Face query embedding failed: {_NO_FACE_DETECTED_DETAIL}",
        )

        with patch(
            "apps.face_detection.data_pb2_grpc.DataServiceStub",
            return_value=mock_stub_instance,
        ):
            loop = FaceDetectionLoop(
                sensor_address="localhost:99999",
                rekognition_client=mock_rek,
                data_address="localhost:99998",
            )

            frame = _make_fake_frame()
            loop._process_frame(frame)

            mock_stub_instance.StoreFaceEmbedding.assert_not_called()
            assert len(loop.recent_faces) == 0

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
        mock_search_resp = MagicMock()
        mock_search_resp.results = []
        mock_stub_instance.SearchFaces.return_value = mock_search_resp
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
