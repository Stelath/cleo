"""Tests for the Save Video tool service."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generated import data_pb2, sensor_pb2, tool_pb2
from apps.save_video import SaveVideoServicer
from services.media.camera_transport import AssembledCameraFrame


def _make_camera_chunks(n: int, width: int = 64, height: int = 48):
    """Create n single-chunk CameraFrameChunk messages (RGB24, is_last=True)."""
    chunks = []
    for i in range(n):
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8).tobytes()
        chunks.append(
            sensor_pb2.CameraFrameChunk(
                data=data,
                frame_id=f"frame-{i}",
                chunk_index=0,
                is_last=True,
                width=width,
                height=height,
                timestamp=1000.0 + i / 30.0,
                encoding=sensor_pb2.FRAME_ENCODING_RGB24,
                key_frame=True,
            )
        )
    return chunks


class TestSaveVideoProperties:
    def test_tool_name(self):
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        assert servicer.tool_name == "save_video"

    def test_tool_type_is_on_demand(self):
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        assert servicer.tool_type == "on_demand"

    def test_tool_description_not_empty(self):
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        assert len(servicer.tool_description) > 0

    def test_tool_input_schema_has_type_and_properties(self):
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        schema = servicer.tool_input_schema
        assert schema["type"] == "object"
        assert "properties" in schema


class TestSaveVideoExecute:
    def _make_servicer(self):
        """Build a SaveVideoServicer with mocked gRPC dependencies."""
        servicer = SaveVideoServicer.__new__(SaveVideoServicer)
        servicer._sensor_channel = MagicMock()
        servicer._sensor = MagicMock()
        servicer._data_client = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._frontend = MagicMock()
        return servicer

    def test_execute_returns_quickly(self):
        servicer = self._make_servicer()
        with patch("threading.Thread") as mock_thread:
            success, msg = servicer.execute({})
        
        assert success
        assert "Started saving clip (past 30s, future 60s)" in msg
        mock_thread.assert_called_once()
        servicer._frontend.ShowNotification.assert_called_once()

    def test_capture_and_save_impl_success(self):
        servicer = self._make_servicer()
        
        past_chunks = _make_camera_chunks(30)
        future_chunks = _make_camera_chunks(30)
        
        # Adjust future chunks timestamps to mimic real forward progression
        last_past_ts = past_chunks[-1].timestamp
        for i, c in enumerate(future_chunks):
            c.timestamp = last_past_ts + 0.1 + (i / 30.0)

        servicer._sensor.GetBufferedFrames.return_value = iter(past_chunks)
        servicer._sensor.StreamCamera.return_value = iter(future_chunks)
        servicer._data_client.store_clip.return_value = data_pb2.StoreVideoClipResponse(
            clip_id=42, faiss_id=7,
        )

        with patch("apps.save_video.encode_rgb_to_h264_annexb", return_value=b"\x00\x00\x00\x01frame"):
            with patch("apps.save_video.h264_frames_to_mp4", return_value=b"mp4-bytes"):
                with patch("apps.save_video.downsample_mp4_for_embedding", return_value=b"embed-bytes"):
                    servicer._capture_and_save_impl(30.0, 1.0)  # 1.0s future to ensure it reads from stream

        servicer._data_client.store_clip.assert_called_once()
        servicer._frontend.ShowNotification.assert_called_once()
        
        # Verify both past and future streams were called
        servicer._sensor.GetBufferedFrames.assert_called_once()
        servicer._sensor.StreamCamera.assert_called_once()

    def test_capture_and_save_impl_no_frames(self):
        servicer = self._make_servicer()
        servicer._sensor.GetBufferedFrames.return_value = iter([])
        servicer._sensor.StreamCamera.return_value = iter([])

        with pytest.raises(RuntimeError, match="No historical or future video frames available"):
            servicer._capture_and_save_impl(30.0, 45.0)

    def test_capture_and_save_impl_store_failure(self):
        servicer = self._make_servicer()
        chunks = _make_camera_chunks(10)
        servicer._sensor.GetBufferedFrames.return_value = iter(chunks)
        servicer._sensor.StreamCamera.return_value = iter([])
        servicer._data_client.store_clip.return_value = None

        with patch("apps.save_video.encode_rgb_to_h264_annexb", return_value=b"\x00\x00\x00\x01frame"):
            with patch("apps.save_video.h264_frames_to_mp4", return_value=b"mp4-bytes"):
                with patch("apps.save_video.downsample_mp4_for_embedding", return_value=b"embed-bytes"):
                    with pytest.raises(RuntimeError, match="Failed to store"):
                        servicer._capture_and_save_impl(30.0, 0.0)

    def test_capture_and_save_impl_respects_duration_param(self):
        servicer = self._make_servicer()
        chunks = _make_camera_chunks(10)
        servicer._sensor.GetBufferedFrames.return_value = iter(chunks)
        servicer._data_client.store_clip.return_value = data_pb2.StoreVideoClipResponse(
            clip_id=1, faiss_id=1,
        )

        with patch("apps.save_video.encode_rgb_to_h264_annexb", return_value=b"\x00\x00\x00\x01frame"):
            with patch("apps.save_video.h264_frames_to_mp4", return_value=b"mp4-bytes"):
                with patch("apps.save_video.downsample_mp4_for_embedding", return_value=b"embed-bytes"):
                    servicer._capture_and_save_impl(10.0, 0.0)

        call_args = servicer._sensor.GetBufferedFrames.call_args
        request = call_args[0][0]
        assert request.max_duration_seconds == 10.0
