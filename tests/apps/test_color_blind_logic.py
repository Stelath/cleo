"""Unit tests for the Color Blindness skill."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import grpc

from apps.color_blind import ColorBlindnessServicer, apply_color_correction, get_user_preference
from generated import sensor_pb2, tool_pb2

@pytest.fixture
def mock_frame():
    """Return a CameraFrame with synthetic 2x2 RGB data."""
    width, height = 2, 2
    # Simple RGB data: [Red, Green, Blue, White]
    data = np.array([
        [255, 0, 0], [0, 255, 0],
        [0, 0, 255], [255, 255, 255]
    ], dtype=np.uint8).tobytes()
    return sensor_pb2.CameraFrame(
        data=data,
        width=width,
        height=height,
        timestamp=123.456,
    )

class TestColorBlindLogic:
    def test_apply_color_correction_protanopia(self):
        width, height = 2, 2
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        # Just verify it doesn't crash and returns bytes of correct length
        corrected = apply_color_correction(data, width, height, "protanopia")
        assert isinstance(corrected, bytes)
        assert len(corrected) == len(data)

    def test_apply_color_correction_unknown_type(self):
        width, height = 1, 1
        data = np.zeros((height, width, 3), dtype=np.uint8).tobytes()
        corrected = apply_color_correction(data, width, height, "invalid")
        assert corrected == data

    def test_get_user_preference(self):
        pref = get_user_preference()
        assert pref in ["protanopia", "deuteranopia", "tritanopia"]

class TestColorBlindnessServicer:
    @patch("apps.color_blind.grpc.insecure_channel")
    @patch("generated.sensor_pb2_grpc.SensorServiceStub")
    def test_execute_success(self, mock_stub_class, mock_channel, mock_frame):
        # Setup mocks
        mock_stub = MagicMock()
        mock_stub_class.return_value = mock_stub
        mock_stub.CaptureFrame.return_value = mock_frame
        
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json=json.dumps({"query": "test"})
        )
        
        # We need a mock gRPC context
        context = MagicMock()
        
        response = servicer.Execute(request, context)
        assert response.success
        assert "Color assistance active" in response.result_text
        mock_stub.CaptureFrame.assert_called_once()

    @patch("apps.color_blind.grpc.insecure_channel")
    @patch("generated.sensor_pb2_grpc.SensorServiceStub")
    def test_execute_sensor_failure(self, mock_stub_class, mock_channel):
        mock_stub = MagicMock()
        mock_stub_class.return_value = mock_stub
        mock_stub.CaptureFrame.side_effect = Exception("Sensor offline")
        
        servicer = ColorBlindnessServicer()
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="{}"
        )
        context = MagicMock()
        
        response = servicer.Execute(request, context)
        assert not response.success
        assert "Failed to capture frame" in response.result_text


