"""Tests for Color Blindness Assist tool."""

import json
import numpy as np

from apps.color_blind import ColorBlindnessServicer
from generated import data_pb2, sensor_pb2, tool_pb2


def test_color_blind_correction_math(mocker):
    """Test the raw cv2 math logic for color blindness correction."""
    mocker.patch("apps.color_blind.grpc.insecure_channel")
    servicer = ColorBlindnessServicer()
    
    servicer.data_stub = mocker.MagicMock()
    pref_resp = data_pb2.GetPreferenceResponse(value="protanopia", found=True)
    servicer.data_stub.GetPreference.return_value = pref_resp

    servicer.sensor_stub = mocker.MagicMock()
    
    # Create a 2x2 image (4 pixels) of pure black
    # RGB format: [0, 0, 0]
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_resp = sensor_pb2.CameraFrame()
    frame_resp.data = img.tobytes()
    frame_resp.width = 2
    frame_resp.height = 2
    servicer.sensor_stub.CaptureFrame.return_value = frame_resp

    mock_imwrite = mocker.patch("apps.color_blind.cv2.imwrite")
    
    success, msg = servicer.execute({"query": ""})
    
    assert success
    assert "Applied protanopia" in msg
    mock_imwrite.assert_called_once()
    
    # Check the image written. 
    # cv2.imwrite converts RGB back to BGR inside execute
    args, _ = mock_imwrite.call_args
    written_img = args[1]
    
    # Check shape
    assert written_img.shape == (2, 2, 3)
    
    # Protanopia boosts green. Black (0,0,0) becomes (0,50,0) RGB -> (0,50,0) BGR
    assert written_img[0, 0, 0] == 0  # Blue
    assert written_img[0, 0, 1] == 50 # Green
    assert written_img[0, 0, 2] == 0  # Red


class TestColorBlindnessServicer:
    def test_execute_returns_success(self, mock_grpc_context, mocker):
        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()
        
        # mock stubs
        servicer.data_stub = mocker.MagicMock()
        pref_resp = mocker.MagicMock()
        pref_resp.found = True
        pref_resp.value = "deuteranopia"
        servicer.data_stub.GetPreference.return_value = pref_resp

        servicer.sensor_stub = mocker.MagicMock()
        frame_resp = mocker.MagicMock()
        frame_resp.data = b"\x00" * 300
        frame_resp.width = 10
        frame_resp.height = 10
        servicer.sensor_stub.CaptureFrame.return_value = frame_resp

        mocker.patch("apps.color_blind.cv2.imwrite")
        
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json=json.dumps({"query": "what color is this?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "Applied deuteranopia correction" in response.result_text

    def test_tool_name(self):
        assert ColorBlindnessServicer().tool_name == "color_blindness_assist"
