"""Tests for Color Blindness Assist tool."""

import json
import numpy as np

from apps.color_blind import ColorBlindnessServicer
from generated import data_pb2, sensor_pb2, tool_pb2


def test_color_blind_correction_math():
    """Test the Daltonization matrix math via apply_daltonization directly."""
    from apps.color_blind import apply_daltonization

    # Use a white pixel — all channels at 255.
    # With Daltonization, the values are transformed linearly, but the result
    # for a white input should remain white (all-ones row sums ≈ 1 for protanopia).
    img = np.full((2, 2, 3), 255, dtype=np.uint8)
    result = apply_daltonization(img, "protanopia")

    # Shape must be preserved
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8

    # Protanopia matrix row sums: [1.0, 1.0, 1.0] — white stays white
    np.testing.assert_array_equal(result, np.full((2, 2, 3), 255, dtype=np.uint8))


def test_daltonization_black_pixel_unchanged():
    """Black pixels produce no output regardless of correction type."""
    from apps.color_blind import apply_daltonization

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    for ctype in ("protanopia", "deuteranopia", "tritanopia"):
        result = apply_daltonization(img, ctype)
        np.testing.assert_array_equal(result, img, err_msg=f"Failed for {ctype}")


def test_daltonization_unknown_type_returns_original():
    """An unknown correction type returns the image unchanged and logs a warning."""
    from apps.color_blind import apply_daltonization

    img = np.full((2, 2, 3), 128, dtype=np.uint8)
    result = apply_daltonization(img, "not_a_real_type")
    np.testing.assert_array_equal(result, img)


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

    def test_execute_rejects_wrong_tool_name(self, mock_grpc_context, mocker):
        """The base class should reject a request with the wrong tool name."""
        import grpc

        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()

        request = tool_pb2.ToolRequest(
            tool_name="wrong_tool",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_execute_works_with_empty_params(self, mock_grpc_context, mocker):
        """Tool should work fine with empty parameters (no query needed)."""
        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()

        servicer.data_stub = mocker.MagicMock()
        pref_resp = mocker.MagicMock(found=True, value="protanopia")
        servicer.data_stub.GetPreference.return_value = pref_resp

        servicer.sensor_stub = mocker.MagicMock()
        frame_resp = mocker.MagicMock()
        frame_resp.data = b"\x00" * 300
        frame_resp.width = 10
        frame_resp.height = 10
        servicer.sensor_stub.CaptureFrame.return_value = frame_resp

        mocker.patch("apps.color_blind.cv2.imwrite", return_value=True)

        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "Applied protanopia correction" in response.result_text

    def test_execute_reports_imwrite_failure(self, mock_grpc_context, mocker):
        """If cv2.imwrite fails, execute should return failure."""
        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()

        servicer.data_stub = mocker.MagicMock()
        pref_resp = mocker.MagicMock(found=False)
        servicer.data_stub.GetPreference.return_value = pref_resp

        servicer.sensor_stub = mocker.MagicMock()
        frame_resp = mocker.MagicMock()
        frame_resp.data = b"\x00" * 300
        frame_resp.width = 10
        frame_resp.height = 10
        servicer.sensor_stub.CaptureFrame.return_value = frame_resp

        mocker.patch("apps.color_blind.cv2.imwrite", return_value=False)

        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Failed to save" in response.result_text
