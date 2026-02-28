"""Tests for Color Blindness Assist tool."""

import json
import numpy as np

from apps.color_blind import ColorBlindnessServicer
from generated import data_pb2, sensor_pb2, tool_pb2
from services.config import SENSOR_ADDRESS


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
    def test_sensor_channel_allows_large_frames(self, mocker):
        mock_insecure_channel = mocker.patch("apps.color_blind.grpc.insecure_channel")

        ColorBlindnessServicer()

        sensor_calls = [
            call for call in mock_insecure_channel.call_args_list if call.args[0] == SENSOR_ADDRESS
        ]
        assert len(sensor_calls) == 1
        options = dict(sensor_calls[0].kwargs["options"])
        assert options["grpc.max_receive_message_length"] == 32 * 1024 * 1024
        assert options["grpc.max_send_message_length"] == 32 * 1024 * 1024

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
        frame_chunk = mocker.MagicMock()
        frame_chunk.data = b"\x00" * 300
        frame_chunk.frame_id = "frame-1"
        frame_chunk.chunk_index = 0
        frame_chunk.is_last = True
        frame_chunk.width = 10
        frame_chunk.height = 10
        frame_chunk.encoding = sensor_pb2.FRAME_ENCODING_RGB24
        servicer.sensor_stub.CaptureFrame.return_value = iter([frame_chunk])

        servicer.frontend_stub = mocker.MagicMock()
        mocker.patch("apps.color_blind.cv2.imwrite", return_value=True)
        mocker.patch(
            "apps.color_blind.cv2.imencode",
            return_value=(True, np.frombuffer(b"jpeg-bytes", dtype=np.uint8)),
        )
        
        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json=json.dumps({"query": "what color is this?"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "Applied deuteranopia correction" in response.result_text
        servicer.frontend_stub.StreamImage.assert_called_once()

        # Verify duration_ms=8000 is set on every ImageChunk
        chunk_iter = servicer.frontend_stub.StreamImage.call_args[0][0]
        chunks = list(chunk_iter)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.duration_ms == 8000

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
        frame_chunk = mocker.MagicMock()
        frame_chunk.data = b"\x00" * 300
        frame_chunk.frame_id = "frame-2"
        frame_chunk.chunk_index = 0
        frame_chunk.is_last = True
        frame_chunk.width = 10
        frame_chunk.height = 10
        frame_chunk.encoding = sensor_pb2.FRAME_ENCODING_RGB24
        servicer.sensor_stub.CaptureFrame.return_value = iter([frame_chunk])

        servicer.frontend_stub = mocker.MagicMock()
        mocker.patch("apps.color_blind.cv2.imwrite", return_value=True)
        mocker.patch(
            "apps.color_blind.cv2.imencode",
            return_value=(True, np.frombuffer(b"jpeg-bytes", dtype=np.uint8)),
        )

        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success
        assert "Applied protanopia correction" in response.result_text
        servicer.frontend_stub.StreamImage.assert_called_once()

        # Verify duration_ms=8000 is set on every ImageChunk
        chunk_iter = servicer.frontend_stub.StreamImage.call_args[0][0]
        chunks = list(chunk_iter)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.duration_ms == 8000

    def test_execute_reports_imwrite_failure(self, mock_grpc_context, mocker):
        """If cv2.imwrite fails, execute should return failure."""
        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()

        servicer.data_stub = mocker.MagicMock()
        pref_resp = mocker.MagicMock(found=False)
        servicer.data_stub.GetPreference.return_value = pref_resp

        servicer.sensor_stub = mocker.MagicMock()
        frame_chunk = mocker.MagicMock()
        frame_chunk.data = b"\x00" * 300
        frame_chunk.frame_id = "frame-3"
        frame_chunk.chunk_index = 0
        frame_chunk.is_last = True
        frame_chunk.width = 10
        frame_chunk.height = 10
        frame_chunk.encoding = sensor_pb2.FRAME_ENCODING_RGB24
        servicer.sensor_stub.CaptureFrame.return_value = iter([frame_chunk])

        servicer.frontend_stub = mocker.MagicMock()
        mocker.patch("apps.color_blind.cv2.imwrite", return_value=False)

        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Failed to save" in response.result_text
        servicer.frontend_stub.StreamImage.assert_not_called()

    def test_execute_reports_imencode_failure(self, mock_grpc_context, mocker):
        """If cv2.imencode fails, execute should return failure."""
        mocker.patch("apps.color_blind.grpc.insecure_channel")
        servicer = ColorBlindnessServicer()

        servicer.data_stub = mocker.MagicMock()
        pref_resp = mocker.MagicMock(found=False)
        servicer.data_stub.GetPreference.return_value = pref_resp

        servicer.sensor_stub = mocker.MagicMock()
        frame_chunk = mocker.MagicMock()
        frame_chunk.data = b"\x00" * 300
        frame_chunk.frame_id = "frame-4"
        frame_chunk.chunk_index = 0
        frame_chunk.is_last = True
        frame_chunk.width = 10
        frame_chunk.height = 10
        frame_chunk.encoding = sensor_pb2.FRAME_ENCODING_RGB24
        servicer.sensor_stub.CaptureFrame.return_value = iter([frame_chunk])

        servicer.frontend_stub = mocker.MagicMock()
        mocker.patch("apps.color_blind.cv2.imwrite", return_value=True)
        mocker.patch(
            "apps.color_blind.cv2.imencode",
            return_value=(False, None),
        )

        request = tool_pb2.ToolRequest(
            tool_name="color_blindness_assist",
            parameters_json="{}",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Failed to encode" in response.result_text
        servicer.frontend_stub.StreamImage.assert_not_called()
