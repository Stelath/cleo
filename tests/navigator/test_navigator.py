"""Tests for apps.navigator — continuous visual guide for blind/VI users."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from generated import sensor_pb2

from apps.navigator import (
    NavigatorBedrockClient,
    NavigatorLoop,
    NavigatorServicer,
    _rgb_to_jpeg,
)


# ── _rgb_to_jpeg ──


class TestRgbToJpeg:
    def test_basic_conversion(self):
        height, width = 48, 64
        frame_rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        jpeg = _rgb_to_jpeg(frame_rgb)
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_quality_affects_size(self):
        height, width = 48, 64
        frame_rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        low_q = _rgb_to_jpeg(frame_rgb, quality=10)
        high_q = _rgb_to_jpeg(frame_rgb, quality=95)
        assert len(low_q) < len(high_q)


# ── NavigatorBedrockClient ──


def _mock_bedrock_response(text: str) -> dict:
    return {
        "output": {
            "message": {
                "content": [{"text": text}],
            }
        }
    }


class TestNavigatorBedrockClient:
    def test_analyze_frame_extracts_text(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _mock_bedrock_response("Curb ahead at 12 o'clock.")
        client = NavigatorBedrockClient(client=mock_client)

        result = client.analyze_frame(b"fake-jpeg", "help me find curbs")
        assert result == "Curb ahead at 12 o'clock."

    def test_analyze_frame_sends_image_block(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _mock_bedrock_response("All clear.")
        client = NavigatorBedrockClient(client=mock_client)

        jpeg_bytes = b"\xff\xd8fake-jpeg-data"
        client.analyze_frame(jpeg_bytes, "navigate")

        call_kwargs = mock_client.converse.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        image_blocks = [b for b in content if "image" in b]
        assert len(image_blocks) == 1
        assert image_blocks[0]["image"]["source"]["bytes"] == jpeg_bytes

    def test_analyze_frame_includes_previous_guidance(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _mock_bedrock_response("Curb cleared.")
        client = NavigatorBedrockClient(client=mock_client)

        client.analyze_frame(b"jpeg", "navigate", previous_guidance="Curb ahead.")

        call_kwargs = mock_client.converse.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        content = messages[0]["content"]
        text_blocks = [b for b in content if "text" in b]
        assert any("Curb ahead." in b["text"] for b in text_blocks)

    def test_analyze_frame_sends_system_prompt(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _mock_bedrock_response("Clear path.")
        client = NavigatorBedrockClient(client=mock_client)

        client.analyze_frame(b"jpeg", "navigate")

        call_kwargs = mock_client.converse.call_args
        system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        system_text = system[0]["text"]
        assert "blind" in system_text.lower() or "visually impaired" in system_text.lower()


# ── NavigatorLoop ──


def _make_fake_frame(width: int = 64, height: int = 48) -> MagicMock:
    frame = MagicMock()
    frame.data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()
    frame.width = width
    frame.height = height
    frame.timestamp = 1000.0
    frame.encoding = sensor_pb2.FRAME_ENCODING_RGB24
    return frame


class TestNavigatorLoop:
    def test_stop_event_terminates_thread(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        loop = NavigatorLoop(
            user_context="test",
            sensor_address="localhost:99999",
            bedrock_client=mock_bedrock,
        )
        loop.start()
        loop.stop()
        loop.join(timeout=5.0)
        assert not loop.is_alive()

    def test_process_frame_calls_vlm(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.return_value = "Obstacle at 2 o'clock."
        loop = NavigatorLoop(
            user_context="find obstacles",
            bedrock_client=mock_bedrock,
        )

        frame = _make_fake_frame()
        loop._process_frame(frame)

        mock_bedrock.analyze_frame.assert_called_once()
        call_args = mock_bedrock.analyze_frame.call_args
        # First arg is jpeg bytes
        assert call_args[0][0][:2] == b"\xff\xd8"

    def test_guidance_buffer_populated(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.return_value = "Step down ahead."
        loop = NavigatorLoop(
            user_context="curbs",
            bedrock_client=mock_bedrock,
        )

        frame = _make_fake_frame()
        loop._process_frame(frame)

        assert len(loop.recent_guidance) == 1
        assert loop.recent_guidance[0][1] == "Step down ahead."

    def test_previous_guidance_carried_forward(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.side_effect = ["Curb ahead.", "Curb passed."]
        loop = NavigatorLoop(
            user_context="curbs",
            bedrock_client=mock_bedrock,
        )

        loop._process_frame(_make_fake_frame())
        loop._process_frame(_make_fake_frame())

        # Second call should have received previous guidance
        second_call = mock_bedrock.analyze_frame.call_args_list[1]
        assert second_call[0][2] == "Curb ahead."  # previous_guidance arg

    def test_vlm_error_does_not_crash(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.side_effect = RuntimeError("VLM timeout")
        loop = NavigatorLoop(
            user_context="test",
            bedrock_client=mock_bedrock,
        )

        frame = _make_fake_frame()
        # Should not raise
        loop._process_frame(frame)
        assert len(loop.recent_guidance) == 0


    def test_process_frame_calls_speak_text(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.return_value = "Curb at 12 o'clock, two steps ahead."
        loop = NavigatorLoop(
            user_context="find curbs",
            bedrock_client=mock_bedrock,
        )

        frame = _make_fake_frame()
        with patch.object(loop, "_speak_guidance") as mock_speak:
            loop._process_frame(frame)
            mock_speak.assert_called_once_with("Curb at 12 o'clock, two steps ahead.")

    def test_speak_guidance_error_does_not_crash(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        mock_bedrock.analyze_frame.return_value = "All clear."
        loop = NavigatorLoop(
            user_context="test",
            bedrock_client=mock_bedrock,
        )

        with patch("apps.navigator.grpc.insecure_channel", side_effect=RuntimeError("no frontend")):
            frame = _make_fake_frame()
            # Should not raise even when TTS fails
            loop._process_frame(frame)
            assert loop.recent_guidance[0][1] == "All clear."


# ── NavigatorServicer ──


class TestNavigatorServicer:
    def test_tool_name(self):
        servicer = NavigatorServicer()
        assert servicer.tool_name == "navigator"

    def test_tool_type_is_active(self):
        servicer = NavigatorServicer()
        assert servicer.tool_type == "active"

    def test_start_and_stop_flow(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        servicer = NavigatorServicer(
            sensor_address="localhost:99999",
            bedrock_client=mock_bedrock,
        )

        success, text = servicer.execute({"action": "start", "query": "find curbs"})
        assert success is True
        assert "started" in text.lower()

        success, text = servicer.execute({"action": "stop"})
        assert success is True
        assert "stopped" in text.lower()

    def test_stop_without_start_fails(self):
        servicer = NavigatorServicer()
        success, text = servicer.execute({"action": "stop"})
        assert success is False
        assert "not active" in text.lower()

    def test_double_start_returns_already_active(self):
        mock_bedrock = MagicMock(spec=NavigatorBedrockClient)
        servicer = NavigatorServicer(
            sensor_address="localhost:99999",
            bedrock_client=mock_bedrock,
        )

        servicer.execute({"action": "start", "query": "obstacles"})
        success, text = servicer.execute({"action": "start", "query": "obstacles"})
        assert success is True
        assert "already" in text.lower()

        # Cleanup
        servicer.execute({"action": "stop"})

    def test_resolve_action_natural_language(self):
        servicer = NavigatorServicer()
        assert servicer._resolve_action({"query": "help me navigate"}) == "start"
        assert servicer._resolve_action({"query": "guide me around"}) == "start"
        assert servicer._resolve_action({"query": "stop navigating"}) == "stop"
        assert servicer._resolve_action({"query": "cancel guidance"}) == "stop"
        assert servicer._resolve_action({"query": "hello"}) is None

    def test_tool_description_nonempty(self):
        servicer = NavigatorServicer()
        assert len(servicer.tool_description) > 0

    def test_tool_input_schema_has_type_object(self):
        servicer = NavigatorServicer()
        assert servicer.tool_input_schema["type"] == "object"
