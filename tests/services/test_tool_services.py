"""Tests for tool services — base class and individual tool implementations."""

import json
from unittest.mock import MagicMock

import pytest

from generated import tool_pb2
from apps.color_blind import ColorBlindnessServicer
from apps.notetaking import NoteSummaryBedrockClient, NotetakingServicer
from apps.tool_base import ToolServiceBase

class DummyTool(ToolServiceBase):
    @property
    def tool_name(self) -> str:
        return "dummy_tool"

    @property
    def tool_description(self) -> str:
        return "Dummy description"

    @property
    def tool_input_schema(self) -> dict:
        return {"type": "object"}

    def execute(self, params: dict) -> tuple[bool, str]:
        return True, "mocked executed"


class TestToolServiceBase:
    def test_wrong_tool_name_rejected(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="wrong_name",
            parameters_json=json.dumps({"query": "test"}),
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        mock_grpc_context.set_code.assert_called()

    def test_invalid_json_rejected(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="dummy_tool",
            parameters_json="{bad json",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert not response.success
        assert "Invalid parameters" in response.result_text

    def test_empty_params_json_is_ok(self, mock_grpc_context):
        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="dummy_tool",
            parameters_json="",
        )
        response = servicer.Execute(request, mock_grpc_context)
        assert response.success

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ToolServiceBase()

    def test_execute_exception_caught(self, mock_grpc_context):
        """If execute() raises, base class catches it and returns an error response."""

        class BrokenTool(ToolServiceBase):
            @property
            def tool_name(self) -> str:
                return "broken"

            @property
            def tool_description(self) -> str:
                return "A broken tool"

            @property
            def tool_input_schema(self) -> dict:
                return {"type": "object"}

            def execute(self, params: dict) -> tuple[bool, str]:
                raise RuntimeError("something went wrong")

        servicer = BrokenTool()
        request = tool_pb2.ToolRequest(tool_name="broken", parameters_json="{}")
        response = servicer.Execute(request, mock_grpc_context)

        assert not response.success
        assert "something went wrong" in response.result_text
        mock_grpc_context.set_code.assert_called()

    def test_wrong_tool_sets_invalid_argument_status(self, mock_grpc_context):
        import grpc

        servicer = DummyTool()
        request = tool_pb2.ToolRequest(
            tool_name="wrong",
            parameters_json="{}",
        )
        servicer.Execute(request, mock_grpc_context)
        mock_grpc_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


class TestToolServiceProperties:
    """Verify tool_description, tool_input_schema, and tool_type on all servicers."""

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_description_is_nonempty(self, servicer_cls):
        servicer = servicer_cls()
        assert isinstance(servicer.tool_description, str)
        assert len(servicer.tool_description) > 0

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_input_schema_has_type(self, servicer_cls):
        servicer = servicer_cls()
        schema = servicer.tool_input_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema

    @pytest.mark.parametrize(
        "servicer_cls",
        [ColorBlindnessServicer],
    )
    def test_tool_type_defaults_to_on_demand(self, servicer_cls):
        servicer = servicer_cls()
        assert servicer.tool_type == "on_demand"
class TestNotetakingServicer:
    def test_start_and_stop_flow(self):
        import threading

        mock_data = MagicMock()
        mock_data.GetTranscriptionsInRange.return_value = MagicMock(entries=[])
        mock_data.GetVideoClipsInRange.return_value = MagicMock(clips=[])
        mock_data.StoreNoteSummary.return_value = MagicMock(id=1)
        mock_bedrock = MagicMock()

        servicer = NotetakingServicer.__new__(NotetakingServicer)
        servicer._bedrock = mock_bedrock
        servicer._data = mock_data
        servicer._channel = MagicMock()
        servicer._frontend = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._lock = threading.Lock()
        servicer._session_start = None

        start_response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "start"}),
            ),
            MagicMock(),
        )
        assert start_response.success
        assert "Started notetaking" in start_response.result_text

        stop_response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "stop"}),
            ),
            MagicMock(),
        )
        assert stop_response.success
        assert "No transcript or video activity" in stop_response.result_text
        mock_data.StoreNoteSummary.assert_called_once()
        assert servicer._frontend.ShowNotification.call_count == 2
        start_notification = servicer._frontend.ShowNotification.call_args_list[0].args[0]
        saved_notification = servicer._frontend.ShowNotification.call_args_list[1].args[0]
        assert start_notification.title == "Notetaking started"
        assert start_notification.message == "Capturing notes until you stop the session."
        assert start_notification.style == "info"
        assert saved_notification.title == "Note saved"
        assert saved_notification.message == "Saved note summary #1"
        assert saved_notification.style == "success"

    def test_stop_requires_active_session(self, mock_grpc_context):
        import threading

        servicer = NotetakingServicer.__new__(NotetakingServicer)
        servicer._bedrock = MagicMock()
        servicer._data = MagicMock()
        servicer._channel = MagicMock()
        servicer._frontend = MagicMock()
        servicer._frontend_channel = MagicMock()
        servicer._lock = threading.Lock()
        servicer._session_start = None

        response = servicer.Execute(
            tool_pb2.ToolRequest(
                tool_name="notetaking",
                parameters_json=json.dumps({"action": "stop"}),
            ),
            mock_grpc_context,
        )
        assert not response.success
        assert response.result_text == "Notetaking is not active"

    def test_tool_name(self):
        servicer = NotetakingServicer.__new__(NotetakingServicer)
        assert servicer.tool_name == "notetaking"


class TestNoteSummaryBedrockClient:
    def test_uses_sampled_keyframes_as_images(self):
        metadata = MagicMock(
            clip_id=7,
            start_timestamp=1.25,
            end_timestamp=2.5,
            num_frames=12,
        )
        transcript = MagicMock(start_time=1.0, end_time=2.0, text="Discussed action items")
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Fallback summary"},
                    ]
                }
            }
        }

        client = NoteSummaryBedrockClient(client=mock_client)
        client._extract_keyframes = MagicMock(return_value=[b"frame-1", b"frame-2"])
        summary = client.summarize(
            start_timestamp=1.0,
            end_timestamp=3.0,
            transcripts=[transcript],
            clips=[(metadata, b"fake-mp4")],
        )

        assert summary == "Fallback summary"
        assert mock_client.converse.call_count == 1
        content = mock_client.converse.call_args.kwargs["messages"][0]["content"]
        assert all("video" not in block for block in content)
        image_blocks = [block for block in content if "image" in block]
        assert len(image_blocks) == 2
        assert image_blocks[0]["image"]["source"]["bytes"] == b"frame-1"
        assert image_blocks[1]["image"]["source"]["bytes"] == b"frame-2"

    def test_sample_frame_indices_are_evenly_spaced(self):
        assert NoteSummaryBedrockClient._sample_frame_indices(10, 4) == [0, 3, 6, 9]
        assert NoteSummaryBedrockClient._sample_frame_indices(3, 16) == [0, 1, 2]

    def test_notetaking_does_not_fall_back_to_raw_transcript(self):
        servicer = NotetakingServicer.__new__(NotetakingServicer)
        servicer._bedrock = MagicMock()
        servicer._bedrock.summarize.return_value = ""

        context = MagicMock(
            transcripts=[MagicMock(text="This should not be stored verbatim")],
            clips=[],
        )

        summary = servicer._summarize_context(1.0, 2.0, context)

        assert summary == "A note summary could not be generated for this session."


class TestFaceDetectionServicerProperties:
    def _make_servicer(self):
        from apps.face_detection import FaceDetectionServicer

        servicer = FaceDetectionServicer.__new__(FaceDetectionServicer)
        return servicer

    def test_tool_name(self):
        servicer = self._make_servicer()
        assert servicer.tool_name == "face_detection"

    def test_tool_type_is_active(self):
        servicer = self._make_servicer()
        assert servicer.tool_type == "active"

    def test_tool_description_is_nonempty(self):
        servicer = self._make_servicer()
        assert isinstance(servicer.tool_description, str)
        assert len(servicer.tool_description) > 0

    def test_tool_input_schema_has_type(self):
        servicer = self._make_servicer()
        schema = servicer.tool_input_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema
