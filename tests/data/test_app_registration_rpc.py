"""Tests for app registration gRPC handlers in DataServiceServicer."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from generated import data_pb2


@pytest.fixture
def data_servicer(tmp_path):
    """Create a DataServiceServicer with temp paths and mocked embeddings."""
    with patch("services.data.service.embed_video") as mock_ev, \
         patch("services.data.service.embed_text") as mock_et, \
         patch("services.data.service.embed_image") as mock_ei:

        def _fake_embed(*args, **kwargs):
            v = np.random.randn(1024).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        mock_ev.side_effect = _fake_embed
        mock_et.side_effect = _fake_embed
        mock_ei.side_effect = _fake_embed

        from services.data.service import DataServiceServicer

        servicer = DataServiceServicer(
            db_path=str(tmp_path / "test.db"),
            index_path=str(tmp_path / "test.index"),
            video_dir=str(tmp_path / "videos"),
        )
        yield servicer
        servicer.shutdown()


def _mock_context():
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx


class TestRegisterAppRPC:
    def test_register_new_app(self, data_servicer):
        req = data_pb2.RegisterAppRequest(
            name="my_tool",
            description="Does things",
            app_type="on_demand",
            grpc_address="localhost:50060",
            input_schema_json='{"type": "object"}',
        )
        resp = data_servicer.RegisterApp(req, _mock_context())
        assert resp.id >= 1
        assert resp.created is True

    def test_register_upsert(self, data_servicer):
        req = data_pb2.RegisterAppRequest(
            name="my_tool",
            description="v1",
            app_type="on_demand",
            grpc_address="localhost:50060",
            input_schema_json='{"type": "object"}',
        )
        resp1 = data_servicer.RegisterApp(req, _mock_context())
        assert resp1.created is True

        req.description = "v2"
        resp2 = data_servicer.RegisterApp(req, _mock_context())
        assert resp2.created is False
        assert resp2.id == resp1.id

    def test_register_invalid_schema(self, data_servicer):
        import grpc

        req = data_pb2.RegisterAppRequest(
            name="bad_tool",
            description="Bad schema",
            app_type="on_demand",
            grpc_address="localhost:50060",
            input_schema_json="{not valid json",
        )
        ctx = _mock_context()
        data_servicer.RegisterApp(req, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_register_empty_schema_defaults(self, data_servicer):
        """Empty input_schema_json should default to '{}'."""
        req = data_pb2.RegisterAppRequest(
            name="minimal_tool",
            description="Minimal",
            app_type="on_demand",
            grpc_address="localhost:50060",
            input_schema_json="",
        )
        resp = data_servicer.RegisterApp(req, _mock_context())
        assert resp.created is True


class TestListAppsRPC:
    def test_list_registered_apps(self, data_servicer):
        for name in ["tool_a", "tool_b"]:
            req = data_pb2.RegisterAppRequest(
                name=name,
                description=f"{name} desc",
                app_type="on_demand",
                grpc_address=f"localhost:5006{name[-1]}",
                input_schema_json='{}',
            )
            data_servicer.RegisterApp(req, _mock_context())

        list_req = data_pb2.ListAppsRequest()
        resp = data_servicer.ListApps(list_req, _mock_context())
        assert len(resp.apps) == 2
        names = {a.name for a in resp.apps}
        assert names == {"tool_a", "tool_b"}

    def test_list_enabled_only(self, data_servicer):
        for name in ["tool_a", "tool_b"]:
            req = data_pb2.RegisterAppRequest(
                name=name, description="", app_type="on_demand",
                grpc_address="localhost:50060", input_schema_json="{}",
            )
            data_servicer.RegisterApp(req, _mock_context())

        data_servicer.SetAppEnabled(
            data_pb2.SetAppEnabledRequest(name="tool_b", enabled=False),
            _mock_context(),
        )

        resp = data_servicer.ListApps(
            data_pb2.ListAppsRequest(enabled_only=True), _mock_context()
        )
        assert len(resp.apps) == 1
        assert resp.apps[0].name == "tool_a"

    def test_list_by_type(self, data_servicer):
        data_servicer.RegisterApp(
            data_pb2.RegisterAppRequest(
                name="on_demand_tool", description="", app_type="on_demand",
                grpc_address="localhost:50060", input_schema_json="{}",
            ),
            _mock_context(),
        )
        data_servicer.RegisterApp(
            data_pb2.RegisterAppRequest(
                name="polling_tool", description="", app_type="polling",
                grpc_address="localhost:50061", input_schema_json="{}",
            ),
            _mock_context(),
        )

        resp = data_servicer.ListApps(
            data_pb2.ListAppsRequest(app_type="polling"), _mock_context()
        )
        assert len(resp.apps) == 1
        assert resp.apps[0].name == "polling_tool"


class TestSetAppEnabledRPC:
    def test_toggle_enabled(self, data_servicer):
        data_servicer.RegisterApp(
            data_pb2.RegisterAppRequest(
                name="my_tool", description="", app_type="on_demand",
                grpc_address="localhost:50060", input_schema_json="{}",
            ),
            _mock_context(),
        )

        resp = data_servicer.SetAppEnabled(
            data_pb2.SetAppEnabledRequest(name="my_tool", enabled=False),
            _mock_context(),
        )
        assert resp.success is True

        # Verify it's disabled
        list_resp = data_servicer.ListApps(
            data_pb2.ListAppsRequest(enabled_only=True), _mock_context()
        )
        assert len(list_resp.apps) == 0

    def test_nonexistent_app(self, data_servicer):
        import grpc

        ctx = _mock_context()
        resp = data_servicer.SetAppEnabled(
            data_pb2.SetAppEnabledRequest(name="no_such_app", enabled=False),
            ctx,
        )
        assert resp.success is False
        ctx.set_code.assert_called_with(grpc.StatusCode.NOT_FOUND)
