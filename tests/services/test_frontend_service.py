"""Tests for services.frontend.service — SpeakText RPC."""

from unittest.mock import MagicMock

import grpc

from generated import frontend_pb2
from services.frontend.service import FrontendServiceServicer


def _make_context():
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


def test_speak_text_empty_text_returns_error():
    servicer = FrontendServiceServicer()
    ctx = _make_context()
    request = frontend_pb2.SpeakTextRequest(text="")
    resp = servicer.SpeakText(request, ctx)

    assert resp.ok is False
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)


def test_speak_text_whitespace_only_returns_error():
    servicer = FrontendServiceServicer()
    ctx = _make_context()
    request = frontend_pb2.SpeakTextRequest(text="   ")
    resp = servicer.SpeakText(request, ctx)

    assert resp.ok is False
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)


def test_speak_text_with_mocked_voice():
    servicer = FrontendServiceServicer()
    ctx = _make_context()

    servicer._voice = MagicMock()
    servicer._voice.synthesize.return_value = ("AAAA", 22050)

    request = frontend_pb2.SpeakTextRequest(text="Hello")
    resp = servicer.SpeakText(request, ctx)

    assert resp.ok is True
    servicer._voice.synthesize.assert_called_once_with("Hello")


def test_speak_text_synthesis_failure():
    servicer = FrontendServiceServicer()
    ctx = _make_context()

    servicer._voice = MagicMock()
    servicer._voice.synthesize.side_effect = RuntimeError("API down")

    request = frontend_pb2.SpeakTextRequest(text="Hello")
    resp = servicer.SpeakText(request, ctx)

    assert resp.ok is False
    ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
