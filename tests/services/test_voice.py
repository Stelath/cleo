"""Tests for services.frontend.voice module."""

import base64
import struct

import numpy as np
import pytest

from services.frontend.voice import ElevenLabsVoice, _pcm16_to_f32_base64


# ── _pcm16_to_f32_base64 ──


def test_pcm16_to_f32_base64_empty():
    result = _pcm16_to_f32_base64(b"")
    decoded = base64.b64decode(result)
    assert decoded == b""


def test_pcm16_to_f32_base64_max_int16():
    """Max int16 (32767) should map to ~1.0 float32."""
    pcm16 = struct.pack("<h", 32767)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    (f32_val,) = struct.unpack("<f", decoded)
    assert abs(f32_val - 1.0) < 1e-4


def test_pcm16_to_f32_base64_min_int16():
    """Min int16 (-32768) should map close to -1.0 float32."""
    pcm16 = struct.pack("<h", -32768)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    (f32_val,) = struct.unpack("<f", decoded)
    assert abs(f32_val - (-32768 / 32767.0)) < 1e-4


def test_pcm16_to_f32_base64_silence():
    """Zero samples should stay zero."""
    pcm16 = struct.pack("<hh", 0, 0)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    np.testing.assert_array_equal(f32_arr, [0.0, 0.0])


def test_pcm16_to_f32_base64_roundtrip_length():
    """Output should have 2x as many bytes (int16=2B -> float32=4B per sample)."""
    num_samples = 100
    pcm16 = np.random.randint(-32768, 32767, size=num_samples, dtype=np.int16).tobytes()
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    assert len(decoded) == num_samples * 4  # float32 = 4 bytes


# ── ElevenLabsVoice ──


def test_voice_lazy_init_no_api_key(monkeypatch):
    """Client should not be created until synthesize is called."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    voice = ElevenLabsVoice()
    assert voice._client is None


def test_voice_empty_text_raises():
    voice = ElevenLabsVoice()
    with pytest.raises(ValueError, match="non-empty"):
        voice.synthesize("")

    with pytest.raises(ValueError, match="non-empty"):
        voice.synthesize("   ")


def test_voice_synthesize_mocked(monkeypatch):
    """End-to-end with a mocked ElevenLabs client."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

    # Create fake PCM16 audio: 10 samples of silence
    fake_pcm16 = np.zeros(10, dtype=np.int16).tobytes()

    class FakeConvert:
        def convert(self, **kwargs):
            assert kwargs["voice_id"] == "MClEFoImJXBTgLwdLI5n"
            assert kwargs["text"] == "Hello world"
            return iter([fake_pcm16])

    class FakeClient:
        text_to_speech = FakeConvert()

    voice = ElevenLabsVoice()
    voice._client = FakeClient()

    data_b64, sample_rate = voice.synthesize("Hello world")
    assert sample_rate == 22050

    decoded = base64.b64decode(data_b64)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    assert len(f32_arr) == 10
    np.testing.assert_array_equal(f32_arr, np.zeros(10, dtype=np.float32))
