"""Tests for services.frontend.voice module."""

import base64
import struct

import numpy as np
import pytest

from services.frontend.voice import (
    ELEVENLABS_SAMPLE_RATE,
    PLAYBACK_SAMPLE_RATE,
    ElevenLabsVoice,
    _pcm16_to_f32_base64,
    _resample,
)


# ── _pcm16_to_f32_base64 ──


def test_pcm16_to_f32_base64_empty():
    result = _pcm16_to_f32_base64(b"")
    decoded = base64.b64decode(result)
    assert decoded == b""


def test_pcm16_to_f32_base64_max_int16():
    """Max int16 (32767) should map to ~1.0 float32 after resampling."""
    pcm16 = struct.pack("<h", 32767)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    # All resampled values should be ~1.0 (constant signal stays constant)
    assert np.allclose(f32_arr, 1.0, atol=1e-4)


def test_pcm16_to_f32_base64_min_int16():
    """Min int16 (-32768) should map close to -1.0 float32 after resampling."""
    pcm16 = struct.pack("<h", -32768)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    expected = -32768 / 32767.0
    assert np.allclose(f32_arr, expected, atol=1e-4)


def test_pcm16_to_f32_base64_silence():
    """Zero samples should stay zero after resampling."""
    pcm16 = struct.pack("<hh", 0, 0)
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    np.testing.assert_array_equal(f32_arr, np.zeros_like(f32_arr))


def test_pcm16_to_f32_base64_resampled_length():
    """Output sample count should reflect resampling from ELEVENLABS to PLAYBACK rate."""
    num_samples = 44100  # 1 second at ElevenLabs rate
    pcm16 = np.zeros(num_samples, dtype=np.int16).tobytes()
    result = _pcm16_to_f32_base64(pcm16)
    decoded = base64.b64decode(result)
    expected_samples = int(num_samples * PLAYBACK_SAMPLE_RATE / ELEVENLABS_SAMPLE_RATE)
    actual_samples = len(decoded) // 4
    assert actual_samples == expected_samples


# ── _resample ──


def test_resample_same_rate():
    """No-op when rates match."""
    samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _resample(samples, 48000, 48000)
    np.testing.assert_array_equal(result, samples)


def test_resample_upsample_length():
    """Upsampling from 44100 to 48000 produces correct number of samples."""
    n = 44100
    samples = np.ones(n, dtype=np.float32)
    result = _resample(samples, 44100, 48000)
    assert len(result) == 48000


def test_resample_preserves_constant_signal():
    """A constant signal should remain constant after resampling."""
    samples = np.full(1000, 0.5, dtype=np.float32)
    result = _resample(samples, 22050, 48000)
    np.testing.assert_allclose(result, 0.5, atol=1e-6)


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
    assert sample_rate == PLAYBACK_SAMPLE_RATE

    decoded = base64.b64decode(data_b64)
    f32_arr = np.frombuffer(decoded, dtype=np.float32)
    # 10 input samples resampled from ELEVENLABS_SAMPLE_RATE to PLAYBACK_SAMPLE_RATE
    expected_len = int(10 * PLAYBACK_SAMPLE_RATE / ELEVENLABS_SAMPLE_RATE)
    assert len(f32_arr) == expected_len
    # Silence stays silent after resampling
    np.testing.assert_array_equal(f32_arr, np.zeros(expected_len, dtype=np.float32))
