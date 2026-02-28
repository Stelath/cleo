"""ElevenLabs TTS integration for Cleo frontend service."""

import base64
import os

import numpy as np
import structlog

log = structlog.get_logger()

VOICE_ID = "MClEFoImJXBTgLwdLI5n"  # Ivy
MODEL_ID = "eleven_turbo_v2_5"
OUTPUT_FORMAT = "pcm_22050"
ELEVENLABS_SAMPLE_RATE = 22050
PLAYBACK_SAMPLE_RATE = 48000

VOICE_SETTINGS = {
    "speed": 1.03,
    "stability": 0.42,
    "similarity_boost": 0.75,
    "style": 0.40,
}


def _resample(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio via linear interpolation."""
    if orig_rate == target_rate:
        return samples
    target_len = int(len(samples) * target_rate / orig_rate)
    indices = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def _pcm16_to_f32_base64(pcm16_bytes: bytes) -> str:
    """Convert signed 16-bit LE PCM to float32 LE, resample to playback rate, then base64-encode.

    This matches the Tauri client's ``play_pcm_base64`` expectation.
    """
    if not pcm16_bytes:
        return base64.b64encode(b"").decode("ascii")
    samples_i16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
    samples_f32 = samples_i16.astype(np.float32) / 32767.0
    samples_f32 = _resample(samples_f32, ELEVENLABS_SAMPLE_RATE, PLAYBACK_SAMPLE_RATE)
    return base64.b64encode(samples_f32.tobytes()).decode("ascii")


class ElevenLabsVoice:
    """Lazy-initialized ElevenLabs TTS client."""

    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY environment variable is not set"
            )
        from elevenlabs import ElevenLabs

        self._client = ElevenLabs(api_key=api_key)
        log.info("elevenlabs.client_initialized")

    def synthesize(self, text: str) -> tuple[str, int]:
        """Synthesize *text* and return ``(data_base64, sample_rate)``.

        Raises ``ValueError`` for empty text.
        """
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        self._ensure_client()

        log.info("elevenlabs.synthesize", text_len=len(text))
        audio_iter = self._client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT,
            text=text,
            voice_settings=VOICE_SETTINGS,
        )

        pcm_chunks: list[bytes] = []
        for chunk in audio_iter:
            if chunk:
                pcm_chunks.append(chunk)
        pcm16_bytes = b"".join(pcm_chunks)

        data_base64 = _pcm16_to_f32_base64(pcm16_bytes)
        log.info("elevenlabs.synthesized", pcm_bytes=len(pcm16_bytes))
        return data_base64, PLAYBACK_SAMPLE_RATE
