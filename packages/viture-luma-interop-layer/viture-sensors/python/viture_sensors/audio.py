from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
import pyaudio


class AudioRecorder:
    """Thin PyAudio wrapper for microphone recording and playback."""

    def __init__(
        self,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        frames_per_buffer: int = 1024,
    ) -> None:
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.frames_per_buffer = frames_per_buffer
        self._audio: Optional[pyaudio.PyAudio] = pyaudio.PyAudio()

    def close(self) -> None:
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

    def record(
        self,
        duration_ms: int,
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> NDArray[np.float32]:
        if self._audio is None:
            raise RuntimeError("AudioRecorder is closed")
        if duration_ms <= 0:
            return np.zeros((0,), dtype=np.float32)

        total_frames = max(1, int((sample_rate * duration_ms) / 1000))
        try:
            stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.frames_per_buffer,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to open input audio stream: {exc}") from exc

        chunks: list[bytes] = []
        remaining = total_frames
        try:
            while remaining > 0:
                to_read = min(self.frames_per_buffer, remaining)
                chunks.append(stream.read(to_read, exception_on_overflow=False))
                remaining -= to_read
        finally:
            stream.stop_stream()
            stream.close()

        data = np.frombuffer(b"".join(chunks), dtype=np.float32)
        if channels > 1 and data.size > 0:
            usable = (data.size // channels) * channels
            data = data[:usable].reshape(-1, channels).mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32)

    def play(
        self,
        samples: NDArray[np.float32],
        sample_rate: int = 48000,
        channels: int = 1,
    ) -> None:
        if self._audio is None:
            raise RuntimeError("AudioRecorder is closed")

        pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
        pcm = np.clip(pcm, -1.0, 1.0)
        if channels > 1:
            pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)

        try:
            stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.frames_per_buffer,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to open output audio stream: {exc}") from exc

        try:
            stream.write(pcm.tobytes())
        finally:
            stream.stop_stream()
            stream.close()

    @staticmethod
    def list_input_devices() -> list[dict[str, object]]:
        audio = pyaudio.PyAudio()
        devices: list[dict[str, object]] = []
        try:
            for idx in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(idx)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append({"index": idx, "name": str(info.get("name", f"Device {idx}"))})
        finally:
            audio.terminate()
        return devices

    def __enter__(self) -> "AudioRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
