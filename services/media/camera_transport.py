"""Helpers for camera chunk transport, assembly, and codec conversion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import queue
import select
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from generated import sensor_pb2
from services.config import SENSOR_DEFAULT_FPS

_FFMPEG_TIMEOUT_SECONDS = 12.0


@dataclass(frozen=True)
class AssembledCameraFrame:
    frame_id: str
    data: bytes
    width: int
    height: int
    timestamp: float
    encoding: int
    key_frame: bool


class CameraFrameAssembler:
    """Assemble a stream of CameraFrameChunk messages into full frame payloads."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._frame_id: str | None = None
        self._encoding: int = sensor_pb2.FRAME_ENCODING_UNSPECIFIED
        self._key_frame: bool = False
        self._width = 0
        self._height = 0
        self._timestamp = 0.0
        self._expected_chunk_index = 0
        self._parts: list[bytes] = []

    def push(self, chunk: sensor_pb2.CameraFrameChunk) -> AssembledCameraFrame | None:
        if self._frame_id is None:
            self._frame_id = chunk.frame_id
            self._encoding = chunk.encoding
            self._key_frame = bool(chunk.key_frame)
            self._width = int(chunk.width)
            self._height = int(chunk.height)
            self._timestamp = float(chunk.timestamp)
            self._expected_chunk_index = 0
            self._parts = []

        if chunk.frame_id != self._frame_id:
            got = chunk.frame_id
            expected = self._frame_id
            self.reset()
            raise ValueError(
                f"camera chunk frame_id changed mid-frame: expected={expected}, got={got}"
            )

        if chunk.chunk_index != self._expected_chunk_index:
            expected = self._expected_chunk_index
            got = chunk.chunk_index
            self.reset()
            raise ValueError(f"camera chunk gap: expected={expected}, got={got}")

        if chunk.encoding != self._encoding:
            expected = self._encoding
            got = chunk.encoding
            self.reset()
            raise ValueError(
                f"camera chunk encoding changed mid-frame: expected={expected}, got={got}"
            )

        self._parts.append(chunk.data)
        self._expected_chunk_index += 1

        if not chunk.is_last:
            return None

        if self._frame_id is None:
            self.reset()
            raise ValueError("camera assembler internal error: missing frame_id")

        frame = AssembledCameraFrame(
            frame_id=self._frame_id,
            data=b"".join(self._parts),
            width=self._width,
            height=self._height,
            timestamp=self._timestamp,
            encoding=self._encoding,
            key_frame=self._key_frame,
        )
        self.reset()
        return frame


class PersistentH264Encoder:
    """Long-lived ffmpeg encoder that avoids per-frame process spawning."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 28,
        preset: str = "ultrafast",
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._fps = max(1.0, float(fps))
        self._crf = int(crf)
        self._preset = preset
        self._lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._closed = False

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self._width}x{self._height}",
            "-r",
            f"{self._fps:.3f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self._preset,
            "-tune",
            "zerolatency",
            "-crf",
            str(self._crf),
            "-g",
            "1",
            "-keyint_min",
            "1",
            "-bf",
            "0",
            "-x264-params",
            "scenecut=0:repeat-headers=1:aud=1",
            "-flush_packets",
            "1",
            "-f",
            "h264",
            "pipe:1",
        ]

        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self.close()
            raise RuntimeError("Failed to initialize ffmpeg encoder pipes")

        self._stdout_fd = self._proc.stdout.fileno()
        os.set_blocking(self._stdout_fd, False)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="PersistentH264EncoderStderr",
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        if stderr is None:
            return
        while True:
            line = stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="ignore").strip()
            if text:
                self._stderr_tail.append(text)

    def _stderr_text(self) -> str:
        if not self._stderr_tail:
            return ""
        return " | ".join(self._stderr_tail)

    def _ensure_running(self) -> None:
        rc = self._proc.poll()
        if rc is not None:
            details = self._stderr_text()
            raise RuntimeError(
                f"Persistent H.264 encoder exited rc={rc}{(': ' + details) if details else ''}"
            )

    def encode_frame(self, frame_rgb: np.ndarray, timeout: float = 2.0) -> bytes:
        if self._closed:
            raise RuntimeError("Persistent H.264 encoder is closed")
        if frame_rgb.shape[:2] != (self._height, self._width):
            raise ValueError(
                "Frame shape does not match encoder dimensions: "
                f"expected {(self._height, self._width)}, got {frame_rgb.shape[:2]}"
            )

        frame_bytes = np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes()
        with self._lock:
            self._ensure_running()
            stdin = self._proc.stdin
            if stdin is None:
                raise RuntimeError("Encoder stdin is unavailable")
            stdin.write(frame_bytes)
            stdin.flush()
            return self._read_one_encoded_frame(timeout=timeout)

    def _read_one_encoded_frame(self, timeout: float) -> bytes:
        payload = bytearray()
        deadline = time.monotonic() + max(0.01, float(timeout))
        last_read_ts = 0.0
        idle_grace_seconds = 0.008

        while time.monotonic() < deadline:
            self._ensure_running()
            remaining = max(0.0, deadline - time.monotonic())
            wait_s = min(0.05, remaining)
            readable, _, _ = select.select([self._stdout_fd], [], [], wait_s)
            if readable:
                try:
                    chunk = os.read(self._stdout_fd, 65536)
                except BlockingIOError:
                    chunk = b""

                if chunk:
                    payload.extend(chunk)
                    last_read_ts = time.monotonic()
                    continue

                if payload:
                    return bytes(payload)
                raise RuntimeError("Encoder stdout closed unexpectedly")

            if payload and last_read_ts > 0.0:
                if time.monotonic() - last_read_ts >= idle_grace_seconds:
                    return bytes(payload)

        details = self._stderr_text()
        raise RuntimeError(
            "Timed out waiting for encoded H.264 frame bytes"
            + (f": {details}" if details else "")
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)

    def __enter__(self) -> "PersistentH264Encoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PersistentH264Decoder:
    """Long-lived ffmpeg decoder that avoids per-frame process spawning."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._frame_bytes = self._width * self._height * 3
        self._lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._frame_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=16)
        self._primed = False
        self._closed = False

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]

        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._proc.stdin is None or self._proc.stdout is None or self._proc.stderr is None:
            self.close()
            raise RuntimeError("Failed to initialize ffmpeg decoder pipes")

        self._stdout_thread = threading.Thread(
            target=self._drain_stdout,
            daemon=True,
            name="PersistentH264DecoderStdout",
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="PersistentH264DecoderStderr",
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stderr = self._proc.stderr
        if stderr is None:
            return
        while True:
            line = stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="ignore").strip()
            if text:
                self._stderr_tail.append(text)

    def _stderr_text(self) -> str:
        if not self._stderr_tail:
            return ""
        return " | ".join(self._stderr_tail)

    @staticmethod
    def _read_exact(stream, size: int) -> bytes | None:
        payload = bytearray()
        while len(payload) < size:
            chunk = stream.read(size - len(payload))
            if not chunk:
                return None
            payload.extend(chunk)
        return bytes(payload)

    def _drain_stdout(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            return
        while True:
            frame_bytes = self._read_exact(stdout, self._frame_bytes)
            if frame_bytes is None:
                try:
                    self._frame_queue.put_nowait(None)
                except queue.Full:
                    pass
                return
            self._frame_queue.put(frame_bytes)

    def _ensure_running(self) -> None:
        rc = self._proc.poll()
        if rc is not None:
            details = self._stderr_text()
            raise RuntimeError(
                f"Persistent H.264 decoder exited rc={rc}{(': ' + details) if details else ''}"
            )

    def _read_decoded_frame(self, timeout: float) -> bytes:
        deadline = time.monotonic() + max(0.01, float(timeout))
        while time.monotonic() < deadline:
            self._ensure_running()
            remaining = max(0.0, deadline - time.monotonic())
            wait_s = min(0.2, remaining)
            try:
                item = self._frame_queue.get(timeout=wait_s)
            except queue.Empty:
                continue

            if item is None:
                self._ensure_running()
                raise RuntimeError("Persistent H.264 decoder ended unexpectedly")

            if len(item) != self._frame_bytes:
                raise RuntimeError(
                    f"Decoded H.264 frame size mismatch: expected {self._frame_bytes}, got {len(item)}"
                )

            return item

        details = self._stderr_text()
        raise RuntimeError(
            "Timed out waiting for decoded H.264 frame bytes"
            + (f": {details}" if details else "")
        )

    def _decode_payload(self, payload: bytes) -> np.ndarray:
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(
            self._height,
            self._width,
            3,
        )
        return frame.copy()

    def decode_frame(self, frame_h264: bytes, timeout: float = 2.0) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Persistent H.264 decoder is closed")

        with self._lock:
            self._ensure_running()
            stdin = self._proc.stdin
            if stdin is None:
                raise RuntimeError("Decoder stdin is unavailable")

            attempts = 1 if self._primed else 5
            per_attempt_timeout = max(0.5, float(timeout) / attempts)
            payload: bytes | None = None
            for attempt in range(attempts):
                stdin.write(frame_h264)
                stdin.flush()
                try:
                    payload = self._read_decoded_frame(per_attempt_timeout)
                    break
                except RuntimeError:
                    if attempt >= attempts - 1:
                        raise

            if payload is None:
                raise RuntimeError("Persistent H.264 decoder produced no frame")
            self._primed = True

        return self._decode_payload(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass

        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)

        self._stdout_thread.join(timeout=0.5)
        self._stderr_thread.join(timeout=0.5)

    def __enter__(self) -> "PersistentH264Decoder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_camera_frame_chunks(
    *,
    data: bytes,
    frame_id: str,
    width: int,
    height: int,
    timestamp: float,
    encoding: int,
    key_frame: bool,
    chunk_bytes: int,
) -> Iterator[sensor_pb2.CameraFrameChunk]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    total_len = len(data)
    if total_len == 0:
        yield sensor_pb2.CameraFrameChunk(
            data=b"",
            frame_id=frame_id,
            chunk_index=0,
            is_last=True,
            width=width,
            height=height,
            timestamp=timestamp,
            encoding=encoding,
            key_frame=key_frame,
        )
        return

    chunk_index = 0
    for start in range(0, total_len, chunk_bytes):
        end = min(start + chunk_bytes, total_len)
        yield sensor_pb2.CameraFrameChunk(
            data=data[start:end],
            frame_id=frame_id,
            chunk_index=chunk_index,
            is_last=end >= total_len,
            width=width,
            height=height,
            timestamp=timestamp,
            encoding=encoding,
            key_frame=key_frame,
        )
        chunk_index += 1


def encode_rgb_to_jpeg(frame_rgb: np.ndarray, quality: int = 85) -> bytes:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB frame as JPEG")
    return encoded.tobytes()


def decode_jpeg_to_rgb(jpeg_data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_rgb_to_h264_annexb(
    frame_rgb: np.ndarray,
    *,
    fps: float,
    crf: int = 28,
    preset: str = "ultrafast",
) -> bytes:
    height, width = frame_rgb.shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{max(1.0, float(fps)):.3f}",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-tune",
        "zerolatency",
        "-crf",
        str(int(crf)),
        "-x264-params",
        "keyint=1:min-keyint=1:scenecut=0:repeat-headers=1:aud=1",
        "-f",
        "h264",
        "pipe:1",
    ]

    proc = subprocess.run(
        command,
        input=frame_rgb.tobytes(),
        capture_output=True,
        check=False,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg H.264 encode failed: {stderr or proc.returncode}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg H.264 encode produced empty payload")
    return bytes(proc.stdout)


def decode_h264_annexb_to_rgb(
    frame_h264: bytes,
    width: int,
    height: int,
    decoder: PersistentH264Decoder | None = None,
) -> np.ndarray:
    if decoder is not None:
        return decoder.decode_frame(frame_h264, timeout=_FFMPEG_TIMEOUT_SECONDS)

    return _decode_h264_annexb_to_rgb_subprocess(frame_h264, width, height)


def _decode_h264_annexb_to_rgb_subprocess(
    frame_h264: bytes,
    width: int,
    height: int,
) -> np.ndarray:

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    proc = subprocess.run(
        command,
        input=frame_h264,
        capture_output=True,
        check=False,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg H.264 decode failed: {stderr or proc.returncode}")

    expected = int(width) * int(height) * 3
    if len(proc.stdout) < expected:
        raise RuntimeError(
            f"Decoded H.264 frame size mismatch: expected at least {expected} bytes, got {len(proc.stdout)}"
        )

    frame = np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(height, width, 3)
    return frame.copy()


def assembled_frame_to_rgb(
    frame: AssembledCameraFrame,
    h264_decoder: PersistentH264Decoder | None = None,
) -> np.ndarray:
    if frame.encoding == sensor_pb2.FRAME_ENCODING_JPEG:
        return decode_jpeg_to_rgb(frame.data)
    if frame.encoding == sensor_pb2.FRAME_ENCODING_H264:
        return decode_h264_annexb_to_rgb(
            frame.data,
            frame.width,
            frame.height,
            decoder=h264_decoder,
        )
    if frame.encoding == sensor_pb2.FRAME_ENCODING_RGB24:
        expected = frame.width * frame.height * 3
        if len(frame.data) != expected:
            raise RuntimeError(
                f"RGB24 frame size mismatch: expected {expected}, got {len(frame.data)}"
            )
        rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        return rgb.copy()
    raise RuntimeError(f"Unsupported frame encoding: {frame.encoding}")


def h264_frames_to_mp4(frame_payloads: Iterable[bytes], fps: float) -> bytes:
    payloads = list(frame_payloads)
    if not payloads:
        return b""

    with tempfile.TemporaryDirectory(prefix="cleo_h264_") as tmp_dir:
        tmp = Path(tmp_dir)
        src_h264 = tmp / "clip.h264"
        src_h264.write_bytes(b"".join(payloads))

        out_mp4 = tmp / "clip.mp4"
        target_fps = max(1.0, float(fps))
        # Sensor-side stream encoding is produced with SENSOR_DEFAULT_FPS timing.
        # When frames are dropped before this stage, we need to scale timestamps
        # so playback duration still matches wall-clock capture time.
        pts_scale = max(0.001, float(SENSOR_DEFAULT_FPS) / target_fps)

        reencode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "h264",
            "-i",
            str(src_h264),
            "-an",
            "-vf",
            f"setpts={pts_scale:.6f}*PTS",
            "-r",
            f"{target_fps:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
        proc = subprocess.run(
            reencode_cmd,
            capture_output=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )

        if proc.returncode != 0 or not out_mp4.exists() or out_mp4.stat().st_size == 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg H.264->MP4 failed: {stderr or proc.returncode}")

        return out_mp4.read_bytes()


def downsample_mp4_for_embedding(mp4_data: bytes, target_fps: float) -> bytes:
    if not mp4_data:
        return b""

    with tempfile.TemporaryDirectory(prefix="cleo_embed_") as tmp_dir:
        tmp = Path(tmp_dir)
        src = tmp / "input.mp4"
        dst = tmp / "embed.mp4"
        src.write_bytes(mp4_data)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-an",
            "-vf",
            f"fps={max(0.1, float(target_fps)):.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ]
        proc = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg embed downsample failed: {stderr or proc.returncode}")
        return dst.read_bytes()
