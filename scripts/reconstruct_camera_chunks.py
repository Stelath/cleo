"""Reconstruct and decode chunked camera frames from SensorService.

Usage:
    uv run python scripts/reconstruct_camera_chunks.py --frames 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import grpc

from generated import sensor_pb2, sensor_pb2_grpc
from services.config import SENSOR_ADDRESS
from services.media.camera_transport import CameraFrameAssembler, assembled_frame_to_rgb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct chunked camera frames")
    parser.add_argument("--sensor-address", default=SENSOR_ADDRESS)
    parser.add_argument("--mode", choices=["stream", "capture"], default="stream")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--output-dir", default="data/reconstructed_frames")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channel = grpc.insecure_channel(
        args.sensor_address,
        options=[
            ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ],
    )
    stub = sensor_pb2_grpc.SensorServiceStub(channel)

    if args.mode == "capture":
        stream = stub.CaptureFrame(sensor_pb2.CaptureRequest())
    else:
        stream = stub.StreamCamera(sensor_pb2.StreamRequest(fps=args.fps))

    assembler = CameraFrameAssembler()
    saved = 0
    for chunk in stream:
        try:
            frame = assembler.push(chunk)
        except ValueError as exc:
            print(f"chunk error: {exc}")
            continue

        if frame is None:
            continue

        rgb = assembled_frame_to_rgb(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        image_path = out_dir / f"frame_{saved:04d}.jpg"
        payload_ext = ".h264" if frame.encoding == sensor_pb2.FRAME_ENCODING_H264 else ".bin"
        payload_path = out_dir / f"frame_{saved:04d}{payload_ext}"

        cv2.imwrite(str(image_path), bgr)
        payload_path.write_bytes(frame.data)

        print(
            "saved",
            image_path,
            f"encoding={frame.encoding}",
            f"bytes={len(frame.data)}",
            f"frame_id={frame.frame_id}",
        )

        saved += 1
        if saved >= args.frames:
            break

    channel.close()


if __name__ == "__main__":
    main()
