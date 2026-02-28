"""Smoke test for face detection against local image fixtures.

Exercises the full pipeline:
1. Fixture images are loaded and sent to AWS Rekognition DetectFaces.
2. Detected faces are cropped and embedded via the DataService StoreFaceEmbedding RPC.
3. Deduplication is verified: group scenes contain the same 2 people as person_a and person_b.
4. SearchFaces RPC is exercised against stored faces.

Requires valid AWS credentials with Rekognition and Bedrock access.

Usage:
    uv run python scripts/test_face_detection.py
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
import cv2
import grpc
import numpy as np

from generated import data_pb2, data_pb2_grpc

ROOT = Path(__file__).resolve().parent.parent
FACES_DIR = ROOT / "tests" / "image" / "faces"

# ── Config ──

DATA_PORT = 50053
DATA_ADDRESS = f"localhost:{DATA_PORT}"
MIN_CONFIDENCE = 90.0


@dataclass(frozen=True)
class FaceCase:
    """One test case for the face detection pipeline."""

    file_name: str
    expected_faces: int  # minimum faces expected
    summary: str


CASES = [
    FaceCase(
        file_name="person_a.jpeg",
        expected_faces=1,
        summary="Single person (A). Should detect 1 face, store as new.",
    ),
    FaceCase(
        file_name="person_b.jpeg",
        expected_faces=1,
        summary="Single person (B). Should detect 1 face, store as new.",
    ),
    FaceCase(
        file_name="group_scene.jpeg",
        expected_faces=2,
        summary="Group photo with persons A and B. Should detect 2 faces, both should dedup-match.",
    ),
    FaceCase(
        file_name="group_scene2.jpeg",
        expected_faces=2,
        summary="Second group photo with same 2 people. Should detect 2 faces, both should dedup-match.",
    ),
]


def _load_image_as_jpeg(path: Path) -> tuple[bytes, np.ndarray]:
    """Load an image file and return (jpeg_bytes, rgb_numpy_array)."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    _, jpeg = cv2.imencode(".jpg", img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return jpeg.tobytes(), img_rgb


def _crop_face_from_image(
    img_rgb: np.ndarray, bbox: dict, padding: float = 0.2
) -> bytes:
    """Crop a face region from an RGB image array and return JPEG bytes."""
    h, w = img_rgb.shape[:2]

    box_left = bbox["Left"] * w
    box_top = bbox["Top"] * h
    box_w = bbox["Width"] * w
    box_h = bbox["Height"] * h

    pad_w = box_w * padding
    pad_h = box_h * padding

    x1 = int(max(0, box_left - pad_w))
    y1 = int(max(0, box_top - pad_h))
    x2 = int(min(w, box_left + box_w + pad_w))
    y2 = int(min(h, box_top + box_h + pad_h))

    cropped = img_rgb[y1:y2, x1:x2]
    bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", bgr)
    return encoded.tobytes()


def _wait_for_grpc(address: str, timeout: float = 15.0) -> None:
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise RuntimeError(f"gRPC server at {address} not ready in {timeout}s")
    finally:
        channel.close()


def _run_data_service() -> None:
    from services.data.service import serve

    serve(port=DATA_PORT)


def main() -> int:
    # Remove stale face data so each run starts fresh
    import shutil

    for stale in ("data/cleo.db", "data/cleo.db-wal", "data/cleo.db-shm",
                   "data/vector/clips.index", "data/vector/clips.index.meta.json",
                   "data/vector/faces.index", "data/vector/faces.index.meta.json"):
        p = ROOT / stale
        if p.exists():
            p.unlink()
    faces_dir = ROOT / "data" / "faces"
    if faces_dir.is_dir():
        shutil.rmtree(faces_dir)

    # Start DataService in background
    print("Starting DataService...", flush=True)
    data_proc = multiprocessing.Process(
        target=_run_data_service, daemon=True, name="data-service"
    )
    data_proc.start()
    _wait_for_grpc(DATA_ADDRESS)
    print("DataService ready.\n", flush=True)

    # Connect clients
    rekognition = boto3.client("rekognition", region_name="us-east-1")
    data_channel = grpc.insecure_channel(DATA_ADDRESS)
    data_stub = data_pb2_grpc.DataServiceStub(data_channel)

    passes = 0
    stored_face_ids: list[int] = []

    # ── Phase 1: Store individual faces ──

    print("=" * 60)
    print("PHASE 1: Detect and store faces from person_a and person_b")
    print("=" * 60)

    for case in CASES[:2]:  # person_a, person_b only
        image_path = FACES_DIR / case.file_name
        print(f"\nImage: {case.file_name}")
        print(f"Expectation: {case.summary}")

        if not image_path.exists() or image_path.stat().st_size == 0:
            print("Result: SKIP (image missing or empty)")
            continue

        try:
            jpeg_bytes, img_rgb = _load_image_as_jpeg(image_path)

            response = rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )
            face_details = response.get("FaceDetails", [])
            high_conf = [f for f in face_details if f.get("Confidence", 0) >= MIN_CONFIDENCE]
            print(f"Rekognition: {len(face_details)} faces detected, {len(high_conf)} above {MIN_CONFIDENCE}%")

            if len(high_conf) < case.expected_faces:
                print(f"Result: FAIL (expected >= {case.expected_faces} faces, got {len(high_conf)})")
                continue

            for i, face in enumerate(high_conf):
                bbox = face["BoundingBox"]
                cropped = _crop_face_from_image(img_rgb, bbox, padding=0.2)

                resp = data_stub.StoreFaceEmbedding(
                    data_pb2.StoreFaceEmbeddingRequest(
                        image_data=cropped,
                        timestamp=time.time(),
                        confidence=face["Confidence"],
                    ),
                    timeout=30,
                )
                status = "NEW" if resp.is_new else f"MATCHED face_id={resp.matched_face_id}"
                print(f"  Face {i+1}: face_id={resp.face_id}, {status}")
                stored_face_ids.append(resp.face_id)

            passes += 1
            print("Result: PASS")

        except Exception as exc:
            print(f"Result: ERROR ({exc})")

    # ── Phase 2: Dedup via group scenes ──

    print("\n" + "=" * 60)
    print("PHASE 2: Deduplication — group scenes have same 2 people")
    print("=" * 60)

    for case in CASES[2:]:  # group_scene, group_scene2
        image_path = FACES_DIR / case.file_name
        print(f"\nImage: {case.file_name}")
        print(f"Expectation: {case.summary}")

        if not image_path.exists() or image_path.stat().st_size == 0:
            print("Result: SKIP (image missing or empty)")
            continue

        try:
            jpeg_bytes, img_rgb = _load_image_as_jpeg(image_path)

            response = rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )
            face_details = response.get("FaceDetails", [])
            high_conf = [f for f in face_details if f.get("Confidence", 0) >= MIN_CONFIDENCE]
            print(f"Rekognition: {len(face_details)} faces detected, {len(high_conf)} above {MIN_CONFIDENCE}%")

            new_count = 0
            matched_count = 0
            for i, face in enumerate(high_conf):
                bbox = face["BoundingBox"]
                cropped = _crop_face_from_image(img_rgb, bbox, padding=0.2)

                resp = data_stub.StoreFaceEmbedding(
                    data_pb2.StoreFaceEmbeddingRequest(
                        image_data=cropped,
                        timestamp=time.time(),
                        confidence=face["Confidence"],
                    ),
                    timeout=30,
                )
                if resp.is_new:
                    new_count += 1
                    status = "NEW (unexpected — should have matched)"
                else:
                    matched_count += 1
                    status = f"MATCHED face_id={resp.matched_face_id}"
                print(f"  Face {i+1}: face_id={resp.face_id}, {status}")

            if matched_count > 0:
                print(f"Result: PASS ({matched_count} matched, {new_count} new)")
                passes += 1
            else:
                print(f"Result: FAIL (no faces matched — all {new_count} stored as new)")

        except Exception as exc:
            print(f"Result: ERROR ({exc})")

    # ── Phase 3: Search ──

    print("\n" + "=" * 60)
    print("PHASE 3: SearchFaces — query with person_b image")
    print("=" * 60)

    person_b_path = FACES_DIR / "person_b.jpeg"
    if person_b_path.exists() and person_b_path.stat().st_size > 0:
        try:
            jpeg_bytes, img_rgb = _load_image_as_jpeg(person_b_path)
            response = rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )
            face_details = response.get("FaceDetails", [])
            high_conf = [f for f in face_details if f.get("Confidence", 0) >= MIN_CONFIDENCE]

            if high_conf:
                bbox = high_conf[0]["BoundingBox"]
                cropped = _crop_face_from_image(img_rgb, bbox, padding=0.2)
                search_resp = data_stub.SearchFaces(
                    data_pb2.SearchFacesRequest(image_data=cropped, top_k=5),
                    timeout=30,
                )
                print(f"Search returned {len(search_resp.results)} result(s)")
                for r in search_resp.results:
                    print(
                        f"  face_id={r.face_id}  score={r.score:.3f}  "
                        f"seen_count={r.seen_count}  thumbnail={r.thumbnail_path}"
                    )
                if search_resp.results:
                    print("Result: PASS")
                    passes += 1
                else:
                    print("Result: FAIL (no search results)")
            else:
                print("Result: SKIP (no faces detected in person_b.jpeg)")
        except Exception as exc:
            print(f"Result: ERROR ({exc})")
    else:
        print("Result: SKIP (person_b.jpeg missing or empty)")

    # ── Summary ──

    total_possible = 5  # 2 individuals + 2 group dedup + 1 search
    print(f"\n{'=' * 60}")
    print(f"Overall: {passes}/{total_possible} phases passed")
    print(f"Stored face_ids: {stored_face_ids}")

    data_channel.close()
    data_proc.terminate()
    data_proc.join(timeout=3)

    return 0 if passes >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
