"""Face embedding & clustering test with 3 photos each of 2 people.

Exercises the full pipeline:
1. Detect faces via AWS Rekognition DetectFaces.
2. Crop and embed each face via DataService StoreFaceEmbedding RPC.
3. Print a full similarity matrix (every face vs every face).
4. Show whether the embeddings cluster by person (dedup analysis).

Images (in tests/image/faces/):
  Person A: IMG_8709.jpeg, IMG_8710.jpeg, IMG_8711.jpeg
  Person B: IMG_8712.jpeg, IMG_8713.jpeg, IMG_8714.jpeg

Requires valid AWS credentials with Rekognition and Bedrock access.

Usage:
    uv run python scripts/test_face_detection.py
"""

from __future__ import annotations

import multiprocessing
import shutil
import sys
import time
from dataclasses import dataclass, field
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

PERSON_A_IMAGES = ["IMG_8709.jpeg", "IMG_8710.jpeg", "IMG_8711.jpeg"]
PERSON_B_IMAGES = ["IMG_8712.jpeg", "IMG_8713.jpeg", "IMG_8714.jpeg"]
ALL_IMAGES = PERSON_A_IMAGES + PERSON_B_IMAGES


@dataclass
class FaceResult:
    """Result of processing a single image."""

    image: str
    person: str
    face_id: int
    faiss_id: int
    is_new: bool
    matched_face_id: int | None
    confidence: float
    bbox_area: float  # fraction of frame


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
    # Remove stale data so each run starts fresh
    for stale in (
        "data/cleo.db", "data/cleo.db-wal", "data/cleo.db-shm",
        "data/vector/clips.index", "data/vector/clips.index.meta.json",
        "data/vector/faces.index", "data/vector/faces.meta.json",
    ):
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

    results: list[FaceResult] = []

    # ── Phase 1: Detect and store all faces ──

    print("=" * 70)
    print("PHASE 1: Detect and store faces from all 6 images")
    print("=" * 70)

    for image_name in ALL_IMAGES:
        person = "A" if image_name in PERSON_A_IMAGES else "B"
        image_path = FACES_DIR / image_name
        print(f"\n  [{person}] {image_name}")

        if not image_path.exists() or image_path.stat().st_size == 0:
            print(f"       SKIP (missing or empty)")
            continue

        try:
            jpeg_bytes, img_rgb = _load_image_as_jpeg(image_path)

            response = rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )
            face_details = response.get("FaceDetails", [])
            high_conf = [
                f for f in face_details if f.get("Confidence", 0) >= MIN_CONFIDENCE
            ]
            print(
                f"       Rekognition: {len(face_details)} detected, "
                f"{len(high_conf)} >= {MIN_CONFIDENCE}% confidence"
            )

            if not high_conf:
                print(f"       NO FACES above threshold")
                continue

            # Use the largest face (highest bbox area) — skip background faces
            best_face = max(
                high_conf,
                key=lambda f: f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"],
            )
            bbox = best_face["BoundingBox"]
            bbox_area = bbox["Width"] * bbox["Height"]

            cropped = _crop_face_from_image(img_rgb, bbox, padding=0.2)

            resp = data_stub.StoreFaceEmbedding(
                data_pb2.StoreFaceEmbeddingRequest(
                    image_data=cropped,
                    timestamp=time.time(),
                    confidence=best_face["Confidence"],
                ),
                timeout=30,
            )

            status = "NEW" if resp.is_new else f"MATCHED face_id={resp.matched_face_id}"
            print(
                f"       face_id={resp.face_id}  {status}  "
                f"bbox_area={bbox_area:.4f}  conf={best_face['Confidence']:.1f}%"
            )

            results.append(
                FaceResult(
                    image=image_name,
                    person=person,
                    face_id=resp.face_id,
                    faiss_id=resp.faiss_id,
                    is_new=resp.is_new,
                    matched_face_id=resp.matched_face_id if not resp.is_new else None,
                    confidence=best_face["Confidence"],
                    bbox_area=bbox_area,
                )
            )

        except Exception as exc:
            print(f"       ERROR: {exc}")

    # ── Phase 2: Cross-search similarity matrix ──

    print("\n" + "=" * 70)
    print("PHASE 2: Similarity matrix (SearchFaces for each image vs all stored)")
    print("=" * 70)

    # Re-embed each image and search against the full index
    similarity_scores: dict[tuple[str, str], float] = {}

    for image_name in ALL_IMAGES:
        image_path = FACES_DIR / image_name
        if not image_path.exists():
            continue

        try:
            jpeg_bytes, img_rgb = _load_image_as_jpeg(image_path)
            response = rekognition.detect_faces(
                Image={"Bytes": jpeg_bytes}, Attributes=["DEFAULT"]
            )
            high_conf = [
                f
                for f in response.get("FaceDetails", [])
                if f.get("Confidence", 0) >= MIN_CONFIDENCE
            ]
            if not high_conf:
                continue

            best_face = max(
                high_conf,
                key=lambda f: f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"],
            )
            cropped = _crop_face_from_image(img_rgb, best_face["BoundingBox"], padding=0.2)

            search_resp = data_stub.SearchFaces(
                data_pb2.SearchFacesRequest(image_data=cropped, top_k=10),
                timeout=30,
            )

            for r in search_resp.results:
                # Find which image this face_id came from
                for res in results:
                    if res.face_id == r.face_id:
                        similarity_scores[(image_name, res.image)] = r.score
                        break

        except Exception as exc:
            print(f"  Error searching {image_name}: {exc}")

    # Print similarity matrix
    print(f"\n{'':>18}", end="")
    for img in ALL_IMAGES:
        print(f"  {img[:10]:>10}", end="")
    print()

    for img_a in ALL_IMAGES:
        person_a = "A" if img_a in PERSON_A_IMAGES else "B"
        print(f"  [{person_a}] {img_a[:10]:>10}", end="")
        for img_b in ALL_IMAGES:
            score = similarity_scores.get((img_a, img_b))
            if score is not None:
                print(f"  {score:>10.4f}", end="")
            else:
                print(f"  {'---':>10}", end="")
        print()

    # ── Phase 3: Clustering analysis ──

    print("\n" + "=" * 70)
    print("PHASE 3: Clustering analysis")
    print("=" * 70)

    # Group by assigned face_id
    clusters: dict[int, list[FaceResult]] = {}
    for r in results:
        # The effective cluster ID is the matched_face_id if deduped, else its own face_id
        cluster_id = r.matched_face_id if r.matched_face_id else r.face_id
        clusters.setdefault(cluster_id, []).append(r)

    print(f"\n  Total images processed: {len(results)}")
    print(f"  Unique face IDs (clusters): {len(clusters)}")
    print(f"  Expected clusters: 2 (Person A and Person B)")

    for cluster_id, members in sorted(clusters.items()):
        persons = set(m.person for m in members)
        images = [m.image for m in members]
        pure = len(persons) == 1
        print(f"\n  Cluster face_id={cluster_id}:")
        print(f"    Members: {', '.join(images)}")
        print(f"    Persons: {', '.join(sorted(persons))}")
        print(f"    Pure cluster: {'YES' if pure else 'NO — MIXED'}")

    # Compute intra-person vs inter-person similarity stats
    intra_scores = []
    inter_scores = []
    for (img_a, img_b), score in similarity_scores.items():
        if img_a == img_b:
            continue
        person_a = "A" if img_a in PERSON_A_IMAGES else "B"
        person_b = "A" if img_b in PERSON_A_IMAGES else "B"
        if person_a == person_b:
            intra_scores.append(score)
        else:
            inter_scores.append(score)

    print(f"\n  Similarity statistics:")
    if intra_scores:
        print(
            f"    Same person (intra):   "
            f"min={min(intra_scores):.4f}  "
            f"max={max(intra_scores):.4f}  "
            f"mean={np.mean(intra_scores):.4f}  "
            f"n={len(intra_scores)}"
        )
    else:
        print(f"    Same person (intra):   no data")

    if inter_scores:
        print(
            f"    Diff person (inter):   "
            f"min={min(inter_scores):.4f}  "
            f"max={max(inter_scores):.4f}  "
            f"mean={np.mean(inter_scores):.4f}  "
            f"n={len(inter_scores)}"
        )
    else:
        print(f"    Diff person (inter):   no data")

    if intra_scores and inter_scores:
        gap = min(intra_scores) - max(inter_scores)
        print(f"    Separation gap:        {gap:.4f} ", end="")
        if gap > 0:
            print("(GOOD — no overlap, clean threshold possible)")
        else:
            print("(BAD — overlap, threshold cannot cleanly separate)")

        ideal_threshold = (np.mean(intra_scores) + np.mean(inter_scores)) / 2
        print(f"    Suggested threshold:   {ideal_threshold:.4f}")
        from services.config import FACE_SIMILARITY_THRESHOLD
        print(f"    Current threshold:     {FACE_SIMILARITY_THRESHOLD} (FACE_SIMILARITY_THRESHOLD)")

    # ── Summary ──

    print(f"\n{'=' * 70}")
    dedup_correct = len(clusters) == 2 and all(
        len(set(m.person for m in members)) == 1 for members in clusters.values()
    )
    if dedup_correct:
        print("RESULT: PASS — 2 clean clusters, one per person")
    else:
        print(
            f"RESULT: FAIL — got {len(clusters)} clusters, expected 2. "
            f"Dedup is not working correctly with current embeddings."
        )

    data_channel.close()
    data_proc.terminate()
    data_proc.join(timeout=3)

    return 0 if dedup_correct else 1


if __name__ == "__main__":
    sys.exit(main())
