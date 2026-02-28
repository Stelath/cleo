"""Bedrock integration test for item retrieval against video embeddings."""

from pathlib import Path

import cv2
import numpy as np

from generated import data_pb2
from services.video.service import _encode_frames_to_mp4
from tests.video.conftest import requires_bedrock

TESTDATA_DIR = Path(__file__).resolve().parent / "testdata"
ITEM_IMAGE_PATH = TESTDATA_DIR / "monster.jpeg"
VIDEO_WITH_ITEM_PATH = TESTDATA_DIR / "vid_w_monster.mp4"
VIDEO_WITHOUT_ITEM_PATH = TESTDATA_DIR / "vid_wo_monster.mp4"


def _downsample_mp4(mp4_path: Path, fps: int = 2, max_seconds: int = 10) -> bytes:
    """Read an MP4, subsample frames, and re-encode a shorter clip."""
    cap = cv2.VideoCapture(str(mp4_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(round(src_fps / fps)))
    max_frames = fps * max_seconds

    frames: list[np.ndarray] = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        idx += 1
    cap.release()

    return _encode_frames_to_mp4(frames, float(fps))


@requires_bedrock
def test_item_retreivel(video_data_client, data_stub):
    """Image query for the item should rank video-with-item above video-without-item."""
    assert ITEM_IMAGE_PATH.exists(), f"Missing required image file: {ITEM_IMAGE_PATH}"
    assert VIDEO_WITH_ITEM_PATH.exists(), f"Missing required video file: {VIDEO_WITH_ITEM_PATH}"
    assert VIDEO_WITHOUT_ITEM_PATH.exists(), f"Missing required video file: {VIDEO_WITHOUT_ITEM_PATH}"

    item_image_bytes = ITEM_IMAGE_PATH.read_bytes()
    with_item_mp4 = VIDEO_WITH_ITEM_PATH.read_bytes()
    without_item_mp4 = VIDEO_WITHOUT_ITEM_PATH.read_bytes()

    with_item_embed = _downsample_mp4(VIDEO_WITH_ITEM_PATH)
    without_item_embed = _downsample_mp4(VIDEO_WITHOUT_ITEM_PATH)

    with_item_resp = video_data_client.store_clip(
        mp4_data=with_item_mp4,
        embed_data=with_item_embed,
        start_timestamp=0.0,
        end_timestamp=15.0,
        num_frames=450,
    )
    assert with_item_resp is not None, "Failed to store video containing the item"

    without_item_resp = video_data_client.store_clip(
        mp4_data=without_item_mp4,
        embed_data=without_item_embed,
        start_timestamp=15.0,
        end_timestamp=30.0,
        num_frames=450,
    )
    assert without_item_resp is not None, "Failed to store video without the item"

    resp = data_stub.Search(
        data_pb2.SearchRequest(image_data=item_image_bytes, top_k=2),
        timeout=30.0,
    )

    assert len(resp.results) == 2
    assert resp.results[0].clip_id == with_item_resp.clip_id
    assert {result.clip_id for result in resp.results} == {
        with_item_resp.clip_id,
        without_item_resp.clip_id,
    }
