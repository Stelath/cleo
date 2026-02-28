"""Regression tests for FAISS metadata sidecar recovery."""

import json

import numpy as np

from services.data.vector.faiss_db import FaissDB


def _normalized(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    arr /= np.linalg.norm(arr)
    return arr


def test_load_recovers_from_empty_metadata_sidecar(tmp_path):
    index_path = tmp_path / "clips.index"
    meta_path = index_path.with_suffix(".meta.json")

    db = FaissDB(dimension=4, index_path=str(index_path))
    query = _normalized([1.0, 0.0, 0.0, 0.0])
    db.add(query, {"clip_id": 123})
    db.save()

    meta_path.write_text("", encoding="utf-8")

    recovered = FaissDB(dimension=4, index_path=str(index_path))
    assert recovered.size == 1

    results = recovered.search(query, k=1)
    assert len(results) == 1
    assert results[0][2] == {}
    assert json.loads(meta_path.read_text(encoding="utf-8")) == [{}]


def test_load_recovers_when_metadata_length_mismatches_index(tmp_path):
    index_path = tmp_path / "clips.index"
    meta_path = index_path.with_suffix(".meta.json")

    db = FaissDB(dimension=4, index_path=str(index_path))
    db.add(_normalized([1.0, 0.0, 0.0, 0.0]), {"id": 1})
    db.add(_normalized([0.0, 1.0, 0.0, 0.0]), {"id": 2})
    db.save()

    meta_path.write_text('[{"id": 1}]', encoding="utf-8")

    recovered = FaissDB(dimension=4, index_path=str(index_path))
    assert recovered.size == 2
    assert json.loads(meta_path.read_text(encoding="utf-8")) == [{}, {}]
