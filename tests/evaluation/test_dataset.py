"""Tests for pinned public dataset loading and JSONL caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swebench_seq.evaluation import dataset


def test_load_rows_pins_public_revision(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def fake_load_dataset(name: str, **kwargs):
        calls.append({"name": name, **kwargs})
        return [{"instance_id": "iid", "dockerhub_tag": "tag"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    dataset.load_swebench_pro_rows.cache_clear()
    try:
        assert dataset.load_swebench_pro_rows() == ({"instance_id": "iid", "dockerhub_tag": "tag"},)
    finally:
        dataset.load_swebench_pro_rows.cache_clear()

    assert calls == [
        {
            "name": dataset.DATASET_NAME,
            "revision": dataset.DATASET_REVISION,
            "split": dataset.DATASET_SPLIT,
        }
    ]


def test_materialize_dataset_jsonl_uses_revision_specific_cache(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    rows = (
        {"instance_id": "first", "value": "café"},
        {"instance_id": "second", "value": 2},
    )
    monkeypatch.setattr(dataset, "load_swebench_pro_rows", lambda: rows)
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    monkeypatch.setattr(
        dataset,
        "DATASET_JSONL_SHA256",
        hashlib.sha256(rendered.encode()).hexdigest(),
    )

    path = dataset.materialize_dataset_jsonl()

    assert path == (
        tmp_path / "swebench-seq" / "datasets" / dataset.DATASET_REVISION / "test.jsonl"
    )
    assert [json.loads(line) for line in path.read_text().splitlines()] == list(rows)

    monkeypatch.setattr(
        dataset,
        "load_swebench_pro_rows",
        lambda: (_ for _ in ()).throw(AssertionError("cache should be reused")),
    )
    assert dataset.materialize_dataset_jsonl() == path


def test_materialize_dataset_rejects_corrupted_cached_file(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = dataset.default_dataset_jsonl_path()
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")

    with pytest.raises(RuntimeError, match="checksum verification"):
        dataset.materialize_dataset_jsonl()


def test_materialize_dataset_jsonl_preserves_explicit_override(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "custom.jsonl"
    explicit.write_text("{}\n")
    monkeypatch.setattr(
        dataset,
        "load_swebench_pro_rows",
        lambda: (_ for _ in ()).throw(AssertionError("HF should not be loaded")),
    )

    assert dataset.materialize_dataset_jsonl(explicit) == explicit.resolve()


def test_load_rows_from_explicit_jsonl_without_hugging_face(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.jsonl"
    explicit.write_text(
        json.dumps({"instance_id": "first", "dockerhub_tag": "tag-1"})
        + "\n"
        + json.dumps({"instance_id": "second", "dockerhub_tag": "tag-2"})
        + "\n"
    )
    dataset.load_swebench_pro_rows.cache_clear()
    try:
        rows = dataset.load_swebench_pro_rows(explicit)
    finally:
        dataset.load_swebench_pro_rows.cache_clear()

    assert [row["instance_id"] for row in rows] == ["first", "second"]


def test_load_rows_from_explicit_csv(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.csv"
    explicit.write_text("instance_id,dockerhub_tag\nfirst,tag-1\n")
    dataset.load_swebench_pro_rows.cache_clear()
    try:
        rows = dataset.load_swebench_pro_rows(explicit)
    finally:
        dataset.load_swebench_pro_rows.cache_clear()

    assert rows == ({"instance_id": "first", "dockerhub_tag": "tag-1"},)
