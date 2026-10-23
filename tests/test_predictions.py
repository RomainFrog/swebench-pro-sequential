"""Tests for PredictionsAccumulator."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from swebench_seq.predictions import PredictionsAccumulator


def test_record_appends_to_jsonl(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    acc.record("q-1", "diff --git a/x b/x\n")
    acc.record("q-2", "diff --git a/y b/y\n")
    lines = acc.log_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["instance_id"] == "q-1"
    assert json.loads(lines[1])["patch"].startswith("diff --git a/y")


def test_flush_produces_predictions_json(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    acc.record("q-1", "patch-1")
    acc.record("q-2", "patch-2")
    final = acc.flush()
    assert final == tmp_path / "predictions.json"
    data = json.loads(final.read_text())
    assert data == {"q-1": "patch-1", "q-2": "patch-2"}


def test_flush_with_empty_log_writes_empty_object(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    final = acc.flush()
    assert json.loads(final.read_text()) == {}


def test_flush_dedupes_by_instance_id_keeping_last(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    acc.record("q-1", "first")
    acc.record("q-1", "second")  # re-run replaces
    data = json.loads(acc.flush().read_text())
    assert data == {"q-1": "second"}


def test_len_counts_recorded_predictions(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    assert len(acc) == 0
    acc.record("q-1", "p")
    acc.record("q-2", "p")
    assert len(acc) == 2


def test_concurrent_record_does_not_lose_lines(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    threads = [
        threading.Thread(target=acc.record, args=(f"q-{i}", f"patch-{i}"))
        for i in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(acc) == 50
    data = json.loads(acc.flush().read_text())
    assert len(data) == 50
    assert data["q-7"] == "patch-7"


def test_flush_skips_malformed_log_lines(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    acc.record("q-1", "p")
    with acc.log_path.open("a") as f:
        f.write("not-json\n")
    acc.record("q-2", "p2")
    data = json.loads(acc.flush().read_text())
    assert data == {"q-1": "p", "q-2": "p2"}


def test_flush_atomic_rename_overwrites_prior_final(tmp_path: Path) -> None:
    acc = PredictionsAccumulator(tmp_path)
    acc.record("q-1", "first")
    acc.flush()
    acc.record("q-2", "second")
    acc.flush()
    data = json.loads(acc.final_path.read_text())
    assert data == {"q-1": "first", "q-2": "second"}
