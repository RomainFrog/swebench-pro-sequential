"""Pinned access to the public SWE-bench Pro dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

DATASET_NAME = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DATASET_SPLIT = "test"
DATASET_JSONL_SHA256 = "7edd8237eadd866b2fd0f8f3f16f63d29622cabb94e914e28b82b9872bbb07bd"


@cache
def load_swebench_pro_rows(dataset_path: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Load rows from an explicit local file or the pinned public test split."""

    if dataset_path is not None:
        return _load_local_rows(dataset_path.expanduser().resolve())

    from datasets import load_dataset  # type: ignore[import-untyped]

    dataset = load_dataset(
        DATASET_NAME,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
    )
    return tuple(dict(row) for row in dataset)


def _load_local_rows(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            rows.append(row)
    return tuple(rows)


def default_dataset_jsonl_path() -> Path:
    """Return the revision-specific cache path used by file-based consumers."""

    configured = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return cache_root / "swebench-seq" / "datasets" / DATASET_REVISION / f"{DATASET_SPLIT}.jsonl"


def materialize_dataset_jsonl(dataset_path: Path | None = None) -> Path:
    """Return an explicit dataset file or cache the pinned HF split as JSONL."""

    if dataset_path is not None:
        explicit = dataset_path.expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"dataset file not found: {explicit}")
        return explicit

    target = default_dataset_jsonl_path()
    try:
        if target.is_file() and target.stat().st_size > 0:
            if _sha256(target) != DATASET_JSONL_SHA256:
                raise RuntimeError(
                    f"cached SWE-bench Pro dataset failed checksum verification: {target}"
                )
            return target
    except OSError:
        pass

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            try:
                rows = load_swebench_pro_rows()
            except Exception as exc:
                raise RuntimeError(f"failed to load {DATASET_NAME} at {DATASET_REVISION}") from exc
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        if _sha256(temp_path) != DATASET_JSONL_SHA256:
            raise RuntimeError(
                "materialized SWE-bench Pro dataset failed checksum verification"
            )
        temp_path.replace(target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
