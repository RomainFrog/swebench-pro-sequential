"""Atomic predictions.json writer.

SWE-bench Pro's evaluator expects a single JSON file mapping `instance_id` →
patch string. We accumulate results as tasks finish (often in parallel), so:

- Use a JSONL append-log on the side as the source of truth (one line per
  finished task — never partial; safe under concurrent writers via O_APPEND).
- Build the final `predictions.json` from the JSONL via `flush()` (atomic
  rename: write to `.tmp`, fsync, rename).

This keeps in-flight runs crash-resilient: if the process dies, the JSONL has
everything that finished, and `flush()` can be re-run later.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOG_NAME = "predictions.jsonl"
_FINAL_NAME = "predictions.json"


class PredictionsAccumulator:
    """Thread-safe accumulator for per-task patch outcomes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = output_dir / _LOG_NAME
        self._final_path = output_dir / _FINAL_NAME
        self._lock = threading.Lock()

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def final_path(self) -> Path:
        return self._final_path

    def record(self, instance_id: str, patch: str) -> None:
        """Append one prediction to the JSONL log. Safe under concurrent calls.

        We rely on POSIX O_APPEND atomicity for line-sized writes (well under
        PIPE_BUF). The mutex protects against threads in this process tripping
        over each other on platforms with weaker guarantees.
        """
        line = json.dumps(
            {"instance_id": instance_id, "patch": patch}, ensure_ascii=False
        )
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def flush(self) -> Path:
        """Build final predictions.json from the JSONL. Atomic rename.

        Later JSONL entries for the same `instance_id` overwrite earlier ones —
        re-runs of the same task replace the prediction.
        """
        merged: dict[str, str] = {}
        if self._log_path.exists():
            with self._log_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    merged[entry["instance_id"]] = entry["patch"]

        tmp_path = self._final_path.with_suffix(self._final_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self._final_path)
        return self._final_path

    def __len__(self) -> int:
        if not self._log_path.exists():
            return 0
        with self._log_path.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
