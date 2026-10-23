"""Single source of truth for cross-module constants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Hatch installs the repository's data directory under ``swebench_seq/data``.
# The source-tree fallback keeps editable checkouts working without duplicating
# the benchmark files under ``src``.
_PACKAGED_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR = _PACKAGED_DATA_DIR if _PACKAGED_DATA_DIR.is_dir() else REPO_ROOT / "data"

BENCHMARK_VERSION = "swebench-pro-sequential-v1"
SEQUENTIAL_ORDER_PATH = DATA_DIR / "sequential_order.json"
COMMIT_DATES_PATH = DATA_DIR / "repo_commit_dates.json"
COMMIT_TOPO_RANKS_PATH = DATA_DIR / "repo_topo_ranks.json"

# Public SWE-bench Pro images. Alternative OCI registries can be selected with
# ``--image-prefix``.
IMAGE_PREFIX_DEFAULT = "docker.io/jefzda/sweap-images"
