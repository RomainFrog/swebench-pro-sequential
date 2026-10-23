"""Release metadata stays consistent across package files."""

from __future__ import annotations

import tomllib
from pathlib import Path

from swebench_seq import __version__


def test_project_and_runtime_versions_match() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    locked_project = next(
        package for package in lock["package"] if package["name"] == "swebench-seq"
    )

    assert project["project"]["version"] == __version__
    assert locked_project["version"] == __version__
