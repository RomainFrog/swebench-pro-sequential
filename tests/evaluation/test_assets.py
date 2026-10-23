"""Tests for pinned evaluator acquisition and validation."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from swebench_seq.evaluation import assets as evaluator_assets


def _make_evaluator_tree(root: Path) -> None:
    (root / "helper_code").mkdir(parents=True)
    (root / "run_scripts" / "instance_one").mkdir(parents=True)
    (root / "dockerfiles" / "base_dockerfile").mkdir(parents=True)
    (root / "dockerfiles" / "instance_dockerfile").mkdir(parents=True)
    (root / "swe_bench_pro_eval.py").write_text("print('evaluator')\n")
    (root / "helper_code" / "image_uri.py").write_text("def image(): return 'image'\n")
    (root / "run_scripts" / "instance_one" / "run_script.sh").write_text("true\n")


def _make_archive(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    _make_evaluator_tree(source)
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname="SWE-bench_Pro-os-revision")
    return source, archive_path


def test_resolve_evaluator_dir_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "checkout"
    _make_evaluator_tree(explicit)

    assert evaluator_assets.resolve_evaluator_dir(explicit) == explicit.resolve()


def test_resolve_evaluator_dir_uses_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "checkout"
    _make_evaluator_tree(explicit)
    monkeypatch.setenv(evaluator_assets.EVALUATOR_DIR_ENV, str(explicit))

    assert evaluator_assets.resolve_evaluator_dir() == explicit.resolve()


def test_materialize_evaluator_verifies_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive_path = _make_archive(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        evaluator_assets,
        "EVALUATOR_ARCHIVE_SHA256",
        evaluator_assets._sha256(archive_path),
    )
    monkeypatch.setattr(
        evaluator_assets,
        "EVALUATOR_TREE_SHA256",
        evaluator_assets._tree_sha256(source),
    )
    downloads = 0

    def download(destination: Path) -> None:
        nonlocal downloads
        downloads += 1
        destination.write_bytes(archive_path.read_bytes())

    monkeypatch.setattr(evaluator_assets, "_download_archive", download)

    first = evaluator_assets.materialize_evaluator()
    second = evaluator_assets.materialize_evaluator()

    assert first == second == evaluator_assets.default_evaluator_dir()
    assert (first / "swe_bench_pro_eval.py").is_file()
    assert downloads == 1


def test_materialize_evaluator_rejects_wrong_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _source, archive_path = _make_archive(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(evaluator_assets, "EVALUATOR_ARCHIVE_SHA256", "0" * 64)
    monkeypatch.setattr(
        evaluator_assets,
        "_download_archive",
        lambda destination: destination.write_bytes(archive_path.read_bytes()),
    )

    with pytest.raises(RuntimeError, match="checksum verification"):
        evaluator_assets.materialize_evaluator()
    assert not evaluator_assets.default_evaluator_dir().exists()


def test_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("root/../../outside")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe path"):
        evaluator_assets._extract_archive(archive_path, tmp_path / "out")
    assert not (tmp_path / "outside").exists()
