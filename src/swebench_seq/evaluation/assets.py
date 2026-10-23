"""Pinned acquisition of the public SWE-bench Pro evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import httpx

EVALUATOR_REPOSITORY = "scaleapi/SWE-bench_Pro-os"
EVALUATOR_REVISION = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
EVALUATOR_ARCHIVE_SHA256 = (
    "f65ce909a991b1d8f4ba2476b7a9b81c48d6ac2bbe69a3b1204c5e2a74219ad9"
)
EVALUATOR_TREE_SHA256 = "4f8870ad4c225de34aa00ef5df99eaa62bfd80d1e3d54a68bc7134fa91603478"
EVALUATOR_ARCHIVE_URL = (
    f"https://codeload.github.com/{EVALUATOR_REPOSITORY}/tar.gz/{EVALUATOR_REVISION}"
)
EVALUATOR_DIR_ENV = "SWEBENCH_PRO_EVALUATOR_DIR"

_CACHE_MARKER = ".swebench-seq-evaluator.json"
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000


def default_evaluator_dir() -> Path:
    """Return the cache location for the pinned evaluator revision."""

    configured = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return cache_root / "swebench-seq" / "evaluators" / EVALUATOR_REVISION


def resolve_evaluator_dir(explicit: Path | None = None) -> Path:
    """Resolve an override or materialize the pinned public evaluator."""

    configured = explicit
    if configured is None:
        value = os.environ.get(EVALUATOR_DIR_ENV)
        configured = Path(value).expanduser() if value else None
    if configured is not None:
        return validate_evaluator_dir(configured.expanduser().resolve())
    return materialize_evaluator()


def validate_evaluator_dir(path: Path) -> Path:
    """Validate the files used by the local evaluator adapter."""

    required = (
        path / "swe_bench_pro_eval.py",
        path / "helper_code" / "image_uri.py",
        path / "run_scripts",
        path / "dockerfiles" / "base_dockerfile",
        path / "dockerfiles" / "instance_dockerfile",
    )
    missing = [item.relative_to(path).as_posix() for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            f"invalid SWE-bench Pro evaluator directory: {path}; "
            f"missing {', '.join(missing)}"
        )
    return path


def materialize_evaluator() -> Path:
    """Download, verify, and atomically cache the pinned evaluator snapshot."""

    target = default_evaluator_dir()
    if _cached_evaluator_is_valid(target):
        return target
    if target.exists():
        raise RuntimeError(
            f"evaluator cache is incomplete or invalid: {target}; remove that "
            "revision directory and retry"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".evaluator-", dir=target.parent))
    archive_path = temporary_root / "source.tar.gz"
    extracted = temporary_root / "extracted"
    try:
        _download_archive(archive_path)
        digest = _sha256(archive_path)
        if digest != EVALUATOR_ARCHIVE_SHA256:
            raise RuntimeError(
                "downloaded SWE-bench Pro evaluator failed checksum verification"
            )
        _extract_archive(archive_path, extracted)
        validate_evaluator_dir(extracted)
        if _tree_sha256(extracted) != EVALUATOR_TREE_SHA256:
            raise RuntimeError(
                "downloaded SWE-bench Pro evaluator failed content verification"
            )
        (extracted / _CACHE_MARKER).write_text(
            json.dumps(
                {
                    "repository": EVALUATOR_REPOSITORY,
                    "revision": EVALUATOR_REVISION,
                    "archive_sha256": EVALUATOR_ARCHIVE_SHA256,
                    "tree_sha256": EVALUATOR_TREE_SHA256,
                },
                indent=2,
            )
            + "\n"
        )
        try:
            extracted.rename(target)
        except FileExistsError:
            if not _cached_evaluator_is_valid(target):
                raise RuntimeError(
                    f"another process created an invalid cache at {target}"
                ) from None
        return target
    except (httpx.HTTPError, OSError, tarfile.TarError) as exc:
        raise RuntimeError(
            f"failed to download the pinned SWE-bench Pro evaluator from "
            f"{EVALUATOR_ARCHIVE_URL}"
        ) from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _cached_evaluator_is_valid(path: Path) -> bool:
    try:
        marker = json.loads((path / _CACHE_MARKER).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(marker, dict):
        return False
    if marker.get("revision") != EVALUATOR_REVISION:
        return False
    if marker.get("archive_sha256") != EVALUATOR_ARCHIVE_SHA256:
        return False
    if marker.get("tree_sha256") != EVALUATOR_TREE_SHA256:
        return False
    try:
        validate_evaluator_dir(path)
    except RuntimeError:
        return False
    return _tree_sha256(path, exclude={_CACHE_MARKER}) == EVALUATOR_TREE_SHA256


def _download_archive(destination: Path) -> None:
    timeout = httpx.Timeout(120, connect=20)
    with httpx.stream(
        "GET",
        EVALUATOR_ARCHIVE_URL,
        follow_redirects=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        try:
            announced_size = int(content_length) if content_length is not None else None
        except ValueError:
            announced_size = None
        if announced_size is not None and announced_size > _MAX_ARCHIVE_BYTES:
            raise RuntimeError("SWE-bench Pro evaluator archive exceeds the size limit")
        size = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > _MAX_ARCHIVE_BYTES:
                    raise RuntimeError("SWE-bench Pro evaluator archive exceeds the size limit")
                handle.write(chunk)


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract regular files/directories after validating every archive path."""

    destination.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
            raise RuntimeError("SWE-bench Pro evaluator archive has an invalid file count")
        roots = {PurePosixPath(member.name).parts[0] for member in members if member.name}
        if len(roots) != 1:
            raise RuntimeError("SWE-bench Pro evaluator archive has an invalid root")
        total_size = sum(member.size for member in members if member.isfile())
        if total_size > _MAX_EXTRACTED_BYTES:
            raise RuntimeError("SWE-bench Pro evaluator archive exceeds the size limit")

        root = next(iter(roots))
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != root or ".." in parts or member.name.startswith("/"):
                raise RuntimeError("SWE-bench Pro evaluator archive contains an unsafe path")
            relative_parts = parts[1:]
            if not relative_parts:
                continue
            output = destination.joinpath(*relative_parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError("SWE-bench Pro evaluator archive contains a link")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("SWE-bench Pro evaluator archive contains an unreadable file")
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            output.chmod(member.mode & 0o777)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path, *, exclude: set[str] | None = None) -> str:
    """Hash relative paths and contents independent of archive metadata."""

    excluded = exclude or set()
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if relative in excluded or not item.is_file():
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
