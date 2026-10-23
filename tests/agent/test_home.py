"""Tests for rootless-Podman agent-home persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from swebench_seq.agent.home import (
    AgentHomePersistence,
    clear_agent_home,
    prune_agent_home,
    restore_agent_home,
    snapshot_agent_home,
)


@dataclass
class FakePodman:
    """Stub: cp_from drops a fixture tree at the destination."""

    fixture_builder: callable
    calls: list[tuple[str, str, Path]] = field(default_factory=list)

    def cp_from(self, container: str, src: str, dst: Path) -> None:
        self.calls.append((container, src, dst))
        self.fixture_builder(dst)


def _build_simple_tree(dst: Path) -> None:
    """A plausible /home/agent layout: files, subdir, and a symlink."""
    (dst / "MEMORY.md").write_text("- learned X\n- learned Y\n")
    (dst / "notes").mkdir()
    (dst / "notes" / "qutebrowser.md").write_text("test command: tox -e py312\n")
    (dst / "bin").mkdir()
    # Symlink that, if followed, would point to a system binary that must
    # NOT have its ownership stomped.
    (dst / "bin" / "ls").symlink_to("/usr/bin/ls")


def test_extract_invokes_podman_with_trailing_slash_dot(tmp_path: Path) -> None:
    podman = FakePodman(fixture_builder=_build_simple_tree)
    persistence = AgentHomePersistence(host_dir=tmp_path / "home", podman=podman)
    persistence.extract("some-container")

    assert len(podman.calls) == 1
    _, src, dst = podman.calls[0]
    assert src == "/home/agent/."
    assert dst == tmp_path / "home"


def test_extract_materializes_files_and_preserves_symlink(tmp_path: Path) -> None:
    podman = FakePodman(fixture_builder=_build_simple_tree)
    persistence = AgentHomePersistence(host_dir=tmp_path / "home", podman=podman)
    persistence.extract("c")

    home = tmp_path / "home"
    assert (home / "MEMORY.md").read_text().startswith("- learned X")
    assert (home / "notes" / "qutebrowser.md").read_text().startswith("test command")
    # Symlink must stay a symlink, not be followed/copied.
    ls_link = home / "bin" / "ls"
    assert ls_link.is_symlink()
    assert os.readlink(ls_link) == "/usr/bin/ls"


def test_extract_wipes_stale_host_dir(tmp_path: Path) -> None:
    podman = FakePodman(fixture_builder=_build_simple_tree)
    host = tmp_path / "home"
    host.mkdir()
    (host / "stale.txt").write_text("from previous run")
    AgentHomePersistence(host_dir=host, podman=podman).extract("c")
    assert not (host / "stale.txt").exists()
    assert (host / "MEMORY.md").exists()


def test_extract_creates_host_dir_if_missing(tmp_path: Path) -> None:
    podman = FakePodman(fixture_builder=_build_simple_tree)
    host = tmp_path / "doesnt" / "exist" / "yet"
    assert not host.exists()
    AgentHomePersistence(host_dir=host, podman=podman).extract("c")
    assert host.exists()
    assert (host / "MEMORY.md").exists()


def test_extract_uses_fallback_when_container_copy_fails(tmp_path: Path) -> None:
    def fail_copy(_dst: Path) -> None:
        raise RuntimeError("missing container")

    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "MEMORY.md").write_text("# exact memory\n")
    (fallback / ".agent-state").mkdir()
    (fallback / ".agent-state" / "session.json").write_text("{}\n")

    host = tmp_path / "home"
    podman = FakePodman(fixture_builder=fail_copy)
    AgentHomePersistence(host_dir=host, podman=podman).extract(
        "gone-container",
        fallback_dir=fallback,
    )

    assert (host / "MEMORY.md").read_text() == "# exact memory\n"
    assert (host / ".agent-state" / "session.json").read_text() == "{}\n"


@pytest.mark.parametrize("subpath", ["MEMORY.md", "notes/qutebrowser.md", "bin/ls"])
def test_extract_preserves_specific_artifacts(tmp_path: Path, subpath: str) -> None:
    podman = FakePodman(fixture_builder=_build_simple_tree)
    host = tmp_path / "home"
    AgentHomePersistence(host_dir=host, podman=podman).extract("c")
    target = host / subpath
    assert target.exists() or target.is_symlink()


# ---- snapshot_agent_home ----


def _build_rich_home(host: Path) -> None:
    """Fixture: memory, custom state, package-cache junk, and scratch files."""
    host.mkdir(parents=True, exist_ok=True)
    (host / "MEMORY.md").write_text("# carried memory\n- learned X\n")
    (host / "memory").mkdir()
    (host / "memory" / "2026-05-20.md").write_text("today I learned ...\n")
    (host / ".agent-state").mkdir()
    (host / ".agent-state" / "session.json").write_text("{}")
    (host / ".npm").mkdir()
    (host / ".npm" / "_cacache").mkdir()
    (host / ".npm" / "_cacache" / "blob.bin").write_text("x" * 10_000)
    (host / ".config" / "go" / "telemetry" / "local").mkdir(parents=True)
    (host / ".config" / "go" / "telemetry" / "local" / "upload.token").write_text(
        "generated-token"
    )
    (host / "scratch.py").write_text("print('agent-invented file')\n")


def test_snapshot_returns_false_when_src_missing(tmp_path: Path) -> None:
    src = tmp_path / "doesnt_exist"
    dst = tmp_path / "snap"
    assert snapshot_agent_home(src, dst) is False
    assert not dst.exists()


def test_snapshot_copies_agent_files_and_excludes_npm(tmp_path: Path) -> None:
    src = tmp_path / "home"
    dst = tmp_path / "snap"
    _build_rich_home(src)
    ok = snapshot_agent_home(src, dst)
    assert ok is True
    # The important agent-written files are present
    assert (dst / "MEMORY.md").read_text() == "# carried memory\n- learned X\n"
    assert (dst / "memory" / "2026-05-20.md").exists()
    # Custom agent state is preserved.
    assert (dst / ".agent-state" / "session.json").exists()
    # Agent-invented files are preserved (blocklist not allowlist)
    assert (dst / "scratch.py").exists()
    # .npm is excluded (the noise)
    assert not (dst / ".npm").exists()
    assert not (dst / ".config" / "go" / "telemetry").exists()


def test_snapshot_replaces_existing_dst(tmp_path: Path) -> None:
    src = tmp_path / "home"
    dst = tmp_path / "snap"
    _build_rich_home(src)
    dst.mkdir()
    (dst / "stale.txt").write_text("from previous run")
    snapshot_agent_home(src, dst)
    # Stale content must be wiped, not merged with the new snapshot
    assert not (dst / "stale.txt").exists()
    assert (dst / "MEMORY.md").exists()


def test_snapshot_preserves_symlinks_as_symlinks(tmp_path: Path) -> None:
    """Never resolve or copy through symlinks during snapshot."""
    src = tmp_path / "home"
    src.mkdir()
    (src / "MEMORY.md").write_text("x")
    target = tmp_path / "outside_target"
    target.write_text("should not be copied through")
    (src / "bin").mkdir()
    os.symlink(target, src / "bin" / "ls")
    dst = tmp_path / "snap"
    snapshot_agent_home(src, dst)
    assert (dst / "bin" / "ls").is_symlink()
    # The symlink should still point to the original target path (not have
    # the target's contents copied in its place).
    assert os.readlink(dst / "bin" / "ls") == str(target)


def test_snapshot_custom_exclude(tmp_path: Path) -> None:
    src = tmp_path / "home"
    src.mkdir()
    (src / "MEMORY.md").write_text("keep")
    (src / "secrets").mkdir()
    (src / "secrets" / "key").write_text("redact")
    dst = tmp_path / "snap"
    snapshot_agent_home(src, dst, exclude=("secrets",))
    assert (dst / "MEMORY.md").exists()
    assert not (dst / "secrets").exists()


def test_restore_agent_home_replaces_existing_contents(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "MEMORY.md").write_text("restored")
    (snapshot / "link").symlink_to("MEMORY.md")
    home = tmp_path / "home"
    home.mkdir()
    (home / "stale.txt").write_text("stale")

    restore_agent_home(snapshot, home)

    assert not (home / "stale.txt").exists()
    assert (home / "MEMORY.md").read_text() == "restored"
    assert (home / "link").is_symlink()


def test_clear_agent_home_creates_empty_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "stale.txt").write_text("stale")

    clear_agent_home(home)

    assert home.is_dir()
    assert list(home.iterdir()) == []


def test_prune_agent_home_missing_dir_noop(tmp_path: Path) -> None:
    # Stub-driver safe: pruning a non-existent home dir does nothing, no raise.
    prune_agent_home(tmp_path / "nope")


def test_prune_agent_home_removes_caches_keeps_state(tmp_path: Path) -> None:
    home = tmp_path / "_agent_home"
    home.mkdir()
    # Real agent state — must survive.
    (home / "MEMORY.md").write_text("# carried memory\n")
    (home / "memory").mkdir()
    (home / "memory" / "2026-06-09.md").write_text("learned\n")
    (home / ".agent-state").mkdir()
    (home / ".agent-state" / "state.json").write_text("{}")
    (home / "scratch.py").write_text("print(1)\n")
    # Toolchain caches — must be dropped.
    (home / ".cache" / "go-build").mkdir(parents=True)
    (home / ".cache" / "go-build" / "ab").write_text("x" * 1000)
    (home / ".gradle").mkdir()
    (home / ".gradle" / "caches").mkdir()
    # Nested .cache (any-depth match) and a relative-path exclude.
    (home / "sub").mkdir()
    (home / "sub" / ".cache").mkdir()
    (home / "sub" / ".cache" / "blob").write_text("y" * 1000)
    (home / ".yarn" / "cache").mkdir(parents=True)
    (home / ".yarn" / "cache" / "pkg.zip").write_text("z" * 1000)
    (home / ".config" / "go" / "telemetry" / "local").mkdir(parents=True)
    (home / ".config" / "go" / "telemetry" / "local" / "upload.token").write_text(
        "generated-token"
    )

    prune_agent_home(home)

    # State preserved
    assert (home / "MEMORY.md").exists()
    assert (home / "memory" / "2026-06-09.md").exists()
    assert (home / ".agent-state" / "state.json").exists()
    assert (home / "scratch.py").exists()
    # Caches gone — top-level, nested, and relative-path
    assert not (home / ".cache").exists()
    assert not (home / ".gradle").exists()
    assert not (home / "sub" / ".cache").exists()
    assert (home / "sub").exists()  # the non-cache parent dir stays
    assert not (home / ".yarn" / "cache").exists()
    assert not (home / ".config" / "go" / "telemetry").exists()


def test_prune_agent_home_keeps_symlinks(tmp_path: Path) -> None:
    home = tmp_path / "_agent_home"
    home.mkdir()
    (home / "MEMORY.md").write_text("x")
    target = tmp_path / "outside"
    target.write_text("do not delete through symlink")
    (home / "bin").mkdir()
    os.symlink(target, home / "bin" / "ls")
    prune_agent_home(home)
    assert (home / "bin" / "ls").is_symlink()
    assert target.read_text() == "do not delete through symlink"
