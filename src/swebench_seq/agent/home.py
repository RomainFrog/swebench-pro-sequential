"""Agent-home persistence across tasks in a universe (sequential mode).

The implementation uses an extract → bind-mount cycle so state survives while
each task receives a fresh container.

Lifecycle per universe:
  1. memory_seeded = False
  2. For task in universe:
     - is_first = not memory_seeded
     - if not is_first: bind-mount <host_home_dir> at /home/agent
     - run agent
     - if is_first: extract /home/agent/. from the container into
       <host_home_dir>, set memory_seeded = True
     - subsequent tasks pick up via bind mount
  3. After each task: snapshot the host_home_dir into the per-task
     artifacts dir so intermediate agent state (MEMORY.md, memory/<date>.md,
     and anything else the agent persisted under /home/agent) is preserved
     for offline analysis.

Symlinks in /home/agent/bin/* (or anywhere) must be preserved as
symlinks and never followed while copying or pruning.

`podman cp container:/home/agent/. <dst>` (trailing /.) copies CONTENTS
into dst, not a nested 'agent' dir.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

_AGENT_HOME_IN_CONTAINER = "/home/agent"
# Default exclusion list for snapshot_agent_home AND prune_agent_home — caches
# and package directories the agent doesn't write to deliberately. Excluding
# these keeps per-task snapshots small AND keeps the live carried /home/agent
# small (low MB instead of multi-GB on Go/Java/Rust repos that warm up
# toolchain caches inside /home/agent, e.g. .cache/go-build).
#
# Intentionally KEPT in snapshots (do NOT add):
#   custom agent state  — useful for offline analysis
#   .git                — if the agent cloned something, we want to see it
SNAPSHOT_EXCLUDE_DEFAULT: tuple[str, ...] = (
    # Generic
    ".cache",                  # generic XDG cache (any level)
    # JS / Node
    ".npm",                    # npm install cache — ~20 MB on JS repos
    "node_modules",            # in case the agent cargo-cults one into $HOME
    ".local/share/pnpm-store", # pnpm shared store
    ".yarn/cache",             # yarn cache
    # Python
    ".pytest_cache",
    "__pycache__",
    # JVM
    ".gradle",                 # multi-GB on element-web/flipt-style Java tests
    ".m2",                     # Maven repo
    # Rust
    ".cargo",
    ".rustup",
    # Generated diagnostics (may contain machine-local upload tokens)
    ".config/go/telemetry",
    # IDE / dev tools
    ".vscode-server",          # easily hundreds of MB
)


class _ContainerCopier(Protocol):
    """Minimal interface PodmanCli satisfies; lets tests stub the seam."""

    def cp_from(self, container: str, src: str, dst: Path) -> None: ...


@dataclass
class AgentHomePersistence:
    """Extract /home/agent from a finished container; chown the host copy."""

    host_dir: Path
    podman: _ContainerCopier

    def extract(self, container_name: str, fallback_dir: Path | None = None) -> None:
        """Copy the container's /home/agent/. into `host_dir` (replacing it).

        Trailing `/.` semantics: contents go directly into host_dir without
        a wrapping `agent/` directory.
        """
        # Wipe any prior state. The first-task extraction is authoritative.
        if self.host_dir.exists():
            _rmtree_skip_symlinks(self.host_dir)
        self.host_dir.mkdir(parents=True, exist_ok=True)

        # `podman cp` with trailing /. copies the directory's *contents* into
        # dst, never an extra nesting level.
        try:
            self.podman.cp_from(
                container_name,
                f"{_AGENT_HOME_IN_CONTAINER}/.",
                self.host_dir,
            )
            log.info("extracted %s/. → %s", _AGENT_HOME_IN_CONTAINER, self.host_dir)
        except Exception:
            if fallback_dir is None or not fallback_dir.is_dir():
                raise
            log.warning(
                "podman cp %s/. failed for %s; restoring agent home from %s",
                _AGENT_HOME_IN_CONTAINER,
                container_name,
                fallback_dir,
            )
            _copy_tree_contents(fallback_dir, self.host_dir)


def clear_agent_home(home_dir: Path) -> None:
    """Replace an agent-home directory with an empty directory."""

    if home_dir.exists() or home_dir.is_symlink():
        _rmtree_skip_symlinks(home_dir)
    home_dir.mkdir(parents=True, exist_ok=True)


def restore_agent_home(snapshot_dir: Path, home_dir: Path) -> None:
    """Replace an agent home with a previously captured snapshot."""

    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"agent-home snapshot not found: {snapshot_dir}")
    clear_agent_home(home_dir)
    _copy_tree_contents(snapshot_dir, home_dir)


def snapshot_agent_home(
    src_dir: Path,
    dst_dir: Path,
    exclude: Iterable[str] = SNAPSHOT_EXCLUDE_DEFAULT,
) -> bool:
    """Snapshot src_dir's contents into dst_dir, excluding given patterns.

    Returns True if the snapshot was written, False if src_dir is missing.

    Rootless Podman maps container-root files to the invoking host user, so a
    standard-library copy is sufficient. Symlinks are preserved and never
    followed.

    Why a *blocklist* not allowlist: the agent may persist arbitrary files
    under /home/agent (custom scripts in bin/, scratch notes, etc.) that
    we want to capture without enumerating in advance. We only skip
    well-known caches.
    """
    if not src_dir.is_dir():
        return False
    # Wipe any prior snapshot at this exact path (idempotent re-runs).
    if dst_dir.exists():
        _rmtree_skip_symlinks(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        _copy_tree_contents(src_dir, dst_dir, exclude=tuple(exclude))
    except OSError as exc:
        log.warning(
            "snapshot_agent_home(%s → %s) copy failed: %s",
            src_dir,
            dst_dir,
            exc,
        )
        return False
    return True


def prune_agent_home(
    home_dir: Path,
    exclude: Iterable[str] = SNAPSHOT_EXCLUDE_DEFAULT,
) -> None:
    """Delete toolchain caches in-place from the LIVE carried /home/agent dir.

    The agent runs as root with $HOME=/home/agent, so toolchains drop large
    caches (.cache/go-build, .gradle, .m2, .cargo, ...) into the dir we persist
    across a universe's tasks. Left alone they accumulate multi-GB and re-ship
    every task. Call this after each task to keep only real agent state
    (MEMORY.md, memory/, and tool-specific configuration).

    Exclude semantics mirror :func:`snapshot_agent_home`: a bare name (".cache")
    matches at ANY depth; a name containing "/" (".yarn/cache") matches a
    relative path. Deletion is symlink-safe and assumes standard rootless
    Podman ownership.
    """
    if not home_dir.is_dir():
        return
    names = {e for e in exclude if "/" not in e}
    relpaths = [e for e in exclude if "/" in e]
    for rel in relpaths:
        _rmtree_skip_symlinks(home_dir / rel)
    for dirpath, dirnames, filenames in os.walk(home_dir, topdown=True, followlinks=False):
        for name in list(dirnames):
            if name in names:
                _rmtree_skip_symlinks(Path(dirpath) / name)
                dirnames.remove(name)  # don't descend into a just-removed dir
        for name in filenames:
            if name in names:
                target = Path(dirpath) / name
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass


def _rmtree_skip_symlinks(path: Path) -> None:
    """Recursive delete that doesn't follow symlinks into their targets.

    ``shutil.rmtree`` already skips symlink targets; the explicit top-level
    check also handles broken symlinks.
    """
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    shutil.rmtree(path)


def _copy_tree_contents(
    src: Path,
    dst: Path,
    *,
    exclude: tuple[str, ...] = (),
    relative_root: Path = Path(),
) -> None:
    """Copy directory contents without following symlinks.

    A bare exclusion matches that name at any depth. An exclusion containing
    ``/`` matches a path relative to ``src`` and all of its descendants.
    """
    for entry in src.iterdir():
        relative = relative_root / entry.name
        relative_text = relative.as_posix()
        if any(
            ("/" not in pattern and entry.name == pattern)
            or (
                "/" in pattern
                and (
                    relative_text == pattern
                    or relative_text.startswith(f"{pattern}/")
                )
            )
            for pattern in exclude
        ):
            continue
        target = dst / entry.name
        if entry.is_symlink():
            target.symlink_to(os.readlink(entry))
        elif entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_tree_contents(
                entry,
                target,
                exclude=exclude,
                relative_root=relative,
            )
        else:
            shutil.copy2(entry, target, follow_symlinks=False)
