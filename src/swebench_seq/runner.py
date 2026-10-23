"""Per-task container lifecycle.

The runner is responsible for the *container envelope* — image pull, mount
wiring, base-commit reset, patch extraction via `git diff HEAD`, teardown.
It does NOT drive the agent; that's the mini-swe-agent driver's job
(`agent/driver.py`), which executes shell actions via `exec_in_container`.

Designed for testability: the `PodmanCli` wrapper is the only place
subprocess actually fires. The runner takes it as a dependency, so unit tests
can swap in a stub.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .constants import IMAGE_PREFIX_DEFAULT as _IMAGE_PREFIX
from .container_network import network_mode_for
from .types import Task

log = logging.getLogger(__name__)

_AGENT_USER = "agent"
_APP_DIR = "/app"
_AGENT_HOME = "/home/agent"
_WORKSPACE = "/workspace"

# Per-step podman exec timeouts. The work inside the container is fast;
# these budgets exist for the daemon's exec-queue overhead under 40-way
# concurrent load.
_POST_SETUP_COMMIT_TIMEOUT = 120  # git commit --allow-empty baseline
_HARDEN_GIT_HISTORY_TIMEOUT = 300  # fresh baseline repo creation
_RESET_TO_BASE_TIMEOUT = 600


def default_podman_prefix() -> tuple[str, ...]:
    """Return the configured Podman command.

    Rootless ``podman`` is the default. Deployments may select a wrapper or a
    different executable with ``SWEBENCH_PODMAN_COMMAND``.
    """

    value = os.environ.get("SWEBENCH_PODMAN_COMMAND", "podman")
    prefix = tuple(shlex.split(value))
    if not prefix:
        raise ValueError("SWEBENCH_PODMAN_COMMAND must not be empty")
    return prefix

class ContainerError(RuntimeError):
    """Raised when a podman invocation fails or the container is in a bad state."""


def _is_container_not_running_error(exc: ContainerError) -> bool:
    text = str(exc).lower()
    return (
        "container state improper" in text
        or "can only create exec sessions on running containers" in text
        or "is not running" in text
        or "no container with name or id" in text
        or "no such container" in text
    )


def _is_transient_podman_run_error(exc: ContainerError) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "unable to copy from source",
            "writing blob",
            "unpacking failed",
            "mkdir /app: no such file or directory",
            "unexpected eof",
            "connection reset by peer",
            "i/o timeout",
            "tls handshake timeout",
            "tls: expired certificate",
            "temporary failure",
            "context deadline exceeded",
        )
    )


def _is_git_index_lock_error(exc: ContainerError) -> bool:
    text = str(exc).lower()
    return ".git/index.lock" in text and "file exists" in text


@dataclass
class PodmanCli:
    """Thin wrapper around the `podman` binary. The single subprocess seam.

    Defaults to ordinary rootless Podman. A different command is an explicit
    deployment choice through ``SWEBENCH_PODMAN_COMMAND`` or constructor
    injection.
    """

    prefix: tuple[str, ...] = field(default_factory=default_podman_prefix)

    def _argv(self, *args: str) -> list[str]:
        return [*self.prefix, *args]

    def version(self) -> str:
        """Return the Podman client version or raise ``ContainerError``."""

        result = self._run(
            self._argv("version", "--format", "{{.Client.Version}}"),
            timeout=15,
        )
        return result.stdout.strip()

    def run_detached(
        self,
        *,
        image: str,
        name: str,
        env: Mapping[str, str],
        bind_mounts: Iterable[tuple[Path, str, str]] = (),  # (host, container, mode)
        entrypoint: str | None = None,
    ) -> str:
        """Spawn a detached container."""
        argv = self._argv("run", "-d")
        network_mode = network_mode_for("rollout")
        if network_mode is not None:
            argv.append(f"--network={network_mode}")
        argv += ["--name", name]
        # Always force the entrypoint to `sleep` so the container stays alive
        # for `podman exec`. SWE-bench Pro images have ENTRYPOINT=/bin/bash,
        # which would consume our `sleep infinity` args and exit immediately.
        effective_entrypoint = entrypoint if entrypoint is not None else "sleep"
        argv += ["--entrypoint", effective_entrypoint]
        for host, container, mode in bind_mounts:
            argv += ["-v", f"{host}:{container}:{mode}"]
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        argv.append(image)
        if entrypoint is None:
            argv += ["infinity"]
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                return self._run(argv).stdout.strip()
            except ContainerError as e:
                if attempt >= attempts or not _is_transient_podman_run_error(e):
                    raise
                log.warning(
                    "podman run failed with transient pull/unpack error "
                    "(attempt %d/%d) for %s; retrying",
                    attempt,
                    attempts,
                    image,
                )
                self._run(
                    self._argv("rm", "-f", "-t", "1", name),
                    check=False,
                )
                time.sleep(2 * attempt)
        raise ContainerError(f"podman run retry loop exhausted for {image}")

    def exec(
        self,
        container: str,
        *,
        cmd: list[str],
        user: str | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = self._argv("exec")
        if user is not None:
            argv += ["--user", user]
        if workdir is not None:
            argv += ["--workdir", workdir]
        argv.append(container)
        argv += cmd
        return self._run(argv, timeout=timeout, check=check)

    def cp_from(self, container: str, src: str, dst: Path) -> None:
        self._run(self._argv("cp", f"{container}:{src}", str(dst)))

    def stop(self, container: str, *, timeout_seconds: int = 5) -> None:
        # `rm -f` is a hard kill; for SWE-bench Pro per-task containers we
        # don't care about graceful shutdown — the patch is already on disk.
        self._run(
            self._argv("rm", "-f", "-t", str(timeout_seconds), container),
            check=False,
        )

    def start(self, container: str) -> None:
        self._run(self._argv("start", container))

    def pull(self, image: str) -> None:
        self._run(self._argv("pull", image))

    def image_exists(self, image: str) -> bool:
        result = self._run(self._argv("image", "exists", image), check=False)
        return result.returncode == 0

    def image_identity(self, image: str) -> dict[str, object]:
        """Return immutable identity fields for a locally available image."""

        result = self._run(self._argv("image", "inspect", image))
        try:
            payload = json.loads(result.stdout)
            details = payload[0] if isinstance(payload, list) and payload else payload
        except json.JSONDecodeError as exc:
            raise ContainerError(f"Podman returned invalid image metadata for {image}") from exc
        if not isinstance(details, dict):
            raise ContainerError(f"Podman returned invalid image metadata for {image}")
        repo_digests = details.get("RepoDigests")
        return {
            "reference": image,
            "id": details.get("Id") if isinstance(details.get("Id"), str) else None,
            "digest": (
                details.get("Digest") if isinstance(details.get("Digest"), str) else None
            ),
            "repo_digests": (
                [item for item in repo_digests if isinstance(item, str)]
                if isinstance(repo_digests, list)
                else []
            ),
        }

    def tag(self, source: str, target: str) -> None:
        self._run(self._argv("tag", source, target))

    def _run(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        log.debug("podman: %s", shlex.join(argv))
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise ContainerError(f"podman timed out: {shlex.join(argv)}") from e
        except OSError as e:
            raise ContainerError(f"could not execute Podman: {shlex.join(argv)}: {e}") from e
        if check and result.returncode != 0:
            raise ContainerError(
                f"podman exited {result.returncode}: {shlex.join(argv)}\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
            )
        return result


@dataclass
class ContainerRunner:
    """Per-task container lifecycle.

    Usage:
        runner = ContainerRunner(task, artifacts_dir)
        runner.start(env={"HTTPS_PROXY": "..."})
        try:
            ... # mini driver execs the agent's shell actions here
            runner.extract_patch()
        finally:
            runner.stop()
    """

    task: Task
    artifacts_dir: Path
    agent_home_mount: Path | None = None
    podman: PodmanCli = field(default_factory=PodmanCli)
    container_name: str = ""
    image_prefix: str = _IMAGE_PREFIX
    _started: bool = False
    _image_identity: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.container_name:
            self.container_name = (
                f"swebench-seq-{self.task.instance_id}-{uuid.uuid4().hex[:8]}"
            )

    @property
    def image(self) -> str:
        return f"{self.image_prefix}:{self.task.dockerhub_tag}"

    def start(self, *, env: Mapping[str, str]) -> None:
        if self._started:
            raise ContainerError(f"already started: {self.container_name}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        bind_mounts: list[tuple[Path, str, str]] = [
            (self.artifacts_dir, _WORKSPACE, "rw"),
        ]
        if self.agent_home_mount is not None:
            self.agent_home_mount.mkdir(parents=True, exist_ok=True)
            bind_mounts.append((self.agent_home_mount, _AGENT_HOME, "rw"))

        if not self.podman.image_exists(self.image):
            log.info(
                "image %s is not cached; Podman will download it before startup "
                "(this may take several minutes)",
                self.image,
            )
        self.podman.run_detached(
            image=self.image,
            name=self.container_name,
            env=env,
            bind_mounts=bind_mounts,
            entrypoint=None,
        )
        self._started = True
        self._image_identity = self.podman.image_identity(self.image)
        log.info(
            "started container %s (image=%s)",
            self.container_name,
            self.image,
        )

        # Every benchmark image starts at the solution commit. Always reset to
        # the task's declared base commit before exposing the repository to the
        # agent; this invariant must not depend on an unrelated dataset field.
        self._reset_to_base_commit()
        # Commit the reset state to HEAD so `git diff HEAD` after the agent runs
        # contains only the agent's net changes. `--no-verify` and the disabled
        # hooks path prevent repository-local hooks from running here.
        self.podman.exec(
            self.container_name,
            cmd=[
                "bash",
                "-lc",
                "git -c user.email=swebench-seq@local "
                "-c user.name=swebench-seq "
                "-c core.hooksPath=/dev/null "
                "add -A && git -c user.email=swebench-seq@local "
                "-c user.name=swebench-seq "
                "-c core.hooksPath=/dev/null "
                "commit --allow-empty --no-verify "
                "-m 'swebench-seq: post-setup baseline' "
                "--quiet",
            ],
            workdir=_APP_DIR,
            user=None,
            timeout=_POST_SETUP_COMMIT_TIMEOUT,
        )
        self._harden_git_history()

    def _harden_git_history(self) -> None:
        """Replace image Git metadata with a fresh one-commit baseline repo.

        This makes the fix_commit (and any other future commits) undiscoverable
        via `git log --all` / `git show <hash>`.

        This always runs after the post-reset baseline commit. It is not
        configurable because exposing the solution commit invalidates the
        evaluation.

        We intentionally do not mutate refs/packs in place. That previous
        approach used `git repack -ad` + `git prune` after deleting refs, and a
        Podman exec transport failure in the middle could leave `.git`
        half-mutated while the agent continued. Instead, snapshot the tracked
        base-tree paths, remove `.git`, initialize a new repo, and commit that
        exact tree as the only history.
        """
        script = (
            "set -e; "
            'TRACKED=$(mktemp /tmp/swebench-seq-tracked.XXXXXX); '
            'git ls-files -z > "$TRACKED"; '
            "rm -rf .git; "
            "git init -q -b main 2>/dev/null || "
            "{ git init -q; git checkout -q -B main; }; "
            "if git add -h 2>&1 | grep -q -- '--pathspec-from-file'; then "
            'git add -f --pathspec-from-file="$TRACKED" --pathspec-file-nul; '
            "else "
            'xargs -0 -r git add -f -- < "$TRACKED"; '
            "fi; "
            'rm -f "$TRACKED"; '
            "git -c user.email=swebench-seq@local "
            "-c user.name=swebench-seq "
            "-c core.hooksPath=/dev/null "
            "commit --allow-empty --no-verify "
            "-m 'swebench-seq: hardened baseline' "
            "--quiet; "
            'echo "[swseq] git history hardened: fresh baseline repo"'
        )
        for attempt in (1, 2):
            result = self.podman.exec(
                self.container_name,
                cmd=["bash", "-lc", script],
                workdir=_APP_DIR,
                user=None,
                timeout=_HARDEN_GIT_HISTORY_TIMEOUT,
                check=False,
            )
            if result.returncode == 0:
                self._validate_git_repository_after_hardening()
                return
            log.warning(
                "_harden_git_history attempt %d exited %d for %s: %s",
                attempt,
                result.returncode,
                self.container_name,
                (result.stderr or "")[:300],
            )
        raise ContainerError(
            "git-history hardening failed; refusing to run agent with "
            f"possibly leaked or corrupted git metadata in {self.container_name}"
        )

    def _validate_git_repository_after_hardening(self) -> None:
        # `git add -N .` is intentionally included: the final patch extractor
        # depends on Git being able to create `.git/index.lock`. A corrupted
        # hardening run can still leave enough metadata for simple reads while
        # failing exactly there.
        script = (
            "set -e; "
            "git rev-parse --is-inside-work-tree >/dev/null; "
            "test -d .git; "
            "git rev-parse --verify HEAD >/dev/null; "
            "git add -N . >/dev/null; "
            "git diff --quiet HEAD -- || test $? -eq 1"
        )
        result = self.podman.exec(
            self.container_name,
            cmd=["bash", "-lc", script],
            workdir=_APP_DIR,
            user=None,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise ContainerError(
                "git repository validation failed after history hardening "
                f"(exit {result.returncode}): stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )

    def _reset_to_base_commit(self) -> None:
        """Reset the agent's working tree to task.base_commit.

        The container image's HEAD is at fix_commit; this moves the source
        back to base so the agent sees the buggy code. Mirrors the
        SWE-Agent reference scaffold
        (helper_code/generate_sweagent_instances.py), which passes only the
        base commit to the agent. Dataset setup commands are deliberately never
        loaded or executed in the rollout process because they can reference
        held-out test bodies.

        The upstream SWE-bench Pro evaluator (swe_bench_pro_eval.py:96)
        runs that held-out-test checkout inside the EVAL container AFTER
        `git apply patch.diff`, so held-out tests overwrite anything the
        agent's patch may have touched in the test tree — and the agent
        never gets to read them at task time.
        """
        cmd = f"git reset --hard {self.task.base_commit} && git clean -fd"
        log.info("resetting agent tree to base_commit: %s", cmd)
        result = self.podman.exec(
            self.container_name,
            cmd=["bash", "-lc", cmd],
            workdir=_APP_DIR,
            timeout=_RESET_TO_BASE_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            raise ContainerError(
                f"reset-to-base failed (exit {result.returncode}): {cmd}\n"
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
            )

    def extract_patch(self) -> str:
        """Run `git diff HEAD` inside the container and return the patch text.

        Also writes the patch to `<artifacts_dir>/patch.diff` so it survives
        teardown. Empty patch is a valid outcome — the agent may have given up.

        We `git add -N .` (intent-to-add) FIRST so that new untracked files the
        agent created show up in `git diff HEAD`. Plain `git diff HEAD` only
        shows changes to tracked files and would silently omit new files.
        """
        if not self._started:
            raise ContainerError("cannot extract patch from un-started container")
        try:
            result = self._run_git_diff_head()
        except ContainerError as e:
            if _is_git_index_lock_error(e):
                result = self._retry_git_diff_after_stale_index_lock(e)
                return self._write_patch(result.stdout)
            if not _is_container_not_running_error(e):
                raise
            log.warning(
                "container %s stopped before patch extraction; restarting once",
                self.container_name,
            )
            self.podman.start(self.container_name)
            result = self._run_git_diff_head()
        return self._write_patch(result.stdout)

    def _retry_git_diff_after_stale_index_lock(
        self,
        original_error: ContainerError,
    ) -> subprocess.CompletedProcess[str]:
        if not self._remove_stale_git_index_lock():
            raise original_error
        log.warning(
            "container %s had a stale git index lock during patch extraction; "
            "removed it and retrying once",
            self.container_name,
        )
        return self._run_git_diff_head()

    def _write_patch(self, patch: str) -> str:
        patch_path = self.artifacts_dir / "patch.diff"
        patch_path.write_text(patch)
        log.info("wrote %d-byte patch to %s", len(patch), patch_path)
        return patch

    def _run_git_diff_head(self) -> subprocess.CompletedProcess[str]:
        # `user=None`: per-task SWE-bench Pro images don't have an `agent` user;
        # exec as the image default (root).
        return self.podman.exec(
            self.container_name,
            cmd=["bash", "-lc", "git add -N . && git diff HEAD"],
            workdir=_APP_DIR,
            user=None,
            timeout=120,
        )

    def _remove_stale_git_index_lock(self) -> bool:
        # Only remove the lock when no live git process is visible. The pgrep
        # pattern uses [g]it so pgrep does not match its own command line.
        script = (
            "if command -v pgrep >/dev/null 2>&1; then "
            "for _ in 1 2 3; do "
            "if pgrep -f '(^|/| )[g]it( |-|$)' >/dev/null 2>&1; then "
            "sleep 1; "
            "else rm -f .git/index.lock; exit 0; "
            "fi; "
            "done; "
            "exit 42; "
            "fi; "
            "rm -f .git/index.lock"
        )
        result = self.podman.exec(
            self.container_name,
            cmd=["bash", "-lc", script],
            workdir=_APP_DIR,
            user=None,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            return True
        log.warning(
            "container %s still appears to have a live git process; "
            "leaving .git/index.lock in place",
            self.container_name,
        )
        return False

    def exec_in_container(
        self,
        cmd: list[str],
        *,
        user: str | None = None,
        workdir: str | None = _APP_DIR,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Escape hatch for the agent driver to run arbitrary commands."""
        if not self._started:
            raise ContainerError("container not started")
        return self.podman.exec(
            self.container_name,
            cmd=cmd,
            user=user,
            workdir=workdir,
            timeout=timeout,
            check=check,
        )

    def stop(self) -> None:
        if not self._started:
            return
        self.podman.stop(self.container_name)
        self._started = False
        if self._image_identity is not None:
            # Write provenance only after the untrusted task container is gone;
            # /workspace is writable inside the container while it is running.
            (self.artifacts_dir / "container_image.json").write_text(
                json.dumps(self._image_identity, indent=2) + "\n"
            )
        log.info("stopped container %s", self.container_name)

    def __enter__(self) -> ContainerRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
