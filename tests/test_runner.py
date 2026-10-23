"""Tests for ContainerRunner. PodmanCli is stubbed so no real podman runs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from swebench_seq.runner import ContainerError, ContainerRunner, PodmanCli
from swebench_seq.types import Task


@dataclass
class FakePodmanCli(PodmanCli):
    """In-memory stand-in for PodmanCli that records every call."""

    run_detached_calls: list[dict] = field(default_factory=list)
    exec_calls: list[dict] = field(default_factory=list)
    start_calls: list[str] = field(default_factory=list)
    stop_calls: list[str] = field(default_factory=list)
    exec_returns: dict[str, subprocess.CompletedProcess[str]] = field(
        default_factory=dict
    )
    next_run_id: str = "fakecontainerid"
    image_cached: bool = True

    def image_exists(self, image: str) -> bool:
        return self.image_cached

    def image_identity(self, image: str) -> dict[str, object]:
        return {
            "reference": image,
            "id": "sha256:image-id",
            "digest": "sha256:image-digest",
            "repo_digests": [f"{image.split(':', 1)[0]}@sha256:image-digest"],
        }

    def run_detached(
        self,
        *,
        image: str,
        name: str,
        env: Mapping[str, str],
        bind_mounts: Iterable[tuple[Path, str, str]] = (),
        entrypoint: str | None = None,
    ) -> str:
        self.run_detached_calls.append(
            {
                "image": image,
                "name": name,
                "env": dict(env),
                "bind_mounts": list(bind_mounts),
                "entrypoint": entrypoint,
            }
        )
        return self.next_run_id

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
        joined = " ".join(cmd)
        self.exec_calls.append(
            {"container": container, "cmd": cmd, "user": user, "workdir": workdir}
        )
        for key, result in self.exec_returns.items():
            if key in joined:
                if check and result.returncode != 0:
                    raise ContainerError(f"stub exec failed: {joined}")
                return result
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def stop(self, container: str, *, timeout_seconds: int = 5) -> None:
        self.stop_calls.append(container)

    def start(self, container: str) -> None:
        self.start_calls.append(container)


def _task(**overrides: object) -> Task:
    base = {
        "instance_id": "qutebrowser-1",
        "repo": "qutebrowser",
        "base_commit": "abc123",
        "problem_statement": "fix it",
        "dockerhub_tag": "qutebrowser-1",
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


def _runner(
    tmp_path: Path, *, agent_home: Path | None = None, **task_overrides
) -> tuple[ContainerRunner, FakePodmanCli]:
    podman = FakePodmanCli()
    runner = ContainerRunner(
        task=_task(**task_overrides),
        artifacts_dir=tmp_path / "artifacts",
        agent_home_mount=agent_home,
        podman=podman,
    )
    return runner, podman


def test_image_uses_dockerhub_tag(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, dockerhub_tag="qutebrowser-1")
    assert runner.image == "docker.io/jefzda/sweap-images:qutebrowser-1"


def test_container_name_is_unique_per_instance(tmp_path: Path) -> None:
    r1, _ = _runner(tmp_path)
    r2, _ = _runner(tmp_path)
    assert r1.container_name != r2.container_name
    assert r1.container_name.startswith("swebench-seq-qutebrowser-1-")


def test_start_workspace_is_bind_mounted(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={"HTTPS_PROXY": "http://proxy:8080"})
    call = podman.run_detached_calls[0]
    bind_mounts = {
        container: (host, mode) for host, container, mode in call["bind_mounts"]
    }

    assert "/workspace" in bind_mounts
    assert bind_mounts["/workspace"][1] == "rw"
    assert "/home/agent" not in bind_mounts  # independent mode
    runner.stop()
    identity = json.loads((runner.artifacts_dir / "container_image.json").read_text())
    assert identity["id"] == "sha256:image-id"
    assert identity["digest"] == "sha256:image-digest"


def test_start_without_agent_home_omits_home_mount(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path, agent_home=None)
    runner.start(env={})
    bind = {c for _, c, _ in podman.run_detached_calls[0]["bind_mounts"]}
    assert "/home/agent" not in bind


def test_start_with_agent_home_includes_home_mount(tmp_path: Path) -> None:
    home = tmp_path / "home"
    runner, podman = _runner(tmp_path, agent_home=home)
    runner.start(env={})
    bind = {
        container: (host, mode)
        for host, container, mode in podman.run_detached_calls[0]["bind_mounts"]
    }
    assert bind["/home/agent"] == (home, "rw")
    assert home.exists(), "agent home dir should be created on start"


def test_start_propagates_env(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={"MODEL": "gpt-5", "OPENAI_API_KEY": "k"})
    assert podman.run_detached_calls[0]["env"]["MODEL"] == "gpt-5"
    assert podman.run_detached_calls[0]["env"]["OPENAI_API_KEY"] == "k"


def test_start_announces_first_image_download(tmp_path: Path, caplog) -> None:
    runner, podman = _runner(tmp_path)
    podman.image_cached = False
    caplog.set_level("INFO")

    runner.start(env={})

    assert "this may take several minutes" in caplog.text


def test_podman_run_detached_retries_transient_pull_unpack_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class FlakyRunPodman(PodmanCli):
        calls: list[list[str]] = field(default_factory=list)
        run_attempts: int = 0

        def _run(
            self,
            argv: list[str],
            *,
            timeout: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            self.calls.append(argv)
            if "run" in argv:
                self.run_attempts += 1
                if self.run_attempts == 1:
                    raise ContainerError(
                        "podman exited 125: podman run\n"
                        "stderr='Error: unable to copy from source: "
                        "writing blob: unpacking failed (error: exit status 1; "
                        "output: mkdir /app: no such file or directory)'"
                    )
                return subprocess.CompletedProcess(
                    args=argv, returncode=0, stdout="container-id\n", stderr=""
                )
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="", stderr=""
            )

    sleeps: list[float] = []
    monkeypatch.setattr("swebench_seq.runner.time.sleep", sleeps.append)
    podman = FlakyRunPodman(prefix=("podman",))

    cid = podman.run_detached(
        image="repo:image",
        name="task-container",
        env={},
    )

    assert cid == "container-id"
    assert podman.run_attempts == 2
    assert sleeps == [2]
    assert any(call[:5] == ["podman", "rm", "-f", "-t", "1"] for call in podman.calls)


def test_podman_run_uses_public_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class RecordingPodman(PodmanCli):
        calls: list[list[str]] = field(default_factory=list)

        def _run(
            self,
            argv: list[str],
            *,
            timeout: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="container-id\n", stderr=""
            )

    monkeypatch.delenv("SWEBENCH_CONTAINER_NETWORK_MODE", raising=False)
    monkeypatch.delenv("SWEBENCH_ROLLOUT_NETWORK_MODE", raising=False)
    podman = RecordingPodman(prefix=("podman",))
    podman.run_detached(
        image="docker.io/jefzda/sweap-images:tag",
        name="task-container",
        env={},
    )

    assert "--tls-verify=false" not in podman.calls[0]
    assert not any(arg.startswith("--network=") for arg in podman.calls[0])


def test_podman_network_mode_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class RecordingPodman(PodmanCli):
        calls: list[list[str]] = field(default_factory=list)

        def _run(
            self,
            argv: list[str],
            *,
            timeout: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            self.calls.append(argv)
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="container-id\n", stderr=""
            )

    monkeypatch.setenv("SWEBENCH_ROLLOUT_NETWORK_MODE", "host")
    podman = RecordingPodman(prefix=("podman",))
    image = "registry.example.test/swebench-pro:tag"
    podman.run_detached(image=image, name="task-container", env={})
    podman.pull(image)

    assert "--network=host" in podman.calls[0]
    assert "--tls-verify=false" not in podman.calls[0]
    assert "--tls-verify=false" not in podman.calls[1]


def test_podman_run_replaces_non_utf8_output() -> None:
    podman = PodmanCli(prefix=())

    result = podman._run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stdout.buffer.write(b'out\\xeb'); "
                "sys.stderr.buffer.write(b'err\\xf6')"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "out\ufffd"
    assert result.stderr == "err\ufffd"


def test_podman_image_identity_reads_immutable_fields() -> None:
    @dataclass
    class InspectPodman(PodmanCli):
        def _run(
            self,
            argv: list[str],
            *,
            timeout: float | None = None,
            check: bool = True,
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "Id": "sha256:id",
                            "Digest": "sha256:digest",
                            "RepoDigests": ["example/image@sha256:digest"],
                        }
                    ]
                ),
                stderr="",
            )

    assert InspectPodman(prefix=("podman",)).image_identity("example/image:tag") == {
        "reference": "example/image:tag",
        "id": "sha256:id",
        "digest": "sha256:digest",
        "repo_digests": ["example/image@sha256:digest"],
    }


def test_start_always_resets_to_base_commit(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={})
    reset = [
        c for c in podman.exec_calls if "git reset --hard abc123" in " ".join(c["cmd"])
    ]
    assert len(reset) == 1
    assert reset[0]["workdir"] == "/app"


def test_start_retries_git_history_hardening_once(tmp_path: Path) -> None:
    class HardenFailsOncePodman(FakePodmanCli):
        failed_once: bool = False

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
            joined = " ".join(cmd)
            if "git history hardened" in joined and not self.failed_once:
                self.failed_once = True
                self.exec_calls.append(
                    {
                        "container": container,
                        "cmd": cmd,
                        "user": user,
                        "workdir": workdir,
                    }
                )
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=255,
                    stdout="",
                    stderr="podman exec transport failed",
                )
            return super().exec(
                container,
                cmd=cmd,
                user=user,
                workdir=workdir,
                timeout=timeout,
                check=check,
            )

    podman = HardenFailsOncePodman()
    runner = ContainerRunner(
        task=_task(),
        artifacts_dir=tmp_path / "artifacts",
        podman=podman,
    )

    runner.start(env={})

    harden_calls = [
        c for c in podman.exec_calls if "git history hardened" in " ".join(c["cmd"])
    ]
    validate_calls = [
        c
        for c in podman.exec_calls
        if "git rev-parse --is-inside-work-tree" in " ".join(c["cmd"])
    ]
    assert len(harden_calls) == 2
    assert len(validate_calls) == 1


def test_start_raises_when_git_history_hardening_fails_twice(
    tmp_path: Path,
) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git history hardened"] = subprocess.CompletedProcess(
        args=[],
        returncode=255,
        stdout="",
        stderr="podman exec transport failed",
    )

    with pytest.raises(ContainerError, match="git-history hardening failed"):
        runner.start(env={})

    harden_calls = [
        c for c in podman.exec_calls if "git history hardened" in " ".join(c["cmd"])
    ]
    assert len(harden_calls) == 2


def test_start_raises_when_git_validation_fails_after_hardening(
    tmp_path: Path,
) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git rev-parse --is-inside-work-tree"] = (
        subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="fatal: Unable to create '/app/.git/index.lock': No such file",
        )
    )

    with pytest.raises(ContainerError, match="git repository validation failed"):
        runner.start(env={})


def test_start_raises_when_already_started(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    runner.start(env={})
    with pytest.raises(ContainerError, match="already started"):
        runner.start(env={})


def test_extract_patch_writes_file_and_returns_text(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git diff HEAD"] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="diff --git a/x b/x\n+content\n", stderr=""
    )
    runner.start(env={})
    patch = runner.extract_patch()
    assert patch.startswith("diff --git")
    written = (runner.artifacts_dir / "patch.diff").read_text()
    assert written == patch


def test_extract_patch_handles_empty_diff(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git diff HEAD"] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    runner.start(env={})
    assert runner.extract_patch() == ""
    assert (runner.artifacts_dir / "patch.diff").read_text() == ""


def test_extract_patch_retries_after_stale_git_index_lock(tmp_path: Path) -> None:
    class IndexLockOncePodman(FakePodmanCli):
        failed_once: bool = False

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
            joined = " ".join(cmd)
            if "git diff HEAD" in joined and not self.failed_once:
                self.failed_once = True
                self.exec_calls.append(
                    {
                        "container": container,
                        "cmd": cmd,
                        "user": user,
                        "workdir": workdir,
                    }
                )
                raise ContainerError(
                    "podman exited 128: git add -N . && git diff HEAD\n"
                    "stderr=\"fatal: Unable to create "
                    "'/app/.git/index.lock': File exists.\\n\""
                )
            return super().exec(
                container,
                cmd=cmd,
                user=user,
                workdir=workdir,
                timeout=timeout,
                check=check,
            )

    podman = IndexLockOncePodman()
    runner = ContainerRunner(
        task=_task(),
        artifacts_dir=tmp_path / "artifacts",
        podman=podman,
    )
    podman.exec_returns["git diff HEAD"] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="diff --git a/x b/x\n+after lock\n", stderr=""
    )

    runner.start(env={})
    patch = runner.extract_patch()

    assert patch.startswith("diff --git")
    diff_calls = [c for c in podman.exec_calls if "git diff HEAD" in " ".join(c["cmd"])]
    cleanup_calls = [
        c for c in podman.exec_calls if "rm -f .git/index.lock" in " ".join(c["cmd"])
    ]
    assert len(diff_calls) == 2
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0]["workdir"] == "/app"
    assert cleanup_calls[0]["user"] is None
    assert (runner.artifacts_dir / "patch.diff").read_text() == patch


def test_extract_patch_keeps_git_index_lock_when_git_is_alive(
    tmp_path: Path,
) -> None:
    class IndexLockPodman(FakePodmanCli):
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
            joined = " ".join(cmd)
            if "git diff HEAD" in joined:
                self.exec_calls.append(
                    {
                        "container": container,
                        "cmd": cmd,
                        "user": user,
                        "workdir": workdir,
                    }
                )
                raise ContainerError(
                    "podman exited 128: git add -N . && git diff HEAD\n"
                    "stderr=\"fatal: Unable to create "
                    "'/app/.git/index.lock': File exists.\\n\""
                )
            return super().exec(
                container,
                cmd=cmd,
                user=user,
                workdir=workdir,
                timeout=timeout,
                check=check,
            )

    podman = IndexLockPodman()
    runner = ContainerRunner(
        task=_task(),
        artifacts_dir=tmp_path / "artifacts",
        podman=podman,
    )
    podman.exec_returns["rm -f .git/index.lock"] = subprocess.CompletedProcess(
        args=[], returncode=42, stdout="", stderr=""
    )

    runner.start(env={})
    with pytest.raises(ContainerError, match=r"index\.lock"):
        runner.extract_patch()

    diff_calls = [c for c in podman.exec_calls if "git diff HEAD" in " ".join(c["cmd"])]
    cleanup_calls = [
        c for c in podman.exec_calls if "rm -f .git/index.lock" in " ".join(c["cmd"])
    ]
    assert len(diff_calls) == 1
    assert len(cleanup_calls) == 1
    assert not (runner.artifacts_dir / "patch.diff").exists()


def test_extract_patch_ignores_stale_sidecar_and_reads_live_tree(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git diff HEAD"] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="diff --git a/live b/live\n+live\n", stderr=""
    )
    runner.start(env={})
    sidecar = runner.artifacts_dir / "patch.diff.in-container"
    sidecar.write_text("diff --git a/stale b/stale\n+stale\n")

    patch = runner.extract_patch()

    assert patch == "diff --git a/live b/live\n+live\n"
    assert any("git diff HEAD" in " ".join(c["cmd"]) for c in podman.exec_calls)
    assert (runner.artifacts_dir / "patch.diff").read_text() == patch


def test_extract_patch_restarts_stopped_container_once(tmp_path: Path) -> None:
    class FlakyPodman(FakePodmanCli):
        failed_once: bool = False

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
            joined = " ".join(cmd)
            if "git diff HEAD" in joined and not self.failed_once:
                self.failed_once = True
                self.exec_calls.append(
                    {
                        "container": container,
                        "cmd": cmd,
                        "user": user,
                        "workdir": workdir,
                    }
                )
                raise ContainerError(
                    "Error: can only create exec sessions on running containers: "
                    "container state improper"
                )
            return super().exec(
                container,
                cmd=cmd,
                user=user,
                workdir=workdir,
                timeout=timeout,
                check=check,
            )

    podman = FlakyPodman()
    runner = ContainerRunner(
        task=_task(),
        artifacts_dir=tmp_path / "artifacts",
        podman=podman,
    )
    podman.exec_returns["git diff HEAD"] = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="diff --git a/x b/x\n+after restart\n", stderr=""
    )

    runner.start(env={})
    patch = runner.extract_patch()

    assert patch.startswith("diff --git")
    assert podman.start_calls == [runner.container_name]
    diff_calls = [c for c in podman.exec_calls if "git diff HEAD" in " ".join(c["cmd"])]
    assert len(diff_calls) == 2
    assert (runner.artifacts_dir / "patch.diff").read_text() == patch


def test_extract_patch_runs_in_app_as_agent(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={})
    runner.extract_patch()
    diff_call = next(
        c for c in podman.exec_calls if "git diff HEAD" in " ".join(c["cmd"])
    )
    assert diff_call["workdir"] == "/app"
    assert diff_call["user"] is None


def test_extract_patch_raises_when_not_started(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    with pytest.raises(ContainerError, match="un-started"):
        runner.extract_patch()


def test_stop_calls_podman_rm(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={})
    runner.stop()
    assert podman.stop_calls == [runner.container_name]


def test_stop_is_idempotent_when_never_started(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.stop()  # should not raise, should not call podman
    assert podman.stop_calls == []


def test_context_manager_stops_on_exit(tmp_path: Path) -> None:
    podman = FakePodmanCli()
    with ContainerRunner(
        task=_task(),
        artifacts_dir=tmp_path / "art",
        podman=podman,
    ) as runner:
        runner.start(env={})
    assert podman.stop_calls == [runner.container_name]


def test_reset_to_base_commit_failure_raises(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    podman.exec_returns["git reset --hard abc123"] = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="error"
    )
    with pytest.raises(ContainerError, match="reset-to-base failed"):
        runner.start(env={})


def test_exec_in_container_defaults_to_agent_user_in_app(tmp_path: Path) -> None:
    runner, podman = _runner(tmp_path)
    runner.start(env={})
    runner.exec_in_container(["ls", "-la"])
    ls_call = next(c for c in podman.exec_calls if c["cmd"] == ["ls", "-la"])
    assert ls_call["user"] is None
    assert ls_call["workdir"] == "/app"
