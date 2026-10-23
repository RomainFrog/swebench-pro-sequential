"""mini-swe-agent environment backed by a running Podman task container."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from ..runner import ContainerError, ContainerRunner

DEFAULT_WORKDIR = "/app"
_TIMEOUT_KILL_AFTER_SECONDS = 5
_PODMAN_EXEC_TIMEOUT_GRACE_SECONDS = 15
_TIMEOUT_RETURN_CODES = {124, 137, 143}
_MAX_AGENT_OBSERVATION_CHARS = 10_000
_OBSERVATION_SLICE_CHARS = 4_000


class AgentDeadlineExceeded(TimeoutError):
    """Raised when an in-flight model or shell action exhausts a task budget."""


class DeadlineModel:
    """Bound each model request by the remaining task wall-time."""

    def __init__(self, model: Any, deadline: float) -> None:
        self._model = model
        self.deadline = deadline

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AgentDeadlineExceeded("agent exceeded wall-time budget")
        configured_timeout = kwargs.get("timeout")
        if isinstance(configured_timeout, (int, float)) and configured_timeout > 0:
            remaining = min(remaining, float(configured_timeout))
        kwargs["timeout"] = remaining
        try:
            return cast("dict[str, Any]", self._model.query(messages, **kwargs))
        except Exception as exc:
            if time.monotonic() >= self.deadline:
                raise AgentDeadlineExceeded("agent exceeded wall-time budget") from exc
            raise


@dataclass
class ContainerBackedEnvironmentConfig:
    cwd: str = DEFAULT_WORKDIR
    env: dict[str, str] = field(default_factory=dict)
    forward_env: list[str] = field(default_factory=list)
    timeout: int = 60


class ContainerBackedEnvironment:
    """mini-swe-agent Environment that executes commands via ContainerRunner."""

    def __init__(
        self,
        runner: ContainerRunner,
        *,
        deadline: float | None = None,
        config_class: type = ContainerBackedEnvironmentConfig,
        **kwargs: Any,
    ) -> None:
        self.runner = runner
        self.deadline = deadline
        self.config = config_class(**kwargs)

    def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        effective_timeout = float(timeout or self.config.timeout)
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise AgentDeadlineExceeded("agent exceeded wall-time budget")
            effective_timeout = min(effective_timeout, remaining)
        wrapped_command = _wrap_command_for_container(
            command,
            env=_merged_environment(self.config.forward_env, self.config.env),
            timeout_seconds=effective_timeout,
        )
        try:
            result = self.runner.exec_in_container(
                ["bash", "-c", wrapped_command],
                user=None,
                workdir=cwd or self.config.cwd,
                timeout=effective_timeout + _PODMAN_EXEC_TIMEOUT_GRACE_SECONDS,
                check=False,
            )
        except ContainerError as exc:
            if _is_podman_exec_timeout(exc):
                raise _timeout_expired(command, effective_timeout, output=str(exc)) from exc
            raise
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode in _TIMEOUT_RETURN_CODES:
            raise _timeout_expired(command, effective_timeout, output=output)
        return {"output": bound_agent_observation(output), "returncode": result.returncode}

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)


def _merged_environment(
    forward_env: list[str],
    configured_env: dict[str, str],
) -> dict[str, str]:
    merged = {
        key: value
        for key in forward_env
        if (value := os.environ.get(key)) is not None
    }
    merged.update(configured_env)
    return merged


def _wrap_command_for_container(
    command: str,
    *,
    env: dict[str, str],
    timeout_seconds: float,
) -> str:
    exports = ""
    if env:
        assignments = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in env.items()
        )
        exports = f"export {assignments}; "
    return (
        f"{exports}timeout -k {_TIMEOUT_KILL_AFTER_SECONDS} "
        f"{timeout_seconds:g} bash -c {shlex.quote(command)}"
    )


def _is_podman_exec_timeout(exc: ContainerError) -> bool:
    return "podman timed out:" in str(exc)


def bound_agent_observation(output: str) -> str:
    """Keep shell observations small before they can enter agent context."""

    if len(output) <= _MAX_AGENT_OBSERVATION_CHARS:
        return output

    submit_prefix = ""
    for sentinel in (
        "MINI_SWE_AGENT_FINAL_OUTPUT",
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ):
        if output.startswith(sentinel):
            first_line, _, rest = output.partition("\n")
            submit_prefix = first_line + "\n"
            output = rest
            break

    head = output[:_OBSERVATION_SLICE_CHARS]
    tail = output[-_OBSERVATION_SLICE_CHARS:]
    omitted = len(output) - (len(head) + len(tail))
    return (
        submit_prefix
        + "<warning>\n"
        "Command output was truncated by swebench-seq before it entered "
        "the live agent context.\n"
        f"Original output length: {len(output)} characters; "
        f"{omitted} middle characters omitted.\n"
        "</warning>\n"
        "<output_head>\n"
        f"{head}\n"
        "</output_head>\n"
        "<output_tail>\n"
        f"{tail}\n"
        "</output_tail>"
    )


def _timeout_expired(
    command: str,
    timeout_seconds: float,
    *,
    output: str,
) -> subprocess.TimeoutExpired:
    bounded_output = bound_agent_observation(output)
    return subprocess.TimeoutExpired(
        cmd=command,
        timeout=timeout_seconds,
        output=bounded_output.encode("utf-8", errors="replace"),
    )


def deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def deadline_result(phase: str) -> tuple[str, str]:
    return "AgentTimeout", f"{phase} exceeded wall-time budget"
