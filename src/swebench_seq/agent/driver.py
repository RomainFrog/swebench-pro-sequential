"""mini-swe-agent integration for SWE-bench Pro task containers.

This driver deliberately reuses :class:`ContainerRunner` for the benchmark
container lifecycle. Upstream mini-swe-agent's own SWE-bench runner is aimed at
standard SWE-bench images; SWE-bench Pro needs our existing image prefix,
entrypoint workaround, base-commit reset, and fix-commit history hardening.
"""

from __future__ import annotations

import copy
import json
import logging
import shlex
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ..redaction import redact_secrets
from ..runner import ContainerRunner
from ..types import RunResult, Task
from .compaction import (
    DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_KEEP_RECENT_TOKENS,
    MINI_SUMMARY_END,
    MINI_SUMMARY_MARKER,
    POST_TASK_COMPACTION_TRANSITION,
    TURN_PREFIX_REQUIRED_HEADERS,
    bound_text,
    coalesce_consecutive_messages,
    compaction_repair_prompt,
    compaction_system_prompt,
    estimate_message_tokens,
    extract_file_ops,
    fallback_compaction_summary,
    fallback_turn_prefix_summary,
    format_file_operations,
    history_compaction_messages,
    latest_compaction_summary,
    locate_current_task_index,
    messages_to_transcript,
    split_messages_for_compaction,
    turn_prefix_compaction_messages,
    validate_compaction_summary,
    without_compaction_summary_messages,
)
from .config import (
    MiniSweAgentOptions,
    build_mini_config,
    load_mini_imports,
    prepare_config,
    prepare_sequential_config,
    redact_json_file,
)
from .environment import (
    AgentDeadlineExceeded,
    ContainerBackedEnvironment,
    DeadlineModel,
    bound_agent_observation,
    deadline_expired,
    deadline_result,
)
from .providers import wrap_model_for_provider

log = logging.getLogger(__name__)

_AGENT_HOME = "/home/agent"
_MEMORY_CONTEXT_HEADER = """\
<sequential_memory>
The following memory file is carried from earlier tasks in this repository.
"""
_MEMORY_CONTEXT_FOOTER = "</sequential_memory>"
_MEMORY_FILE_DISPLAY_PATH = "/home/agent/MEMORY.md"
_MEMORY_FILE_TAG = f'<file path="{_MEMORY_FILE_DISPLAY_PATH}">'
# Cap the MEMORY.md injected into the system prompt. Generous (normal files are
# 1-5 KB) but prevents a runaway MEMORY.md from bloating the live context.
_MAX_MEMORY_INJECTION_CHARS = 16_000
_TASK_BOUNDARY_PREAMBLE = (
    "The repository has advanced to a new commit. "
    "Continue with the next task below."
)


@dataclass(frozen=True)
class MiniUsage:
    """Token accounting from a mini-swe-agent model response."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MiniConversationSession:
    """Resumable mini-swe-agent conversation for sequential task runs.

    Upstream mini-swe-agent deliberately keeps a linear ``messages`` list, but
    ``DefaultAgent.run()`` always resets it. This session owns that list and
    drives the same query/execute loop directly so later tasks can continue the
    existing conversation.
    """

    def __init__(
        self,
        *,
        options: MiniSweAgentOptions,
        state_dir: Path,
        agent_home_dir: Path,
        context_window: int = 200_000,
        reserve_tokens: int = DEFAULT_COMPACTION_RESERVE_TOKENS,
        keep_recent_messages: int = 24,
        keep_recent_tokens: int | None = None,
    ) -> None:
        self.options = options
        self.state_dir = state_dir
        self.agent_home_dir = agent_home_dir
        self.context_window = context_window
        self.reserve_tokens = reserve_tokens
        # Keep the most recent messages after compaction, bounded by BOTH a
        # message count and a token budget. The token budget makes the
        # post-compaction size predictable regardless of how large individual
        # recent messages (e.g. big tool outputs) happen to be.
        self.keep_recent_messages = keep_recent_messages
        self.keep_recent_tokens = (
            keep_recent_tokens
            if keep_recent_tokens is not None
            else DEFAULT_KEEP_RECENT_TOKENS
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.imports = load_mini_imports()
        self.config = prepare_sequential_config(
            build_mini_config(self.imports, options),
            options,
        )
        self.messages: list[dict[str, Any]] = _load_saved_messages(
            self.state_dir / "messages.json"
        )
        self._agent: Any | None = None
        self._current_task_start_index: int | None = None
        self._last_observed_input_tokens: int | None = None
        self._current_task_usage: list[MiniUsage] = []
        self._in_compaction = False
        self._previous_compaction_summary = latest_compaction_summary(self.messages)

    def run_task(
        self,
        *,
        runner: ContainerRunner,
        task: Task,
        artifacts_dir: Path,
        timeout_seconds: float | None = None,
    ) -> RunResult:
        """Run one task as a continuation of this mini conversation."""

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._current_task_usage = []
        runner.exec_in_container(
            ["bash", "-lc", f"mkdir -p {shlex.quote(_AGENT_HOME)}"],
            user=None,
            workdir="/",
            timeout=30,
            check=False,
        )

        started = time.monotonic()
        deadline = started + timeout_seconds if timeout_seconds is not None else None
        agent = self._attach_runner(runner, deadline=deadline)
        (artifacts_dir / "mini_config.yaml").write_text(
            yaml.safe_dump(redact_secrets(self.config), sort_keys=False)
        )

        task_text = self._build_task_text(task)
        (artifacts_dir / "mini_task.txt").write_text(task_text)
        self._append_task_message(agent, task_text)
        self._save_state()

        exit_status = ""
        result_text = ""
        extra_info = None
        model = agent.model
        try:
            exit_status, result_text = self._run_until_terminal(
                agent,
                deadline=deadline,
            )
        except Exception as e:
            exit_status = type(e).__name__
            result_text = str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            self.messages = agent.messages
            self._save_state()
            traj_path = artifacts_dir / f"{task.instance_id}.mini.traj.json"
            self.imports["save_traj"](
                agent,
                traj_path,
                exit_status=exit_status,
                result=result_text,
                extra_info=extra_info,
                instance_id=task.instance_id,
                print_path=False,
            )
            redact_json_file(traj_path)

        patch = runner.extract_patch()
        wall = time.monotonic() - started
        error = None if exit_status == "Submitted" else exit_status
        usage = _sum_mini_usage(self._current_task_usage)
        return RunResult(
            patch=patch,
            turns_used=int(getattr(model, "n_calls", 0)),
            wall_seconds=wall,
            error=error,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    def compact_completed_history(self, *, reason: str = "after_task") -> bool:
        """Force a strict task-boundary compaction before the next task starts."""

        if self._agent is None:
            return False
        self._current_task_start_index = None
        return self._compact_now(
            self._agent,
            reason=reason,
            force=True,
            keep_recent_messages=0,
            keep_recent_tokens=0,
            min_recent_messages=0,
            append_transition_message=True,
        )

    def _attach_runner(
        self, runner: ContainerRunner, *, deadline: float | None = None
    ) -> Any:
        env_cfg = copy.deepcopy(self.config.get("environment", {}))
        env_cfg["cwd"] = self.options.workdir
        mini_env = ContainerBackedEnvironment(runner, deadline=deadline, **env_cfg)
        model = self._new_model()
        if deadline is not None:
            model = DeadlineModel(model, deadline)
        DefaultAgent = self.imports["DefaultAgent"]
        if self._agent is None:
            self._agent = DefaultAgent(model, mini_env, **self.config.get("agent", {}))
            self._agent.messages = self.messages
        else:
            self._agent.model = model
            self._agent.env = mini_env
            self._agent.messages = self.messages
        return self._agent

    def _new_model(self) -> Any:
        model = self.imports["get_model"](config=self.config.get("model", {}))
        return wrap_model_for_provider(model, self.options.provider)

    def _build_task_text(self, task: Task) -> str:
        chunks: list[str] = []
        if self.messages:
            chunks.append(_TASK_BOUNDARY_PREAMBLE)
        chunks.append(_format_system_prompt_mode_task_text(task))
        return "\n\n".join(chunks)

    def _append_task_message(self, agent: Any, task_text: str) -> None:
        # At task boundaries, compact old work before appending the new task.
        self._current_task_start_index = None
        self._compact_if_needed(agent, reason="before_task")

        if not agent.messages:
            agent.add_message(
                "system",
                self._render_system_prompt(agent),
            )
        self._current_task_start_index = len(agent.messages)
        agent.add_message("user", task_text)
        self.messages = agent.messages

    def _render_system_prompt(self, agent: Any) -> str:
        base = agent.render_template(agent.config.system_template)
        return _inject_system_memory_context(
            base,
            render_agent_home_memory_context(self.agent_home_dir),
        )

    def _refresh_system_prompt_memory(self, agent: Any) -> None:
        system_prompt = self._render_system_prompt(agent)
        for message in agent.messages:
            if message.get("role") == "system":
                message["content"] = system_prompt
                return
        agent.messages.insert(0, {"role": "system", "content": system_prompt})

    def _run_until_terminal(
        self,
        agent: Any,
        *,
        phase: str = "agent",
        deadline: float | None = None,
    ) -> tuple[str, str]:
        NonTerminatingException = self.imports["NonTerminatingException"]
        TerminatingException = self.imports["TerminatingException"]
        while True:
            if deadline_expired(deadline):
                return deadline_result(phase)
            try:
                self._step(agent, phase=phase)
            except AgentDeadlineExceeded:
                return deadline_result(phase)
            except NonTerminatingException as e:
                agent.add_message("user", bound_agent_observation(str(e)))
                self._save_state()
            except TerminatingException as e:
                exit_status = type(e).__name__
                result_text = str(e)
                self._save_state()
                return exit_status, result_text
            except Exception:
                if deadline_expired(deadline):
                    return deadline_result(phase)
                raise

    def _step(self, agent: Any, *, phase: str = "agent") -> dict[str, Any]:
        self._compact_if_needed(agent, reason="pre_query")
        try:
            response = agent.query()
        except Exception as e:
            if not _is_context_window_error(e):
                raise
            self._compact_now(agent, reason="context_window_error", force=True)
            response = agent.query()
        self._record_usage_from_response(
            response,
            phase=phase,
            estimated_input_tokens=estimate_message_tokens(agent.messages),
        )
        self._save_state()
        return cast("dict[str, Any]", agent.get_observation(response))

    def _compact_if_needed(self, agent: Any, *, reason: str) -> bool:
        if self.context_window <= 0:
            return False
        threshold = max(1, self.context_window - max(0, self.reserve_tokens))
        estimated = estimate_message_tokens(agent.messages)
        observed = self._last_observed_input_tokens or 0
        if estimated < threshold and observed < threshold:
            return False
        return self._compact_now(agent, reason=reason, force=False)

    def _compact_now(
        self,
        agent: Any,
        *,
        reason: str,
        force: bool,
        keep_recent_messages: int | None = None,
        keep_recent_tokens: int | None = None,
        min_recent_messages: int = 1,
        append_transition_message: bool = False,
    ) -> bool:
        if self._in_compaction:
            return False
        actual_keep_recent_messages = (
            self.keep_recent_messages
            if keep_recent_messages is None
            else keep_recent_messages
        )
        actual_keep_recent_tokens = (
            self.keep_recent_tokens if keep_recent_tokens is None else keep_recent_tokens
        )
        split = split_messages_for_compaction(
            agent.messages,
            current_task_start_index=self._current_task_start_index,
            keep_recent_messages=actual_keep_recent_messages,
            keep_recent_tokens=actual_keep_recent_tokens,
            min_recent_messages=min_recent_messages,
        )
        if not split["compact"]:
            return False

        self._in_compaction = True
        before_tokens = estimate_message_tokens(agent.messages)
        compaction_id = self._next_compaction_id()
        transcript = messages_to_transcript(
            split["compact"],
            max_chars=max(20_000, self.context_window * 3),
        )
        method = "model_summary"
        error = None
        summary_retry_count = 0
        summary_validation_error = None
        turn_prefix_error = None
        try:
            (
                summary,
                summary_retry_count,
                summary_validation_error,
                turn_prefix_error,
            ) = self._generate_compaction_summary(split)
        except Exception as e:
            method = "deterministic_fallback"
            error = f"{type(e).__name__}: {e}"
            summary = fallback_compaction_summary(
                split["compact"],
                previous_summary=self._previous_compaction_summary,
            )
        finally:
            self._in_compaction = False
        self._previous_compaction_summary = summary.strip()

        summary_message = {
            "role": "user",
            "content": (
                f"{MINI_SUMMARY_MARKER}\n"
                f"{summary.strip()}\n"
                f"{MINI_SUMMARY_END}"
            ),
        }
        new_messages: list[dict[str, Any]] = []
        if split["system"] is not None:
            new_messages.append(split["system"])
        new_messages.append(summary_message)
        new_messages.extend(split["keep"])
        if (
            append_transition_message
            and new_messages
            and new_messages[-1].get("role") == "user"
        ):
            new_messages.append(
                {"role": "assistant", "content": POST_TASK_COMPACTION_TRANSITION}
            )
        # The summary is a user turn; the kept window may also start with a user
        # turn (or the force-kept current-task message can sit across a compacted
        # gap), leaving two adjacent user turns. OpenAI tolerates that, but
        # Anthropic requires strict alternation — coalesce to keep the transcript
        # valid for every provider.
        new_messages = coalesce_consecutive_messages(new_messages)
        agent.messages = new_messages
        self._refresh_system_prompt_memory(agent)
        self.messages = agent.messages
        self._current_task_start_index = locate_current_task_index(
            agent.messages,
            split["current_task_message"],
        )
        after_tokens = estimate_message_tokens(agent.messages)
        audit_path = self._write_compaction_audit(
            compaction_id=compaction_id,
            reason=reason,
            method=method,
            force=force,
            error=error,
            split=split,
            transcript=transcript,
            summary=summary,
            summary_message=summary_message,
            messages_after=agent.messages,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary_retry_count=summary_retry_count,
            summary_validation_error=summary_validation_error,
            turn_prefix_error=turn_prefix_error,
        )
        self._append_jsonl(
            self.state_dir / "compactions.jsonl",
            {
                "ts": time.time(),
                "compaction_id": compaction_id,
                "reason": reason,
                "method": method,
                "force": force,
                "messages_before": len(split["all"]),
                "messages_after": len(agent.messages),
                "estimated_tokens_before": before_tokens,
                "estimated_tokens_after": after_tokens,
                "audit_path": str(audit_path.relative_to(self.state_dir)),
                "summary_retry_count": summary_retry_count,
                **({"error": error} if error else {}),
            },
        )
        self._last_observed_input_tokens = None
        self._save_state()
        return True

    def _generate_compaction_summary(
        self,
        split: dict[str, Any],
    ) -> tuple[str, int, str | None, str | None]:
        history_messages = without_compaction_summary_messages(
            split.get("history_compact", split["compact"])
        )
        previous_summary = self._previous_compaction_summary
        retry_count = 0
        validation_error = None
        turn_prefix_error = None

        if history_messages:
            transcript = messages_to_transcript(
                history_messages,
                max_chars=max(20_000, self.context_window * 3),
            )
            summary, retries, validation_error = self._summarize_history(
                transcript,
                previous_summary=previous_summary,
            )
            retry_count += retries
        elif previous_summary:
            summary = previous_summary
        else:
            summary = fallback_compaction_summary([])

        turn_prefix_messages = split.get("turn_prefix") or []
        if turn_prefix_messages:
            turn_summary, turn_prefix_error = self._summarize_turn_prefix(
                turn_prefix_messages
            )
            if turn_summary:
                summary = (
                    f"{summary.strip()}\n\n---\n\n"
                    f"**Turn Context (split turn):**\n\n{turn_summary.strip()}"
                )

        read_files, modified_files = extract_file_ops(split["compact"])
        summary = summary.strip() + format_file_operations(read_files, modified_files)

        final_validation_error = validate_compaction_summary(summary)
        if final_validation_error:
            repaired = self._repair_compaction_summary(summary, final_validation_error)
            retry_count += 1
            repaired_validation_error = validate_compaction_summary(repaired)
            validation_error = final_validation_error
            if repaired_validation_error:
                raise RuntimeError(
                    "compaction model returned invalid summary after repair: "
                    f"{repaired_validation_error}"
                )
            summary = repaired

        return summary, retry_count, validation_error, turn_prefix_error

    def _summarize_history(
        self,
        transcript: str,
        *,
        previous_summary: str | None,
    ) -> tuple[str, int, str | None]:
        response = self._query_compaction_model(
            history_compaction_messages(
                transcript,
                previous_summary=previous_summary,
            ),
            estimated_input_tokens=max(1, len(transcript) // 4),
        )
        summary = str(response.get("content") or "").strip()
        if not summary:
            raise RuntimeError("compaction model returned empty summary")

        validation_error = validate_compaction_summary(summary)
        if not validation_error:
            return summary, 0, None

        repaired = self._repair_compaction_summary(summary, validation_error)
        repaired_validation_error = validate_compaction_summary(repaired)
        if repaired_validation_error:
            raise RuntimeError(
                "compaction model returned invalid summary after repair: "
                f"{repaired_validation_error}"
            )
        return repaired, 1, validation_error

    def _summarize_turn_prefix(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        transcript = messages_to_transcript(
            messages,
            max_chars=max(10_000, self.keep_recent_tokens),
        )
        response = self._query_compaction_model(
            turn_prefix_compaction_messages(transcript),
            estimated_input_tokens=max(1, len(transcript) // 4),
        )
        summary = str(response.get("content") or "").strip()
        validation_error = validate_compaction_summary(
            summary,
            required_headers=TURN_PREFIX_REQUIRED_HEADERS,
        )
        if summary and not validation_error:
            return summary, None
        return fallback_turn_prefix_summary(messages), validation_error or "empty summary"

    def _repair_compaction_summary(
        self,
        candidate_summary: str,
        validation_error: str,
    ) -> str:
        bounded_candidate = bound_text(candidate_summary, max_chars=24_000)
        response = self._query_compaction_model(
            [
                {
                    "role": "system",
                    "content": compaction_system_prompt(),
                },
                {
                    "role": "user",
                    "content": compaction_repair_prompt(
                        validation_error=validation_error,
                        candidate_summary=bounded_candidate,
                    ),
                },
            ],
            estimated_input_tokens=max(1, len(bounded_candidate) // 4),
        )
        return str(response.get("content") or "").strip()

    def _query_compaction_model(
        self,
        messages: list[dict[str, Any]],
        *,
        estimated_input_tokens: int,
    ) -> dict[str, Any]:
        response = self._new_model().query(messages)
        self._record_usage_from_response(
            response,
            phase="compaction",
            estimated_input_tokens=estimated_input_tokens,
        )
        return cast("dict[str, Any]", response)

    def _next_compaction_id(self) -> str:
        compactions_dir = self.state_dir / "compactions"
        try:
            existing = sum(1 for _ in compactions_dir.glob("*.json"))
        except OSError:
            existing = 0
        return f"{existing + 1:06d}-{int(time.time() * 1000)}"

    def _write_compaction_audit(
        self,
        *,
        compaction_id: str,
        reason: str,
        method: str,
        force: bool,
        error: str | None,
        split: dict[str, Any],
        transcript: str,
        summary: str,
        summary_message: dict[str, Any],
        messages_after: list[dict[str, Any]],
        before_tokens: int,
        after_tokens: int,
        summary_retry_count: int = 0,
        summary_validation_error: str | None = None,
        turn_prefix_error: str | None = None,
    ) -> Path:
        compactions_dir = self.state_dir / "compactions"
        compactions_dir.mkdir(parents=True, exist_ok=True)
        path = compactions_dir / f"{compaction_id}.json"
        payload = {
            "schema_version": 1,
            "compaction_id": compaction_id,
            "ts": time.time(),
            "reason": reason,
            "method": method,
            "force": force,
            "context_window": self.context_window,
            "reserve_tokens": self.reserve_tokens,
            "keep_recent_messages": split.get(
                "keep_recent_messages", self.keep_recent_messages
            ),
            "keep_recent_tokens": split.get(
                "keep_recent_tokens", self.keep_recent_tokens
            ),
            "min_recent_messages": split.get("min_recent_messages", 1),
            "current_task_start_index_before": self._current_task_start_index,
            "messages_before": split["all"],
            "messages_compacted": split["compact"],
            "messages_history_compacted": split.get("history_compact", split["compact"]),
            "messages_turn_prefix_compacted": split.get("turn_prefix", []),
            "messages_kept": split["keep"],
            "transcript_sent_to_compactor": transcript,
            "summary": summary,
            "summary_retry_count": summary_retry_count,
            "summary_validation_error": summary_validation_error,
            "turn_prefix_error": turn_prefix_error,
            "summary_message": summary_message,
            "messages_after": messages_after,
            "estimated_tokens_before": before_tokens,
            "estimated_tokens_after": after_tokens,
            **({"error": error} if error else {}),
        }
        path.write_text(
            json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False)
        )
        return path

    def _record_usage_from_response(
        self,
        response: dict[str, Any],
        *,
        phase: str,
        estimated_input_tokens: int,
    ) -> None:
        usage = extract_mini_usage(response)
        if phase == "agent":
            self._current_task_usage.append(usage)
        if usage.input_tokens is not None:
            self._last_observed_input_tokens = usage.input_tokens
        self._append_jsonl(
            self.state_dir / "usage.jsonl",
            {
                "ts": time.time(),
                "phase": phase,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "estimated_input_tokens": estimated_input_tokens,
                "messages": len(self.messages),
            },
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(redact_secrets(payload), ensure_ascii=False) + "\n")

    def _save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "messages": redact_secrets(self.messages),
            "current_task_start_index": self._current_task_start_index,
        }
        (self.state_dir / "messages.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )


def run_mini_swe_agent_task(
    *,
    runner: ContainerRunner,
    task: Task,
    artifacts_dir: Path,
    options: MiniSweAgentOptions,
    timeout_seconds: float | None = None,
) -> RunResult:
    """Run mini-swe-agent against an already-started task container."""

    imports = load_mini_imports()
    DefaultAgent = imports["DefaultAgent"]
    get_model = imports["get_model"]
    save_traj = imports["save_traj"]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config = build_mini_config(imports, options)
    config = prepare_config(config, options)

    started = time.monotonic()
    deadline = started + timeout_seconds if timeout_seconds is not None else None
    env_cfg = copy.deepcopy(config.get("environment", {}))
    env_cfg["cwd"] = options.workdir
    mini_env = ContainerBackedEnvironment(runner, deadline=deadline, **env_cfg)
    model = wrap_model_for_provider(
        get_model(config=config.get("model", {})),
        options.provider,
    )
    if deadline is not None:
        model = DeadlineModel(model, deadline)
    agent = DefaultAgent(model, mini_env, **config.get("agent", {}))

    task_text = _format_mini_task_text(task)

    (artifacts_dir / "mini_config.yaml").write_text(
        yaml.safe_dump(redact_secrets(config), sort_keys=False)
    )
    (artifacts_dir / "mini_task.txt").write_text(task_text)

    exit_status = ""
    result_text = ""
    extra_info = None
    try:
        if deadline is not None:
            exit_status, result_text = _run_agent_with_deadline(
                agent,
                imports,
                task_text=task_text,
                deadline=deadline,
            )
        else:
            exit_status, result_text = agent.run(task_text)
    except Exception as e:
        exit_status = type(e).__name__
        result_text = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        traj_path = artifacts_dir / f"{task.instance_id}.mini.traj.json"
        save_traj(
            agent,
            traj_path,
            exit_status=exit_status,
            result=result_text,
            extra_info=extra_info,
            instance_id=task.instance_id,
            print_path=False,
        )
        redact_json_file(traj_path)

    patch = runner.extract_patch()
    wall = time.monotonic() - started
    error = None if exit_status == "Submitted" else exit_status
    usage = _sum_mini_usage(
        extract_mini_usage(message)
        for message in getattr(agent, "messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    return RunResult(
        patch=patch,
        turns_used=int(getattr(model, "n_calls", 0)),
        wall_seconds=wall,
        error=error,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def _run_agent_with_deadline(
    agent: Any,
    imports: dict[str, Any],
    *,
    task_text: str,
    deadline: float,
) -> tuple[str, str]:
    """Run a fresh mini agent while enforcing a task-level deadline."""

    NonTerminatingException = imports["NonTerminatingException"]
    TerminatingException = imports["TerminatingException"]
    agent.extra_template_vars |= {"task": task_text}
    agent.messages = []
    agent.add_message("system", agent.render_template(agent.config.system_template))
    agent.add_message("user", agent.render_template(agent.config.instance_template))

    while True:
        if deadline_expired(deadline):
            return deadline_result("agent")
        try:
            agent.step()
        except AgentDeadlineExceeded:
            return deadline_result("agent")
        except NonTerminatingException as exc:
            agent.add_message("user", bound_agent_observation(str(exc)))
        except TerminatingException as exc:
            agent.add_message("user", str(exc))
            return type(exc).__name__, str(exc)
        except Exception:
            if deadline_expired(deadline):
                return deadline_result("agent")
            raise


def _bound_memory_content(
    content: str, max_chars: int = _MAX_MEMORY_INJECTION_CHARS
) -> str:
    """Cap injected MEMORY.md so a runaway file can't bloat the system prompt."""
    if len(content) <= max_chars:
        return content
    head = content[:max_chars].rstrip()
    return (
        f"{head}\n\n[...truncated by swebench-seq: MEMORY.md is longer than "
        f"{max_chars} chars; keep it terse — only the head is shown here...]\n"
    )


def render_agent_home_memory_context(agent_home_dir: Path) -> str:
    """Render exact sequential memory files for the system prompt."""

    files: list[tuple[str, Path]] = []
    memory_md = agent_home_dir / "MEMORY.md"
    if memory_md.is_file() and not memory_md.is_symlink():
        files.append((_MEMORY_FILE_DISPLAY_PATH, memory_md))
    if not files:
        return ""

    parts = [_MEMORY_CONTEXT_HEADER.rstrip()]
    for display_path, path in files:
        try:
            content = _bound_memory_content(
                path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        suffix = "" if content.endswith("\n") else "\n"
        parts.append(
            f'<file path="{display_path}">\n'
            f"{content}"
            f"{suffix}"
            "</file>"
        )
    if len(parts) == 1:
        return ""
    parts.append(_MEMORY_CONTEXT_FOOTER)
    return "\n".join(parts)


def _inject_system_memory_context(system_prompt: str, memory_context: str) -> str:
    system_prompt = _strip_memory_context(system_prompt).rstrip()
    if not memory_context:
        return system_prompt
    marker = "\n\nSequential mode:"
    idx = system_prompt.find(marker)
    if idx < 0:
        return system_prompt + "\n\n" + memory_context.strip()
    return (
        system_prompt[:idx].rstrip()
        + "\n\n"
        + memory_context.strip()
        + system_prompt[idx:]
    )


def _strip_memory_context(text: str) -> str:
    start_token = "<sequential_memory>"
    end_token = "</sequential_memory>"
    start = text.find(start_token)
    if start < 0:
        return text
    end = text.find(end_token, start + len(start_token))
    if end < 0:
        return text
    before = text[:start].rstrip()
    after = text[end + len(end_token) :].lstrip()
    return (before + "\n\n" + after).strip()


def extract_mini_usage(response: dict[str, Any]) -> MiniUsage:
    """Extract token usage from common LiteLLM/OpenAI/Anthropic response shapes."""

    raw = response.get("extra", {}).get("response", {})
    if not isinstance(raw, dict):
        return MiniUsage()
    usage = raw.get("usage") or raw.get("response_metadata", {}).get("usage")
    if not isinstance(usage, dict):
        return MiniUsage()

    prompt_tokens = _first_int(usage, ("prompt_tokens",))
    native_input_tokens = _first_int(usage, ("input_tokens", "input"))
    input_token_details = usage.get("input_tokens_details")
    if not isinstance(input_token_details, dict):
        input_token_details = {}
    cache_creation = _first_int(
        usage,
        ("cache_creation_input_tokens", "cacheWrite"),
    )
    cache_read = _first_int(usage, ("cache_read_input_tokens", "cacheRead"))
    if cache_creation is None:
        cache_creation = _first_int(input_token_details, ("cache_write_tokens",))
    if cache_read is None:
        cache_read = _first_int(input_token_details, ("cached_tokens",))
    if prompt_tokens is not None:
        # LiteLLM's OpenAI-shaped `prompt_tokens` already includes cached
        # prompt tokens. The cache fields are a breakdown, not additional
        # context tokens, so do not add them again.
        input_tokens = prompt_tokens
    elif native_input_tokens is not None:
        if input_token_details:
            # Responses API input_tokens already includes its nested cache
            # breakdown. Adding cached_tokens here would double-count it.
            input_tokens = native_input_tokens
        else:
            # Anthropic-shaped input_tokens excludes its top-level prompt-cache
            # read/write fields, so add them to recover actual context length.
            input_tokens = (
                native_input_tokens
                + (cache_creation or 0)
                + (cache_read or 0)
            )
    elif cache_creation is not None or cache_read is not None:
        input_tokens = (cache_creation or 0) + (cache_read or 0)
    else:
        input_tokens = None
    output_tokens = _first_int(
        usage,
        ("completion_tokens", "output_tokens", "output"),
    )
    total_tokens = _first_int(usage, ("total_tokens", "totalTokens", "total"))
    return MiniUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def _sum_mini_usage(usages: Iterable[MiniUsage]) -> MiniUsage:
    values = list(usages)

    def total(field: str) -> int | None:
        known = [value for usage in values if (value := getattr(usage, field)) is not None]
        return sum(known) if known else None

    return MiniUsage(
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        total_tokens=total("total_tokens"),
        cache_creation_input_tokens=total("cache_creation_input_tokens"),
        cache_read_input_tokens=total("cache_read_input_tokens"),
    )


def _first_int(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _load_saved_messages(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return [m for m in data["messages"] if isinstance(m, dict)]
    return []


def _is_context_window_error(exc: BaseException) -> bool:
    for cls in type(exc).mro():
        if cls.__name__ == "ContextWindowExceededError":
            return True
    text = str(exc).lower()
    return (
        "context window" in text
        or "maximum context" in text
        or "context_length_exceeded" in text
        or "too many tokens" in text
    )


def _format_mini_task_text(task: Task) -> str:
    """Match SWE-bench Pro's upstream problem-statement helper for mini runs."""

    parts = [task.problem_statement]
    if task.requirements:
        parts.extend(["Requirements:", task.requirements])
    if task.interface:
        parts.extend(["New interfaces introduced:", task.interface])
    return "\n\n".join(parts)


def _format_system_prompt_mode_task_text(task: Task) -> str:
    task_text = _format_mini_task_text(task)
    return (
        "<pr_description>\n"
        "Consider the following PR description:\n"
        f"{task_text}\n"
        "</pr_description>"
    )
