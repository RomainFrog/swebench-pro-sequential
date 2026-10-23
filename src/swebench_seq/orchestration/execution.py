"""Per-task and per-repository execution for benchmark runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from ..agent.config import MiniSweAgentOptions, with_repo_prompt_cache_key
from ..agent.driver import MiniConversationSession, run_mini_swe_agent_task
from ..agent.home import AgentHomePersistence, prune_agent_home, snapshot_agent_home
from ..constants import IMAGE_PREFIX_DEFAULT
from ..evaluation.runner import get_cached_eval_result, load_tag_lookup, score_one
from ..manifest import write_manifest
from ..predictions import PredictionsAccumulator
from ..runner import ContainerRunner
from ..types import Task
from .recovery import (
    TERMINAL_EXIT_STATUSES,
    existing_patch_is_reusable,
    mini_traj_exit_status,
    reset_stale_sequential_state_for_fresh_start,
)


@dataclass(frozen=True)
class TaskExecutionResult:
    """Whether a task ran now and whether its rollout reached a gradeable state."""

    executed: bool
    rollout_status: str | None


def run_completion_errors(
    *,
    run_root: Path,
    tasks: list[Task],
    sequential: bool,
    all_repos: bool,
) -> list[str]:
    """Return selected tasks missing a prediction or trustworthy score."""

    predictions_path = run_root / "predictions.json"
    try:
        raw_predictions = json.loads(predictions_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raw_predictions = {}
    predictions = raw_predictions if isinstance(raw_predictions, dict) else {}

    errors: list[str] = []
    for task in tasks:
        patch = predictions.get(task.instance_id)
        if not isinstance(patch, str):
            errors.append(f"run={run_root.name} task={task.instance_id}: missing prediction")
            continue
        if sequential:
            repo_dir = run_root / "universes" / task.repo.replace("/", "__")
        elif all_repos:
            repo_dir = run_root / task.repo.replace("/", "__")
        else:
            repo_dir = run_root
        artifacts_dir = repo_dir / task.instance_id
        status = mini_traj_exit_status(artifacts_dir, task.instance_id)
        if status not in TERMINAL_EXIT_STATUSES:
            errors.append(
                f"run={run_root.name} task={task.instance_id}: "
                f"rollout status {status or 'missing'}"
            )
            continue
        if not (artifacts_dir / "task_result.json").is_file():
            errors.append(
                f"run={run_root.name} task={task.instance_id}: missing task result"
            )
            continue
        try:
            image_identity = json.loads(
                (artifacts_dir / "container_image.json").read_text()
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            image_identity = None
        if not isinstance(image_identity, dict) or not image_identity.get("id"):
            errors.append(
                f"run={run_root.name} task={task.instance_id}: missing image identity"
            )
            continue
        if get_cached_eval_result(repo_dir, task.instance_id, patch) is None:
            errors.append(f"run={run_root.name} task={task.instance_id}: missing score")
    return errors


def grade_task(
    *,
    repo_dir: Path,
    instance_id: str,
    patch: str,
    image_prefix: str,
    tag_lookup: dict[str, str] | None,
    evaluator_dir: Path | None = None,
    dataset_path: Path | None = None,
) -> bool | None:
    """Grade one task with the public evaluator through rootless Podman."""

    return score_one(
        repo_dir=repo_dir,
        instance_id=instance_id,
        patch=patch,
        tag_lookup=tag_lookup,
        image_prefix=image_prefix,
        evaluator_dir=evaluator_dir,
        dataset_path=dataset_path,
    )


def run_universe_sequential(
    *,
    universe_repo: str,
    tasks: list[Task],
    output_dir: Path,
    env: dict[str, str],
    mini_options: MiniSweAgentOptions,
    accumulator: PredictionsAccumulator,
    skip_existing_patches: bool = False,
    overall_timeout: float | None = None,
    context_window: int = 200_000,
    compact_after_each_task: bool = False,
    run_id: int = 0,
    image_prefix: str = IMAGE_PREFIX_DEFAULT,
    evaluator_dir: Path | None = None,
    dataset_path: Path | None = None,
    memory_seeded_initial: bool = False,
    agent_home_name: str = "_agent_home",
) -> None:
    """Run one repository's tasks serially with shared conversation and home."""

    mini_options = with_repo_prompt_cache_key(mini_options, universe_repo)
    universe_dir = output_dir / "universes" / universe_repo.replace("/", "__")
    reset_stale_sequential_state_for_fresh_start(
        universe_dir=universe_dir,
        tasks=tasks,
        skip_existing_patches=skip_existing_patches,
        memory_seeded_initial=memory_seeded_initial,
    )
    agent_home_host = universe_dir / agent_home_name
    mini_session = MiniConversationSession(
        options=mini_options,
        state_dir=universe_dir / "_mini_state",
        agent_home_dir=agent_home_host,
        context_window=context_window,
    )

    tag_lookup = load_tag_lookup(dataset_path)
    memory_seeded = memory_seeded_initial
    for task_index, task in enumerate(tasks):
        is_first = not memory_seeded
        has_next_task = task_index < len(tasks) - 1
        artifacts_dir = universe_dir / task.instance_id
        agent_home_mount = None if is_first else agent_home_host

        task_execution = TaskExecutionResult(executed=False, rollout_status=None)
        outcome: bool | None = None
        try:
            task_execution = run_task(
                task=task,
                output_dir=output_dir,
                image_prefix=image_prefix,
                env=env,
                overall_timeout=overall_timeout,
                mini_options=mini_options,
                mini_session=mini_session,
                accumulator=accumulator,
                sequential=True,
                agent_home_mount=agent_home_mount,
                artifacts_dir_override=artifacts_dir,
                extract_home_to=(agent_home_host if is_first else None),
                skip_existing_patches=skip_existing_patches,
                run_id=run_id,
            )
        except Exception as exc:
            click.echo(
                f"[swebench-seq] run={run_id} task {task.instance_id} "
                f"raised: {exc!r} (skipped, universe continues)",
                err=True,
            )
            continue

        if task_execution.rollout_status not in TERMINAL_EXIT_STATUSES:
            click.echo(
                f"[swebench-seq] {task.instance_id} → score: INFRA-FAIL "
                f"(rollout status {task_execution.rollout_status or 'missing'})",
                err=True,
            )
        else:
            patch_path = artifacts_dir / "patch.diff"
            patch_text = patch_path.read_text() if patch_path.is_file() else ""
            cached = get_cached_eval_result(universe_dir, task.instance_id, patch_text)
            if cached is not None:
                outcome = cached
                click.echo(
                    f"[swebench-seq] {task.instance_id} → score: {outcome} (cached)"
                )
            else:
                try:
                    outcome = grade_task(
                        repo_dir=universe_dir,
                        instance_id=task.instance_id,
                        patch=patch_text,
                        image_prefix=image_prefix,
                        tag_lookup=tag_lookup,
                        evaluator_dir=evaluator_dir,
                        dataset_path=dataset_path,
                    )
                except Exception as exc:
                    click.echo(
                        f"[swebench-seq] {task.instance_id} → score raised: {exc}",
                        err=True,
                    )
                    outcome = None
                click.echo(
                    f"[swebench-seq] {task.instance_id} → score: "
                    f"{outcome if outcome is not None else 'INFRA-FAIL'}"
                )
        if is_first:
            memory_seeded = True

        snapshot_agent_home(
            src_dir=agent_home_host,
            dst_dir=artifacts_dir / "_agent_home_snapshot",
        )
        prune_agent_home(agent_home_host)

        if compact_after_each_task and task_execution.executed and has_next_task:
            compacted = mini_session.compact_completed_history(reason="after_task")
            click.echo(
                f"[swebench-seq] run={run_id} {task.instance_id} "
                f"→ after_task_compaction={'yes' if compacted else 'no'}"
            )


def run_task(
    *,
    task: Task,
    output_dir: Path,
    env: dict[str, str],
    mini_options: MiniSweAgentOptions,
    accumulator: PredictionsAccumulator,
    mini_session: MiniConversationSession | None = None,
    sequential: bool = False,
    agent_home_mount: Path | None = None,
    artifacts_dir_override: Path | None = None,
    extract_home_to: Path | None = None,
    skip_existing_patches: bool = False,
    nest_by_repo: bool = False,
    overall_timeout: float | None = None,
    image_prefix: str = IMAGE_PREFIX_DEFAULT,
    run_id: int = 0,
) -> TaskExecutionResult:
    """Run one agent task, persist its artifacts, and record its prediction."""

    if artifacts_dir_override is not None:
        artifacts_dir = artifacts_dir_override
    elif nest_by_repo:
        artifacts_dir = output_dir / task.repo.replace("/", "__") / task.instance_id
    else:
        artifacts_dir = output_dir / task.instance_id

    if skip_existing_patches:
        existing = artifacts_dir / "patch.diff"
        if existing.is_file() and existing_patch_is_reusable(artifacts_dir, existing):
            patch_text = existing.read_text()
            accumulator.record(task.instance_id, patch_text)
            click.echo(
                f"[swebench-seq] {task.instance_id} → SKIPPED (existing "
                f"patch {existing.stat().st_size}B)"
            )
            return TaskExecutionResult(
                executed=False,
                rollout_status=mini_traj_exit_status(artifacts_dir, task.instance_id),
            )
        if existing.is_file():
            status = mini_traj_exit_status(artifacts_dir) or "no trajectory"
            click.echo(
                f"[swebench-seq] {task.instance_id} → RERUN "
                f"(empty patch, prior exit_status={status})"
            )

    task_env = {
        key: value
        for key, value in env.items()
        if key.endswith("_PROXY") or key.endswith("_proxy")
    }
    runner_kwargs: dict[str, Any] = {
        "task": task,
        "artifacts_dir": artifacts_dir,
        "image_prefix": image_prefix,
        "agent_home_mount": agent_home_mount,
    }
    try:
        with ContainerRunner(**runner_kwargs) as runner:
            runner.start(env=task_env)
            if sequential and mini_session is not None:
                result = mini_session.run_task(
                    runner=runner,
                    task=task,
                    artifacts_dir=artifacts_dir,
                    timeout_seconds=overall_timeout,
                )
            else:
                result = run_mini_swe_agent_task(
                    runner=runner,
                    task=task,
                    artifacts_dir=artifacts_dir,
                    options=with_repo_prompt_cache_key(mini_options, task.repo),
                    timeout_seconds=overall_timeout,
                )
            if extract_home_to is not None:
                AgentHomePersistence(host_dir=extract_home_to, podman=runner.podman).extract(
                    runner.container_name,
                    fallback_dir=artifacts_dir / "agent-home-fallback",
                )
    except Exception as exc:
        raise click.ClickException(f"task {task.instance_id} failed: {exc}") from exc

    accumulator.record(task.instance_id, result.patch)
    rollout_status = result.error or "Submitted"
    write_manifest(
        artifacts_dir / "task_result.json",
        {
            "schema_version": 1,
            "instance_id": task.instance_id,
            "rollout_status": rollout_status,
            "model_calls": result.turns_used,
            "wall_seconds": result.wall_seconds,
            "patch_bytes": len(result.patch.encode("utf-8")),
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
            },
        },
    )
    click.echo(
        f"[swebench-seq] {task.instance_id} → "
        f"turns={result.turns_used} wall={result.wall_seconds:.1f}s "
        f"error={result.error or '-'} patch_bytes={len(result.patch)}"
    )
    return TaskExecutionResult(executed=True, rollout_status=rollout_status)
