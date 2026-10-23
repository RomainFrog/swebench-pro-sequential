"""Sequential-run artifact reuse and recovery state restoration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from ..agent.home import clear_agent_home, restore_agent_home
from ..tasks import TaskUniversePlan
from ..types import Task

TERMINAL_EXIT_STATUSES = frozenset({"Submitted", "LimitsExceeded"})


def existing_patch_is_reusable(artifacts_dir: Path, patch_path: Path) -> bool:
    """Return whether ``patch.diff`` represents a completed agent run."""

    if patch_path.stat().st_size > 0:
        return True
    status = mini_traj_exit_status(artifacts_dir)
    return status in TERMINAL_EXIT_STATUSES


def load_task_manifest_universes(
    manifest_path: Path,
    tasks: list[Task],
    *,
    repo: str | None,
) -> list[TaskUniversePlan] | None:
    """Restore the exact ordered task selection recorded for a prior run."""

    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise click.ClickException(f"failed to load {manifest_path}: {exc}") from exc
    repos = payload.get("repos") if isinstance(payload, dict) else None
    if not isinstance(repos, list):
        raise click.ClickException(f"{manifest_path} has no repos list")

    by_id = {task.instance_id: task for task in tasks}
    plans: list[TaskUniversePlan] = []
    for entry in repos:
        if not isinstance(entry, dict):
            raise click.ClickException(f"{manifest_path} contains invalid repo entry")
        repo_name = entry.get("repo")
        instance_ids = entry.get("instance_ids")
        if not isinstance(repo_name, str) or not isinstance(instance_ids, list):
            raise click.ClickException(f"{manifest_path} contains invalid repo entry")
        if repo is not None and repo_name != repo:
            continue
        selected_tasks: list[Task] = []
        missing: list[str] = []
        for instance_id in instance_ids:
            if not isinstance(instance_id, str):
                raise click.ClickException(
                    f"{manifest_path} contains non-string instance id"
                )
            task = by_id.get(instance_id)
            if task is None:
                missing.append(instance_id)
            elif task.repo != repo_name:
                raise click.ClickException(
                    f"{manifest_path} maps {instance_id} to repo={repo_name!r}, "
                    f"but dataset has repo={task.repo!r}"
                )
            else:
                selected_tasks.append(task)
        if missing:
            raise click.ClickException(
                f"{manifest_path} references missing task(s): {missing[:3]}"
            )
        if not selected_tasks:
            continue
        available_task_count = entry.get("available_task_count")
        if not isinstance(available_task_count, int):
            available_task_count = len(selected_tasks)
        order_source = entry.get("order_source")
        if not isinstance(order_source, str):
            order_source = "manifest"
        plans.append(
            TaskUniversePlan(
                repo=repo_name,
                tasks=tuple(selected_tasks),
                available_task_count=available_task_count,
                order_source=f"manifest:{order_source}",
            )
        )
    if repo is not None and not plans:
        raise click.ClickException(f"{manifest_path} has no tasks for repo={repo!r}")
    return plans


def mini_traj_path(artifacts_dir: Path, instance_id: str) -> Path:
    return artifacts_dir / f"{instance_id}.mini.traj.json"


def mini_traj_exit_status(
    artifacts_dir: Path,
    instance_id: str | None = None,
) -> str | None:
    """Read ``info.exit_status`` from a task's mini trajectory."""

    if instance_id is not None:
        trajectories = [mini_traj_path(artifacts_dir, instance_id)]
    else:
        trajectories = sorted(artifacts_dir.glob("*.mini.traj.json"))
    for trajectory in reversed(trajectories):
        try:
            data = json.loads(trajectory.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        info = data.get("info")
        if isinstance(info, dict):
            status = info.get("exit_status")
            if isinstance(status, str):
                return status
    return None


def first_retryable_task(
    *,
    run_root: Path,
    universe_repo: str,
    tasks: list[Task],
) -> tuple[int, str] | None:
    """Return the first missing or non-terminal task trajectory in a universe."""

    universe_dir = run_root / "universes" / universe_repo.replace("/", "__")
    for index, task in enumerate(tasks):
        artifacts_dir = universe_dir / task.instance_id
        status = mini_traj_exit_status(artifacts_dir, task.instance_id)
        if status not in TERMINAL_EXIT_STATUSES:
            return index, status or "missing"
    return None


def restore_sequential_recovery_state(
    *,
    run_root: Path,
    universe_repo: str,
    tasks: list[Task],
    start_index: int,
    agent_home_name: str,
    status: str,
) -> None:
    """Restore conversation and agent-home state immediately before a retry."""

    universe_dir = run_root / "universes" / universe_repo.replace("/", "__")
    state_dir = universe_dir / "_mini_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    if start_index <= 0:
        (state_dir / "messages.json").write_text(
            json.dumps({"messages": [], "current_task_start_index": None}, indent=2)
        )
        clear_agent_home(universe_dir / agent_home_name)
    else:
        previous = tasks[start_index - 1]
        previous_artifacts = universe_dir / previous.instance_id
        trajectory_path = mini_traj_path(previous_artifacts, previous.instance_id)
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise click.ClickException(
                f"cannot restore recovery state for run_root={run_root} "
                f"repo={universe_repo}: failed to load {trajectory_path}: {exc}"
            ) from exc
        messages = trajectory.get("messages")
        if not isinstance(messages, list):
            raise click.ClickException(
                f"cannot restore recovery state: {trajectory_path} has no messages list"
            )
        (state_dir / "messages.json").write_text(
            json.dumps(
                {"messages": messages, "current_task_start_index": None},
                indent=2,
                ensure_ascii=False,
            )
        )

        snapshot = previous_artifacts / "_agent_home_snapshot"
        if not snapshot.is_dir():
            raise click.ClickException(
                f"cannot restore recovery agent home: missing {snapshot}"
            )
        restore_agent_home(snapshot, universe_dir / agent_home_name)

    recovery = {
        "status": status,
        "run_root": str(run_root),
        "universe_repo": universe_repo,
        "start_index": start_index,
        "start_instance_id": tasks[start_index].instance_id,
        "agent_home_name": agent_home_name,
    }
    with (universe_dir / "sequential_recovery.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(recovery, ensure_ascii=False) + "\n")


def reset_stale_sequential_state_for_fresh_start(
    *,
    universe_dir: Path,
    tasks: list[Task],
    skip_existing_patches: bool,
    memory_seeded_initial: bool,
) -> None:
    """Drop stale conversation state when rebuilding a sequence from task zero."""

    if memory_seeded_initial or not tasks:
        return
    first_artifacts = universe_dir / tasks[0].instance_id
    first_patch = first_artifacts / "patch.diff"
    first_is_reusable = (
        skip_existing_patches
        and first_patch.is_file()
        and existing_patch_is_reusable(first_artifacts, first_patch)
    )
    if first_is_reusable:
        return
    state_dir = universe_dir / "_mini_state"
    if state_dir.exists():
        shutil.rmtree(state_dir)
