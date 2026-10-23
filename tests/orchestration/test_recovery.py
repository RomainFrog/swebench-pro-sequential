"""Tests for mini-trajectory-aware sequential recovery and artifact reuse."""

from __future__ import annotations

import json
from pathlib import Path

from swebench_seq.orchestration.recovery import (
    existing_patch_is_reusable,
    first_retryable_task,
    mini_traj_exit_status,
    reset_stale_sequential_state_for_fresh_start,
)
from swebench_seq.types import Task


def _traj(artifacts: Path, exit_status: str) -> None:
    body = {"info": {"exit_status": exit_status}, "messages": []}
    (artifacts / "i.mini.traj.json").write_text(json.dumps(body))


def _patch(artifacts: Path, content: str) -> Path:
    p = artifacts / "patch.diff"
    p.write_text(content)
    return p


def test_nonempty_patch_always_reusable(tmp_path: Path) -> None:
    p = _patch(tmp_path, "diff --git a/x b/x\n+y\n")
    # even with a crashed trajectory, a real patch is reusable
    _traj(tmp_path, "RetryError")
    assert existing_patch_is_reusable(tmp_path, p) is True


def test_empty_patch_with_submitted_is_reusable(tmp_path: Path) -> None:
    p = _patch(tmp_path, "")
    _traj(tmp_path, "Submitted")
    assert existing_patch_is_reusable(tmp_path, p) is True


def test_empty_patch_with_limits_exceeded_is_reusable(tmp_path: Path) -> None:
    p = _patch(tmp_path, "")
    _traj(tmp_path, "LimitsExceeded")
    assert existing_patch_is_reusable(tmp_path, p) is True


def test_empty_patch_with_retry_error_is_not_reusable(tmp_path: Path) -> None:
    p = _patch(tmp_path, "")
    _traj(tmp_path, "RetryError")
    assert existing_patch_is_reusable(tmp_path, p) is False


def test_empty_patch_without_trajectory_is_not_reusable(tmp_path: Path) -> None:
    p = _patch(tmp_path, "")
    assert existing_patch_is_reusable(tmp_path, p) is False


def test_exit_status_none_when_no_traj(tmp_path: Path) -> None:
    assert mini_traj_exit_status(tmp_path) is None


def test_first_retryable_task_skips_terminal_prefix(tmp_path: Path) -> None:
    tasks = [
        Task(
            instance_id=f"i{index}",
            repo="r/r",
            base_commit=str(index),
            problem_statement="fix",
            dockerhub_tag=f"tag-{index}",
        )
        for index in range(3)
    ]
    universe = tmp_path / "universes" / "r__r"
    for task, status in zip(
        tasks,
        ("Submitted", "AgentTimeout", "RetryError"),
        strict=True,
    ):
        artifacts = universe / task.instance_id
        artifacts.mkdir(parents=True)
        (artifacts / f"{task.instance_id}.mini.traj.json").write_text(
            json.dumps({"info": {"exit_status": status}, "messages": []})
        )

    assert first_retryable_task(
        run_root=tmp_path,
        universe_repo="r/r",
        tasks=tasks,
    ) == (1, "AgentTimeout")


def test_first_retryable_task_includes_missing_trajectory(tmp_path: Path) -> None:
    task = Task(
        instance_id="i0",
        repo="r/r",
        base_commit="abc",
        problem_statement="fix",
        dockerhub_tag="tag",
    )

    assert first_retryable_task(
        run_root=tmp_path,
        universe_repo="r/r",
        tasks=[task],
    ) == (0, "missing")


def test_first_retryable_task_none_when_all_tasks_terminal(tmp_path: Path) -> None:
    task = Task(
        instance_id="i0",
        repo="r/r",
        base_commit="abc",
        problem_statement="fix",
        dockerhub_tag="tag",
    )
    artifacts = tmp_path / "universes" / "r__r" / task.instance_id
    artifacts.mkdir(parents=True)
    (artifacts / f"{task.instance_id}.mini.traj.json").write_text(
        json.dumps({"info": {"exit_status": "LimitsExceeded"}, "messages": []})
    )

    assert first_retryable_task(
        run_root=tmp_path,
        universe_repo="r/r",
        tasks=[task],
    ) is None


def test_stale_sequential_state_is_removed_when_first_task_reruns(
    tmp_path: Path,
) -> None:
    task = Task(
        instance_id="i",
        repo="r/r",
        base_commit="abc",
        problem_statement="fix",
        dockerhub_tag="tag",
    )
    state_dir = tmp_path / "_mini_state"
    state_dir.mkdir()
    (state_dir / "messages.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "stale"}]})
    )

    reset_stale_sequential_state_for_fresh_start(
        universe_dir=tmp_path,
        tasks=[task],
        skip_existing_patches=True,
        memory_seeded_initial=False,
    )

    assert not state_dir.exists()


def test_stale_sequential_state_is_kept_when_first_task_is_reusable(
    tmp_path: Path,
) -> None:
    task = Task(
        instance_id="i",
        repo="r/r",
        base_commit="abc",
        problem_statement="fix",
        dockerhub_tag="tag",
    )
    state_dir = tmp_path / "_mini_state"
    state_dir.mkdir()
    (state_dir / "messages.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "prefix"}]})
    )
    artifacts = tmp_path / "i"
    artifacts.mkdir()
    _patch(artifacts, "diff --git a/x b/x\n+y\n")
    _traj(artifacts, "Submitted")

    reset_stale_sequential_state_for_fresh_start(
        universe_dir=tmp_path,
        tasks=[task],
        skip_existing_patches=True,
        memory_seeded_initial=False,
    )

    assert state_dir.exists()
