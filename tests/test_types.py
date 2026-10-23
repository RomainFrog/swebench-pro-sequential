"""Tests for the core dataclasses."""

from __future__ import annotations

from swebench_seq.types import Task, Universe


def _task(instance_id: str = "qutebrowser-1", repo: str = "qutebrowser") -> Task:
    return Task(
        instance_id=instance_id,
        repo=repo,
        base_commit="abc123",
        problem_statement="Something is broken.",
        dockerhub_tag="qutebrowser-1",
    )


def test_task_is_hashable_and_immutable() -> None:
    t1 = _task()
    t2 = _task()
    assert hash(t1) == hash(t2)
    assert {t1, t2} == {t1}


def test_universe_len_matches_task_count() -> None:
    u = Universe(repo="qutebrowser", tasks=tuple(_task(f"q-{i}") for i in range(3)))
    assert len(u) == 3
