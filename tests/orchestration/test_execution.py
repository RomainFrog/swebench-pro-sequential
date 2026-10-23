"""Tests for task execution completion checks and grading dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from swebench_seq.agent.config import MiniSweAgentOptions, prompt_cache_key_for_repo
from swebench_seq.evaluation import runner as evaluator
from swebench_seq.orchestration import execution
from swebench_seq.predictions import PredictionsAccumulator
from swebench_seq.types import Task


def test_inline_grade_routes_to_local_score_one(monkeypatch, tmp_path: Path) -> None:
    seen: dict = {}

    def fake_score_one(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(execution, "score_one", fake_score_one)
    dataset_path = tmp_path / "tasks.jsonl"

    result = execution.grade_task(
        repo_dir=tmp_path,
        instance_id="inst",
        patch="P",
        image_prefix="img",
        tag_lookup={"inst": "tag"},
        dataset_path=dataset_path,
    )

    assert result is True
    assert seen == {
        "repo_dir": tmp_path,
        "instance_id": "inst",
        "patch": "P",
        "image_prefix": "img",
        "tag_lookup": {"inst": "tag"},
        "evaluator_dir": None,
        "dataset_path": dataset_path,
    }


def _task() -> Task:
    return Task(
        instance_id="instance_org__repo-abc",
        repo="org/repo",
        base_commit="abc",
        problem_statement="fix it",
        dockerhub_tag="tag",
    )


def test_run_completion_errors_reports_missing_prediction(tmp_path: Path) -> None:
    (tmp_path / "predictions.json").write_text("{}")

    assert execution.run_completion_errors(
        run_root=tmp_path,
        tasks=[_task()],
        sequential=True,
        all_repos=False,
    ) == [f"run={tmp_path.name} task={_task().instance_id}: missing prediction"]


def test_run_completion_errors_accepts_false_official_score(tmp_path: Path) -> None:
    task = _task()
    patch_text = "diff --git a/x b/x\n"
    (tmp_path / "predictions.json").write_text(json.dumps({task.instance_id: patch_text}))
    repo_dir = tmp_path / "universes" / "org__repo"
    artifacts_dir = repo_dir / task.instance_id
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / f"{task.instance_id}.mini.traj.json").write_text(
        json.dumps({"info": {"exit_status": "Submitted"}})
    )
    (artifacts_dir / "task_result.json").write_text(
        json.dumps({"instance_id": task.instance_id, "rollout_status": "Submitted"})
    )
    (artifacts_dir / "container_image.json").write_text(
        json.dumps({"id": "sha256:test"})
    )
    evaluator._merge_eval_result(
        repo_dir,
        task.instance_id,
        False,
        patch_sha256=evaluator._patch_sha256(patch_text),
    )

    assert execution.run_completion_errors(
        run_root=tmp_path,
        tasks=[task],
        sequential=True,
        all_repos=False,
    ) == []


def test_run_completion_errors_rejects_model_failure(tmp_path: Path) -> None:
    task = _task()
    (tmp_path / "predictions.json").write_text(json.dumps({task.instance_id: ""}))
    artifacts_dir = tmp_path / "universes" / "org__repo" / task.instance_id
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / f"{task.instance_id}.mini.traj.json").write_text(
        json.dumps({"info": {"exit_status": "AuthenticationError"}})
    )

    errors = execution.run_completion_errors(
        run_root=tmp_path,
        tasks=[task],
        sequential=True,
        all_repos=False,
    )

    assert errors == [
        f"run={tmp_path.name} task={task.instance_id}: rollout status AuthenticationError"
    ]


def test_sequential_does_not_grade_nonterminal_rollout(
    monkeypatch, tmp_path: Path
) -> None:
    task = _task()
    grade = Mock(side_effect=AssertionError("nonterminal rollout must not be graded"))
    session = Mock()
    session_options: list[MiniSweAgentOptions] = []

    def fake_session(**kwargs):
        session_options.append(kwargs["options"])
        return session

    monkeypatch.setattr(execution, "MiniConversationSession", fake_session)
    monkeypatch.setattr(execution, "load_tag_lookup", lambda _path: {})
    monkeypatch.setattr(
        execution,
        "run_task",
        lambda **_kwargs: execution.TaskExecutionResult(
            executed=True,
            rollout_status="AuthenticationError",
        ),
    )
    monkeypatch.setattr(execution, "grade_task", grade)
    monkeypatch.setattr(execution, "snapshot_agent_home", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(execution, "prune_agent_home", lambda *_args, **_kwargs: None)

    execution.run_universe_sequential(
        universe_repo=task.repo,
        tasks=[task],
        output_dir=tmp_path,
        env={},
        mini_options=MiniSweAgentOptions(),
        accumulator=PredictionsAccumulator(tmp_path),
    )

    grade.assert_not_called()
    assert session_options[0].prompt_cache_key == prompt_cache_key_for_repo(task.repo)
