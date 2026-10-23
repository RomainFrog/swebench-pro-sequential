"""No-network CLI smoke covering planning, sequential state, grading, and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

from swebench_seq.agent.config import prompt_cache_key_for_repo
from swebench_seq.cli import main
from swebench_seq.evaluation import runner as evaluator
from swebench_seq.orchestration import execution
from swebench_seq.types import RunResult

pytestmark = pytest.mark.integration


class FakeContainerRunner:
    starts: ClassVar[list[tuple[str, Path | None]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.task = kwargs["task"]
        self.artifacts_dir = kwargs["artifacts_dir"]
        self.agent_home_mount = kwargs.get("agent_home_mount")
        self.image_prefix = kwargs["image_prefix"]
        self.container_name = f"fake-{self.task.instance_id}"
        self.podman = object()

    def __enter__(self) -> FakeContainerRunner:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def start(self, *, env: dict[str, str]) -> None:
        del env
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.starts.append((self.task.base_commit, self.agent_home_mount))
        (self.artifacts_dir / "container_image.json").write_text(
            json.dumps(
                {
                    "reference": f"{self.image_prefix}:{self.task.dockerhub_tag}",
                    "id": f"sha256:{self.task.base_commit}",
                    "digest": f"sha256:{self.task.base_commit}",
                    "repo_digests": [],
                }
            )
        )

    def extract_patch(self) -> str:
        patch = f"diff --git a/{self.task.instance_id} b/{self.task.instance_id}\n"
        (self.artifacts_dir / "patch.diff").write_text(patch)
        return patch


class FakeConversationSession:
    compactions: ClassVar[int] = 0
    prompt_cache_keys: ClassVar[list[str | None]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.prompt_cache_keys.append(kwargs["options"].prompt_cache_key)
        self.state_dir: Path = kwargs["state_dir"]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.task_number = 0

    def run_task(
        self,
        *,
        runner: FakeContainerRunner,
        task: Any,
        artifacts_dir: Path,
        timeout_seconds: float | None = None,
    ) -> RunResult:
        del timeout_seconds
        self.task_number += 1
        boundary = "" if self.task_number == 1 else "The repository has advanced to a new commit."
        (artifacts_dir / f"{task.instance_id}.mini.traj.json").write_text(
            json.dumps(
                {
                    "info": {"exit_status": "Submitted"},
                    "messages": [{"role": "user", "content": boundary}],
                }
            )
        )
        return RunResult(
            patch=runner.extract_patch(),
            turns_used=1,
            wall_seconds=0.25,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )

    def compact_completed_history(self, *, reason: str) -> bool:
        assert reason == "after_task"
        type(self).compactions += 1
        return True


class FakeAgentHomePersistence:
    def __init__(self, *, host_dir: Path, podman: object) -> None:
        del podman
        self.host_dir = host_dir

    def extract(self, container_name: str, fallback_dir: Path | None = None) -> None:
        del container_name, fallback_dir
        self.host_dir.mkdir(parents=True, exist_ok=True)
        (self.host_dir / "MEMORY.md").write_text("# Durable test memory\n")


def test_two_task_public_workflow_without_network_or_model_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeContainerRunner.starts = []
    FakeConversationSession.compactions = 0
    FakeConversationSession.prompt_cache_keys = []
    dataset = tmp_path / "tasks.jsonl"
    rows = [
        {
            "instance_id": f"instance_org__repo-{index}",
            "repo": "org/repo",
            "base_commit": f"commit-{index}",
            "problem_statement": f"Fix issue {index}",
            "dockerhub_tag": f"org.repo-{index}",
        }
        for index in range(2)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "output"
    config = tmp_path / "smoke.yaml"
    config.write_text(
        "repo: org/repo\n"
        "sequential: true\n"
        "compact_after_each_task: true\n"
        "api_key_env: FAKE_MODEL_KEY\n"
        f"dataset: {dataset}\n"
        f"output_dir: {output}\n"
        "model:\n"
        "  model_name: fake-model\n"
        "  model_class: fake.Model\n"
    )

    monkeypatch.setattr(execution, "ContainerRunner", FakeContainerRunner)
    monkeypatch.setattr(execution, "MiniConversationSession", FakeConversationSession)
    monkeypatch.setattr(execution, "AgentHomePersistence", FakeAgentHomePersistence)
    monkeypatch.setattr(execution, "load_tag_lookup", lambda dataset_path: {})
    monkeypatch.setattr(
        "swebench_seq.evaluation.runner.validate_evaluation_environment",
        lambda evaluator_dir: (tmp_path / "evaluator", "unix:///podman.sock"),
    )
    monkeypatch.setattr(
        "swebench_seq.orchestration.run.PodmanCli.version", lambda self: "test"
    )
    monkeypatch.setattr(
        "swebench_seq.manifest._source_git_state",
        lambda: ("0" * 40, False, False),
    )

    def fake_grade(**kwargs: Any) -> bool:
        evaluator._merge_eval_result(
            kwargs["repo_dir"],
            kwargs["instance_id"],
            True,
            patch_sha256=evaluator._patch_sha256(kwargs["patch"]),
        )
        return True

    monkeypatch.setattr(execution, "grade_task", fake_grade)

    result = CliRunner().invoke(
        main,
        ["run", "--config", str(config)],
        env={"FAKE_MODEL_KEY": "local-test-only"},
    )

    assert result.exit_code == 0, result.output
    assert [commit for commit, _home in FakeContainerRunner.starts] == [
        "commit-0",
        "commit-1",
    ]
    assert FakeContainerRunner.starts[0][1] is None
    assert FakeContainerRunner.starts[1][1] is not None
    assert FakeConversationSession.compactions == 1
    assert FakeConversationSession.prompt_cache_keys == [
        prompt_cache_key_for_repo("org/repo")
    ]

    run_dir = output / "run_0"
    assert len(json.loads((run_dir / "predictions.json").read_text())) == 2
    assert all(
        entry["image"] is not None
        for entry in json.loads((run_dir / "image_manifest.json").read_text())["tasks"]
    )
    validation = CliRunner().invoke(
        main,
        ["validate-run", "--run-dir", str(run_dir)],
    )
    assert validation.exit_code == 0, validation.output


def test_independent_public_workflow_without_network_or_model_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeContainerRunner.starts = []
    prompt_cache_keys: list[str | None] = []
    dataset = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "instance_org__repo-0",
        "repo": "org/repo",
        "base_commit": "commit-0",
        "problem_statement": "Fix issue 0",
        "dockerhub_tag": "org.repo-0",
    }
    dataset.write_text(json.dumps(row) + "\n")
    output = tmp_path / "output"
    config = tmp_path / "independent.yaml"
    config.write_text(
        "repo: org/repo\n"
        "sequential: false\n"
        "api_key_env: FAKE_MODEL_KEY\n"
        f"dataset: {dataset}\n"
        f"output_dir: {output}\n"
        "model:\n"
        "  model_name: fake-model\n"
        "  model_class: fake.Model\n"
    )

    def fake_agent_task(**kwargs: Any) -> RunResult:
        prompt_cache_keys.append(kwargs["options"].prompt_cache_key)
        task = kwargs["task"]
        artifacts_dir = kwargs["artifacts_dir"]
        (artifacts_dir / f"{task.instance_id}.mini.traj.json").write_text(
            json.dumps({"info": {"exit_status": "Submitted"}, "messages": []})
        )
        return RunResult(
            patch=kwargs["runner"].extract_patch(),
            turns_used=1,
            wall_seconds=0.25,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )

    def fake_grade(**kwargs: Any) -> bool:
        evaluator._merge_eval_result(
            kwargs["repo_dir"],
            kwargs["instance_id"],
            True,
            patch_sha256=evaluator._patch_sha256(kwargs["patch"]),
        )
        return True

    monkeypatch.setattr(execution, "ContainerRunner", FakeContainerRunner)
    monkeypatch.setattr(execution, "run_mini_swe_agent_task", fake_agent_task)
    monkeypatch.setattr(execution, "grade_task", fake_grade)
    monkeypatch.setattr(
        "swebench_seq.evaluation.runner.validate_evaluation_environment",
        lambda evaluator_dir: (tmp_path / "evaluator", "unix:///podman.sock"),
    )
    monkeypatch.setattr(
        "swebench_seq.orchestration.run.PodmanCli.version", lambda self: "test"
    )
    monkeypatch.setattr(
        "swebench_seq.manifest._source_git_state",
        lambda: ("0" * 40, False, False),
    )

    result = CliRunner().invoke(
        main,
        ["run", "--config", str(config)],
        env={"FAKE_MODEL_KEY": "local-test-only"},
    )

    assert result.exit_code == 0, result.output
    assert FakeContainerRunner.starts == [("commit-0", None)]
    assert prompt_cache_keys == [prompt_cache_key_for_repo("org/repo")]
    run_dir = output / "run_0"
    assert json.loads((run_dir / "predictions.json").read_text()) == {
        row["instance_id"]: f"diff --git a/{row['instance_id']} b/{row['instance_id']}\n"
    }
    validation = CliRunner().invoke(
        main,
        ["validate-run", "--run-dir", str(run_dir)],
    )
    assert validation.exit_code == 0, validation.output
