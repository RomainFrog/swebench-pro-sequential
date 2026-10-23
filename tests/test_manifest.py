"""Tests for sanitized run provenance."""

from __future__ import annotations

import json
from pathlib import Path

from swebench_seq.manifest import (
    build_image_manifest,
    build_run_manifest,
    build_task_manifest,
)
from swebench_seq.tasks import TaskUniversePlan
from swebench_seq.types import Task


def test_task_manifest_records_order_commits_and_images() -> None:
    task = Task(
        instance_id="instance_org__repo-1",
        repo="org/repo",
        base_commit="abc123",
        problem_statement="Fix it",
        dockerhub_tag="org.repo-1",
    )
    plan = TaskUniversePlan(
        repo=task.repo,
        tasks=(task,),
        available_task_count=3,
        order_source="frozen",
    )

    manifest = build_task_manifest(
        task_suite="repo40",
        cap_per_repo=40,
        limit=1,
        sequential=True,
        n_runs=1,
        image_prefix="example/images",
        planned_universes=[plan],
    )

    assert manifest["repos"][0]["tasks"] == [
        {
            "position": 0,
            "instance_id": task.instance_id,
            "base_commit": "abc123",
            "image": "example/images:org.repo-1",
        }
    ]
    assert manifest["benchmark_version"] == "swebench-pro-sequential-v1"
    assert manifest["canonical_inputs"] is False
    assert manifest["complete_suite"] is False
    assert len(manifest["sequential_order_sha256"]) == 64


def test_image_manifest_collects_recorded_identity(tmp_path: Path) -> None:
    task = Task(
        instance_id="instance_org__repo-1",
        repo="org/repo",
        base_commit="abc123",
        problem_statement="Fix it",
        dockerhub_tag="org.repo-1",
    )
    task_dir = tmp_path / "universes" / "org__repo" / task.instance_id
    task_dir.mkdir(parents=True)
    identity = {
        "reference": "example/images:org.repo-1",
        "id": "sha256:id",
        "digest": "sha256:digest",
        "repo_digests": ["example/images@sha256:digest"],
    }
    (task_dir / "container_image.json").write_text(json.dumps(identity))

    manifest = build_image_manifest(
        run_root=tmp_path,
        tasks=[task],
        sequential=True,
        all_repos=False,
    )

    assert manifest["tasks"] == [{"instance_id": task.instance_id, "image": identity}]


def test_run_manifest_redacts_credentials_and_url_auth(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text("{}\n")
    monkeypatch.setenv("MODEL_SECRET", "environment-secret")

    manifest = build_run_manifest(
        experiment_config_source="test.yaml",
        experiment_config_path=None,
        output_dir=Path("runs/test"),
        dataset_override=dataset_path,
        dataset_path=dataset_path,
        evaluator_dir=None,
        provider="openai",
        api_key_env="MODEL_SECRET",
        base_url="https://user:password@example.com/v1?token=hidden",
        cli_model=None,
        config_model={
            "model_name": "test-model",
            "model_kwargs": {
                "api_key": "config-secret",
                "extra_headers": {"X-Api-Key": "header-secret"},
                "max_output_tokens": 123,
            },
        },
        config_overrides=[
            ("model.model_kwargs.authorization", "Bearer override-secret"),
            ("agent.step_limit", "2"),
        ],
        sequential=True,
        concurrency=1,
        n_runs=1,
        overall_timeout=60,
        context_window=128_000,
        compact_after_each_task=False,
        image_prefix="example/images",
        skip_existing_patches=False,
        retry=False,
        selected_task_count=1,
        selected_repo_count=1,
    )

    rendered = json.dumps(manifest)
    assert "environment-secret" not in rendered
    assert "config-secret" not in rendered
    assert "header-secret" not in rendered
    assert "override-secret" not in rendered
    assert "user:password" not in rendered
    assert "token=hidden" not in rendered
    assert manifest["model"]["base_url"] == "https://example.com/v1"
    assert manifest["model"]["config"]["model_kwargs"]["max_output_tokens"] == 123
    assert manifest["benchmark"]["version"] == "swebench-pro-sequential-v1"
    assert manifest["benchmark"]["canonical_inputs"] is False
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["packages"]["swebench-seq"] == "0.1.0"
