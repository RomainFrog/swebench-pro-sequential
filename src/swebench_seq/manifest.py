"""Sanitized run provenance and exact task-plan manifests."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .constants import (
    BENCHMARK_VERSION,
    COMMIT_DATES_PATH,
    COMMIT_TOPO_RANKS_PATH,
    REPO_ROOT,
    SEQUENTIAL_ORDER_PATH,
)
from .evaluation.assets import (
    EVALUATOR_ARCHIVE_SHA256,
    EVALUATOR_REPOSITORY,
    EVALUATOR_REVISION,
)
from .evaluation.dataset import (
    DATASET_JSONL_SHA256,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
)
from .redaction import path_is_sensitive, redact_secrets, sanitize_url
from .tasks import TaskUniversePlan
from .types import Task


def build_task_manifest(
    *,
    task_suite: str,
    cap_per_repo: int | None,
    limit: int | None,
    sequential: bool,
    n_runs: int,
    image_prefix: str,
    planned_universes: list[TaskUniversePlan],
    canonical_inputs: bool = False,
    complete_suite: bool = False,
) -> dict[str, Any]:
    """Describe the exact ordered tasks selected for a run."""

    return {
        "schema_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "canonical_inputs": canonical_inputs,
        "complete_suite": complete_suite,
        "sequential_order_sha256": _file_sha256(SEQUENTIAL_ORDER_PATH),
        "task_suite": task_suite,
        "cap_per_repo": cap_per_repo,
        "limit": limit,
        "sequential": sequential,
        "n_runs": n_runs,
        "repos": [
            {
                "repo": plan.repo,
                "order_source": plan.order_source,
                "available_task_count": plan.available_task_count,
                "selected_task_count": len(plan),
                "instance_ids": [task.instance_id for task in plan.tasks],
                "tasks": [
                    {
                        "position": position,
                        "instance_id": task.instance_id,
                        "base_commit": task.base_commit,
                        "image": f"{image_prefix}:{task.dockerhub_tag}",
                    }
                    for position, task in enumerate(plan.tasks)
                ],
            }
            for plan in planned_universes
        ],
    }


def build_run_manifest(
    *,
    experiment_config_source: str | None,
    experiment_config_path: Path | None,
    output_dir: Path,
    dataset_override: Path | None,
    dataset_path: Path,
    evaluator_dir: Path | None,
    provider: str,
    api_key_env: str | None,
    base_url: str | None,
    cli_model: str | None,
    config_model: dict[str, Any] | None,
    config_overrides: list[tuple[str, str]],
    sequential: bool,
    concurrency: int,
    n_runs: int,
    overall_timeout: float | None,
    context_window: int,
    compact_after_each_task: bool,
    image_prefix: str,
    skip_existing_patches: bool,
    retry: bool,
    selected_task_count: int,
    selected_repo_count: int,
    podman_version: str | None = None,
    canonical_inputs: bool = False,
    complete_suite: bool = False,
) -> dict[str, Any]:
    """Build public run provenance without embedding credentials."""

    model_root: dict[str, Any] = {"model": copy.deepcopy(config_model or {})}
    rendered_overrides: list[dict[str, Any]] = []
    for dotted_key, raw_value in config_overrides:
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"invalid YAML value for --set {dotted_key}: {raw_value!r}"
            ) from exc
        rendered_overrides.append(
            {
                "key": dotted_key,
                "value": (
                    "<redacted>" if path_is_sensitive(dotted_key) else redact_secrets(value)
                ),
            }
        )
        if dotted_key == "model":
            if not isinstance(value, dict):
                raise ValueError("--set model=... must produce a mapping")
            model_root["model"] = copy.deepcopy(value)
        elif dotted_key.startswith("model."):
            _set_nested_value(model_root, dotted_key, value)

    model_config = model_root["model"]
    if not isinstance(model_config, dict):
        raise ValueError("resolved model configuration must be a mapping")
    resolved_model_name = cli_model or model_config.get("model_name")
    if resolved_model_name is not None:
        model_config["model_name"] = resolved_model_name

    resolved_key_env = api_key_env or _default_api_key_env(provider)
    dataset_manifest: dict[str, Any] = {
        "source": "local_file" if dataset_override is not None else "hugging_face",
        "sha256": _file_sha256(dataset_path),
    }
    if dataset_override is None:
        dataset_manifest.update(
            {
                "name": DATASET_NAME,
                "revision": DATASET_REVISION,
                "split": DATASET_SPLIT,
                "expected_sha256": DATASET_JSONL_SHA256,
            }
        )
    else:
        dataset_manifest["file_name"] = dataset_path.name

    if evaluator_dir is None:
        evaluator_manifest = {
            "source": "pinned_archive",
            "repository": EVALUATOR_REPOSITORY,
            "revision": EVALUATOR_REVISION,
            "archive_sha256": EVALUATOR_ARCHIVE_SHA256,
        }
    else:
        evaluator_manifest = {
            "source": "local_directory",
            "directory_name": evaluator_dir.name,
        }

    return {
        "schema_version": 1,
        "benchmark": {
            **_benchmark_provenance(),
            "canonical_inputs": canonical_inputs,
            "complete_suite": complete_suite,
        },
        "runtime": _runtime_provenance(podman_version=podman_version),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "swebench_seq_version": __version__,
        "experiment_config": (
            Path(experiment_config_source).name if experiment_config_source else None
        ),
        "experiment_config_sha256": (
            _file_sha256(experiment_config_path)
            if experiment_config_path is not None and experiment_config_path.is_file()
            else None
        ),
        "output_dir": output_dir.name if output_dir.is_absolute() else str(output_dir),
        "task_manifest": "task_manifest.json",
        "dataset": dataset_manifest,
        "evaluator": evaluator_manifest,
        "model": {
            "provider": provider,
            "api_key_env": resolved_key_env,
            "api_key_configured": bool(resolved_key_env and os.environ.get(resolved_key_env)),
            "base_url": sanitize_url(base_url),
            "config": redact_secrets(model_config),
            "overrides": rendered_overrides,
        },
        "execution": {
            "mode": "sequential" if sequential else "independent",
            "concurrency": concurrency,
            "n_runs": n_runs,
            "overall_timeout": overall_timeout,
            "context_window": context_window,
            "compact_after_each_task": compact_after_each_task,
            "image_prefix": image_prefix,
            "skip_existing_patches": skip_existing_patches,
            "retry": retry,
        },
        "selection": {"repositories": selected_repo_count, "tasks": selected_task_count},
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_image_manifest(
    *,
    run_root: Path,
    tasks: list[Task],
    sequential: bool,
    all_repos: bool,
) -> dict[str, Any]:
    """Collect per-task immutable image identities after a run."""

    entries: list[dict[str, Any]] = []
    for task in tasks:
        if sequential:
            task_dir = (
                run_root
                / "universes"
                / task.repo.replace("/", "__")
                / task.instance_id
            )
        elif all_repos:
            task_dir = run_root / task.repo.replace("/", "__") / task.instance_id
        else:
            task_dir = run_root / task.instance_id
        image_path = task_dir / "container_image.json"
        try:
            identity = json.loads(image_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            identity = None
        entries.append(
            {
                "instance_id": task.instance_id,
                "image": identity,
            }
        )
    return {
        "schema_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "tasks": entries,
    }


def _default_api_key_env(provider: str) -> str:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }[provider]


def _set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = copy.deepcopy(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_provenance() -> dict[str, Any]:
    return {
        "version": BENCHMARK_VERSION,
        "sequential_order_sha256": _file_sha256(SEQUENTIAL_ORDER_PATH),
        "commit_dates_sha256": _file_sha256(COMMIT_DATES_PATH),
        "commit_topo_ranks_sha256": _file_sha256(COMMIT_TOPO_RANKS_PATH),
    }


def _runtime_provenance(*, podman_version: str | None) -> dict[str, Any]:
    distributions = (
        "mini-swe-agent",
        "openai",
        "datasets",
        "docker",
        "httpx",
    )
    versions: dict[str, str] = {"swebench-seq": __version__}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue

    source_commit, source_dirty, source_untracked = _source_git_state()
    lock_path = REPO_ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "podman": podman_version,
        "packages": versions,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_untracked": source_untracked,
        "uv_lock_sha256": _file_sha256(lock_path) if lock_path.is_file() else None,
    }


def _source_git_state() -> tuple[str | None, bool | None, bool | None]:
    if not (REPO_ROOT / ".git").exists():
        return None, None, None
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    return commit or None, bool(status.strip()), bool(untracked.strip())
