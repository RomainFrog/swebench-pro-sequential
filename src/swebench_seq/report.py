"""Summarize the artifacts and official scores from a completed run."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import BENCHMARK_VERSION, SEQUENTIAL_ORDER_PATH
from .orchestration.recovery import TERMINAL_EXIT_STATUSES


@dataclass
class TaskReport:
    """Per-task metrics for one completed run."""

    instance_id: str
    repo: str
    position: int
    patch_bytes: int
    model_calls: int
    passed: bool | None = None
    rollout_status: str | None = None
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    memory_md_bytes: int | None = None
    agent_home_bytes: int | None = None
    has_agent_home: bool = False


@dataclass
class RunReport:
    """Aggregated report for one swebench-seq run."""

    run_dir: Path
    mode: str
    selected_task_count: int
    benchmark_version: str | None
    canonical_inputs: bool
    complete_suite: bool
    tasks: list[TaskReport] = field(default_factory=list)

    def by_repo(self) -> dict[str, list[TaskReport]]:
        grouped: dict[str, list[TaskReport]] = defaultdict(list)
        for task in self.tasks:
            grouped[task.repo].append(task)
        for tasks in grouped.values():
            tasks.sort(key=lambda task: task.position)
        return dict(grouped)

    def pass_rate_by_position(self, repo: str) -> list[tuple[int, float | None]]:
        """Return the running pass rate at each sequence position."""
        tasks = self.by_repo().get(repo, [])
        out: list[tuple[int, float | None]] = []
        scored = 0
        passed = 0
        for task in tasks:
            if task.passed is not None:
                scored += 1
                passed += int(task.passed)
            out.append((task.position, passed / scored if scored else None))
        return out

    def summary(self) -> dict[str, int | float | None]:
        graded = [task for task in self.tasks if task.passed is not None]
        passed = sum(task.passed is True for task in graded)
        usage = _load_run_usage(self.run_dir)
        if usage["events"] == 0:
            usage = {
                "events": sum(task.model_calls for task in self.tasks),
                "compaction_calls": 0,
                "input_tokens": _sum_optional(task.input_tokens for task in self.tasks),
                "output_tokens": _sum_optional(task.output_tokens for task in self.tasks),
                "total_tokens": _sum_optional(task.total_tokens for task in self.tasks),
            }
        return {
            "selected": self.selected_task_count,
            "artifacts": len(self.tasks),
            "completed_rollouts": sum(
                task.rollout_status in TERMINAL_EXIT_STATUSES for task in self.tasks
            ),
            "graded": len(graded),
            "passed": passed,
            "failed": len(graded) - passed,
            "infrastructure_or_incomplete": max(0, self.selected_task_count - len(graded)),
            "pass_rate": passed / len(graded) if graded else None,
            "model_calls": sum(task.model_calls for task in self.tasks),
            "compaction_calls": usage["compaction_calls"],
            "wall_seconds": sum(
                task.wall_seconds for task in self.tasks if task.wall_seconds is not None
            ),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }


def load_report(run_dir: Path) -> RunReport:
    """Read current runner artifacts without relying on legacy analysis files."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    mode = "sequential" if (run_dir / "universes").is_dir() else "independent"
    run_manifest = _load_json_object(run_dir.parent / "run_manifest.json") or {}
    benchmark = run_manifest.get("benchmark")
    benchmark = benchmark if isinstance(benchmark, dict) else {}
    manifest_order = _load_manifest_order(run_dir)
    run_scores = _load_scores(run_dir)
    tasks: list[TaskReport] = []

    if mode == "sequential":
        for universe_dir in sorted((run_dir / "universes").iterdir()):
            if not universe_dir.is_dir():
                continue
            repo = universe_dir.name.replace("__", "/")
            tasks.extend(
                _reports_for_repo(
                    repo=repo,
                    repo_dir=universe_dir,
                    ordered_ids=manifest_order.get(repo),
                    sequential=True,
                    run_scores=run_scores,
                )
            )
    elif manifest_order:
        for repo, ordered_ids in manifest_order.items():
            nested = run_dir / repo.replace("/", "__")
            repo_dir = nested if nested.is_dir() else run_dir
            tasks.extend(
                _reports_for_repo(
                    repo=repo,
                    repo_dir=repo_dir,
                    ordered_ids=ordered_ids,
                    sequential=False,
                    run_scores=run_scores,
                )
            )
    else:
        tasks.extend(_discover_independent_reports(run_dir, run_scores=run_scores))

    selected_task_count = sum(len(ids) for ids in manifest_order.values()) or len(tasks)
    return RunReport(
        run_dir=run_dir,
        mode=mode,
        selected_task_count=selected_task_count,
        benchmark_version=_result_str(benchmark, "version"),
        canonical_inputs=benchmark.get("canonical_inputs") is True,
        complete_suite=benchmark.get("complete_suite") is True,
        tasks=tasks,
    )


def _reports_for_repo(
    *,
    repo: str,
    repo_dir: Path,
    ordered_ids: list[str] | None,
    sequential: bool,
    run_scores: dict[str, bool],
) -> list[TaskReport]:
    task_dirs = {
        entry.name: entry
        for entry in repo_dir.iterdir()
        if entry.is_dir() and _is_task_artifact_dir(entry)
    }
    if ordered_ids is None:
        instance_ids = sorted(task_dirs)
    else:
        instance_ids = [instance_id for instance_id in ordered_ids if instance_id in task_dirs]

    scores = _load_scores(repo_dir)
    # `swebench-seq score` writes one aggregate score file at the run root,
    # while inline grading writes per-repository files. A later explicit bulk
    # score is authoritative when both exist.
    scores.update(run_scores)
    return [
        _task_report(
            instance_id=instance_id,
            repo=repo,
            position=position,
            artifacts_dir=task_dirs[instance_id],
            score_data=scores,
            sequential=sequential,
        )
        for position, instance_id in enumerate(instance_ids)
    ]


def _discover_independent_reports(
    run_dir: Path, *, run_scores: dict[str, bool]
) -> list[TaskReport]:
    direct = [entry for entry in run_dir.iterdir() if _is_task_artifact_dir(entry)]
    if direct:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for task_dir in direct:
            grouped[_instance_id_to_repo(task_dir.name)].append(task_dir)
        reports: list[TaskReport] = []
        for repo, task_dirs in sorted(grouped.items()):
            for position, task_dir in enumerate(sorted(task_dirs)):
                reports.append(
                    _task_report(
                        instance_id=task_dir.name,
                        repo=repo,
                        position=position,
                        artifacts_dir=task_dir,
                        score_data=run_scores,
                        sequential=False,
                    )
                )
        return reports

    reports = []
    for repo_dir in sorted(run_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        task_dirs = [entry for entry in repo_dir.iterdir() if _is_task_artifact_dir(entry)]
        if not task_dirs:
            continue
        repo = repo_dir.name.replace("__", "/")
        reports.extend(
            _reports_for_repo(
                repo=repo,
                repo_dir=repo_dir,
                ordered_ids=None,
                sequential=False,
                run_scores=run_scores,
            )
        )
    return reports


def _task_report(
    *,
    instance_id: str,
    repo: str,
    position: int,
    artifacts_dir: Path,
    score_data: dict[str, bool],
    sequential: bool,
) -> TaskReport:
    snapshot = artifacts_dir / "_agent_home_snapshot"
    has_agent_home = sequential and snapshot.is_dir()
    result = _load_task_result(artifacts_dir, instance_id)
    result_model_calls = _result_int(result, "model_calls")
    rollout_status = _result_str(result, "rollout_status") or _load_rollout_status(
        artifacts_dir, instance_id
    )
    return TaskReport(
        instance_id=instance_id,
        repo=repo,
        position=position,
        patch_bytes=_file_size(artifacts_dir / "patch.diff") or 0,
        model_calls=(
            result_model_calls
            if result_model_calls is not None
            else _load_model_calls(artifacts_dir, instance_id)
        ),
        passed=(
            score_data.get(instance_id)
            if rollout_status in TERMINAL_EXIT_STATUSES
            else None
        ),
        rollout_status=rollout_status,
        wall_seconds=_result_number(result, "wall_seconds"),
        input_tokens=_result_usage_int(result, "input_tokens"),
        output_tokens=_result_usage_int(result, "output_tokens"),
        total_tokens=_result_usage_int(result, "total_tokens"),
        memory_md_bytes=_file_size(snapshot / "MEMORY.md") if has_agent_home else None,
        agent_home_bytes=_dir_size_bytes(snapshot) if has_agent_home else None,
        has_agent_home=has_agent_home,
    )


def _load_manifest_order(run_dir: Path) -> dict[str, list[str]]:
    candidates = (run_dir.parent / "task_manifest.json", run_dir / "task_manifest.json")
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        repos = data.get("repos") if isinstance(data, dict) else None
        if not isinstance(repos, list):
            continue
        order: dict[str, list[str]] = {}
        for entry in repos:
            if not isinstance(entry, dict):
                continue
            repo = entry.get("repo")
            instance_ids = entry.get("instance_ids")
            if isinstance(repo, str) and isinstance(instance_ids, list):
                order[repo] = [item for item in instance_ids if isinstance(item, str)]
        return order
    return {}


def _load_scores(repo_dir: Path) -> dict[str, bool]:
    """Read the score file written by the inline or bulk evaluator."""
    try:
        data = json.loads((repo_dir / "score" / "eval_results.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    scores: dict[str, bool] = {}
    for instance_id, value in data.items():
        if not isinstance(instance_id, str):
            continue
        if isinstance(value, bool):
            scores[instance_id] = value
        elif isinstance(value, dict):
            resolved = value.get("resolved", value.get("passed"))
            if isinstance(resolved, bool):
                scores[instance_id] = resolved
    return scores


def _load_model_calls(artifacts_dir: Path, instance_id: str) -> int:
    trajectory = artifacts_dir / f"{instance_id}.mini.traj.json"
    try:
        data: Any = json.loads(trajectory.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0
    if not isinstance(data, dict):
        return 0
    info = data.get("info")
    stats = info.get("model_stats") if isinstance(info, dict) else None
    api_calls = stats.get("api_calls") if isinstance(stats, dict) else None
    return api_calls if isinstance(api_calls, int) and api_calls >= 0 else 0


def _load_rollout_status(artifacts_dir: Path, instance_id: str) -> str | None:
    trajectory = artifacts_dir / f"{instance_id}.mini.traj.json"
    try:
        data: Any = json.loads(trajectory.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    info = data.get("info") if isinstance(data, dict) else None
    status = info.get("exit_status") if isinstance(info, dict) else None
    return status if isinstance(status, str) else None


def _load_task_result(artifacts_dir: Path, instance_id: str) -> dict[str, Any]:
    try:
        data = json.loads((artifacts_dir / "task_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("instance_id") != instance_id:
        return {}
    return data


def _result_int(result: dict[str, Any], key: str) -> int | None:
    value = result.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _result_number(result: dict[str, Any], key: str) -> float | None:
    value = result.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _result_str(result: dict[str, Any], key: str) -> str | None:
    value = result.get(key)
    return value if isinstance(value, str) else None


def _result_usage_int(result: dict[str, Any], key: str) -> int | None:
    usage = result.get("usage")
    return _result_int(usage, key) if isinstance(usage, dict) else None


def _load_run_usage(run_dir: Path) -> dict[str, int | None]:
    events: list[dict[str, Any]] = []
    for path in run_dir.glob("universes/*/_mini_state/usage.jsonl"):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return {
        "events": len(events),
        "compaction_calls": sum(item.get("phase") == "compaction" for item in events),
        "input_tokens": _sum_optional(_event_int(item, "input_tokens") for item in events),
        "output_tokens": _sum_optional(
            _event_int(item, "output_tokens") for item in events
        ),
        "total_tokens": _sum_optional(_event_int(item, "total_tokens") for item in events),
    }


def _event_int(event: dict[str, Any], key: str) -> int | None:
    value = event.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sum_optional(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def validate_run(run_dir: Path) -> list[str]:
    """Return reproducibility/completeness errors for one run directory."""

    errors: list[str] = []
    run_manifest = _load_json_object(run_dir.parent / "run_manifest.json")
    task_manifest = _load_json_object(run_dir.parent / "task_manifest.json")
    if run_manifest is None:
        errors.append("missing or invalid run_manifest.json")
    else:
        benchmark = run_manifest.get("benchmark")
        if not isinstance(benchmark, dict) or benchmark.get("version") != BENCHMARK_VERSION:
            errors.append(f"run manifest is not {BENCHMARK_VERSION}")
        elif benchmark.get("sequential_order_sha256") != _sha256(SEQUENTIAL_ORDER_PATH):
            errors.append("run manifest sequential-order hash differs from this release")
        runtime = run_manifest.get("runtime")
        if isinstance(runtime, dict) and runtime.get("source_dirty") is True:
            errors.append("run used a source checkout with tracked modifications")
        package_versions = runtime.get("packages") if isinstance(runtime, dict) else None
        recorded_version = run_manifest.get("swebench_seq_version")
        runtime_version = (
            package_versions.get("swebench-seq")
            if isinstance(package_versions, dict)
            else None
        )
        if (
            not isinstance(recorded_version, str)
            or not isinstance(runtime_version, str)
            or recorded_version != runtime_version
        ):
            errors.append("run manifest contains inconsistent harness versions")
    if task_manifest is None:
        errors.append("missing or invalid task_manifest.json")
    elif task_manifest.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(f"task manifest is not {BENCHMARK_VERSION}")

    report = load_report(run_dir)
    summary = report.summary()
    if summary["artifacts"] != summary["selected"]:
        errors.append(
            f"task artifacts {summary['artifacts']}/{summary['selected']}"
        )
    if summary["completed_rollouts"] != summary["selected"]:
        errors.append(
            f"terminal rollouts {summary['completed_rollouts']}/{summary['selected']}"
        )
    if summary["graded"] != summary["selected"]:
        errors.append(f"official scores {summary['graded']}/{summary['selected']}")

    predictions = _load_json_object(run_dir / "predictions.json")
    expected_ids = {task.instance_id for task in report.tasks}
    if predictions is None or set(predictions) != expected_ids or not all(
        isinstance(patch, str) for patch in predictions.values()
    ):
        errors.append("predictions.json does not contain one string patch per task")
    for task in report.tasks:
        artifacts_dir = _task_artifacts_dir(run_dir, report.mode, task)
        if not (artifacts_dir / "task_result.json").is_file():
            errors.append(f"missing task_result.json for {task.instance_id}")

    image_manifest = run_dir / "image_manifest.json"
    try:
        image_data = json.loads(image_manifest.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        errors.append("missing or invalid image_manifest.json")
    else:
        image_tasks = image_data.get("tasks") if isinstance(image_data, dict) else None
        identities = (
            [entry.get("image") for entry in image_tasks if isinstance(entry, dict)]
            if isinstance(image_tasks, list)
            else []
        )
        if len(identities) != report.selected_task_count or any(
            not isinstance(identity, dict) or not identity.get("id")
            for identity in identities
        ):
            errors.append("image manifest lacks an identity for every selected task")
    return errors


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _task_artifacts_dir(run_dir: Path, mode: str, task: TaskReport) -> Path:
    if mode == "sequential":
        return run_dir / "universes" / task.repo.replace("/", "__") / task.instance_id
    nested = run_dir / task.repo.replace("/", "__") / task.instance_id
    return nested if nested.is_dir() else run_dir / task.instance_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_task_artifact_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "patch.diff").is_file() or any(path.glob("*.mini.traj.json"))
    )


def _instance_id_to_repo(instance_id: str) -> str:
    raw = instance_id.removeprefix("instance_")
    org_repo, separator, _suffix = raw.partition("-")
    if separator and "__" in org_repo:
        org, repo = org_repo.split("__", 1)
        return f"{org}/{repo}"
    return instance_id


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except (FileNotFoundError, NotADirectoryError):
        return None


def _dir_size_bytes(path: Path) -> int:
    """Return bytes under ``path`` without following symlinks."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total
