"""Run-level planning, scheduling, and artifact finalization."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..agent.config import MiniSweAgentOptions
from ..constants import IMAGE_PREFIX_DEFAULT
from ..container_network import proxy_env_for
from ..evaluation import runner as evaluation
from ..evaluation.dataset import materialize_dataset_jsonl
from ..experiment import read_instance_ids_file
from ..manifest import (
    build_image_manifest,
    build_run_manifest,
    build_task_manifest,
    write_manifest,
)
from ..predictions import PredictionsAccumulator
from ..runner import PodmanCli
from ..tasks import (
    DEFAULT_REPO_TASK_CAP,
    TaskUniversePlan,
    cap_task_universe_plans,
    load_frozen_sequential_order,
    load_tasks,
    plan_task_universes,
    validate_frozen_sequential_order,
)
from ..types import Task
from . import execution, recovery

SUPPORTED_PROVIDERS = ("openai", "anthropic", "openrouter")


class MessageSink(Protocol):
    def __call__(self, message: str, *, err: bool = False) -> None: ...


class RunConfigurationError(ValueError):
    """Raised when run options are contradictory or malformed."""


class RunExecutionError(RuntimeError):
    """Raised when a run cannot start or finishes incompletely."""


@dataclass(frozen=True)
class RunConfig:
    repo: str | None
    all_repos: bool
    sequential: bool
    limit: int | None
    target_instances: tuple[str, ...]
    target_instances_file: Path | None
    full_dataset: bool
    concurrency: int
    n_runs: int
    output_dir: Path
    image_prefix: str
    evaluator_dir: Path | None
    dataset: Path | None
    overall_timeout: float | None
    context_window: int
    compact_after_each_task: bool
    model: str | None
    api_key_env: str | None
    provider: str
    base_url: str | None
    set_overrides: tuple[str, ...]
    skip_existing_patches: bool
    retry: bool
    dry_run: bool
    experiment_model: dict[str, Any] | None = None
    experiment_config_source: str | None = None
    experiment_config_path: Path | None = None
    configured_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SelectedTasks:
    plans: tuple[TaskUniversePlan, ...]
    task_suite: str
    cap_per_repo: int | None
    manifest_loaded: bool
    explicit_target_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedRun:
    config: RunConfig
    evaluator_dir: Path | None
    dataset_path: Path
    selected: _SelectedTasks
    task_manifest: dict[str, Any]
    run_manifest: dict[str, Any]
    mini_api_key: str | None
    skip_existing_patches: bool
    overrides: tuple[tuple[str, str], ...]

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(task for plan in self.selected.plans for task in plan.tasks)


def execute_run(config: RunConfig, *, emit: MessageSink) -> None:
    """Prepare and execute one benchmark invocation."""

    prepared = _prepare_run(config, emit=emit)
    _emit_selection(prepared, emit=emit)
    if config.dry_run:
        emit("[swebench-seq] dry-run: no containers, model calls, grading, or artifacts")
        emit(
            json.dumps(
                {
                    "run_manifest": prepared.run_manifest,
                    "task_manifest": prepared.task_manifest,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = [config.output_dir / f"run_{index}" for index in range(config.n_runs)]
    accumulators = {
        index: PredictionsAccumulator(run_roots[index])
        for index in range(config.n_runs)
    }
    if not prepared.selected.manifest_loaded:
        write_manifest(config.output_dir / "task_manifest.json", prepared.task_manifest)
    write_manifest(config.output_dir / "run_manifest.json", prepared.run_manifest)

    mini_options = MiniSweAgentOptions(
        config_model=config.experiment_model,
        config_overrides=prepared.overrides,
        model=config.model,
        api_key=prepared.mini_api_key,
        base_url=config.base_url,
        provider=config.provider,
    )
    container_env = proxy_env_for("rollout")

    if config.sequential:
        _execute_sequential(
            prepared,
            run_roots=run_roots,
            accumulators=accumulators,
            mini_options=mini_options,
            container_env=container_env,
            emit=emit,
        )
    else:
        _execute_independent(
            prepared,
            run_roots=run_roots,
            accumulators=accumulators,
            mini_options=mini_options,
            container_env=container_env,
            emit=emit,
        )

    _finalize_runs(
        prepared,
        run_roots=run_roots,
        accumulators=accumulators,
        emit=emit,
    )


def resolve_api_key(*, env_name: str | None, provider: str) -> str | None:
    """Resolve the host-side model credential without exposing its value."""

    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise RunConfigurationError(
                f"model API key environment variable is not set: {env_name}"
            )
        return value
    try:
        default_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }[provider]
    except KeyError as exc:
        raise RunConfigurationError(f"unsupported provider: {provider!r}") from exc
    return os.environ.get(default_env)


def _prepare_run(config: RunConfig, *, emit: MessageSink) -> _PreparedRun:
    _validate_config(config)
    skip_existing_patches = config.skip_existing_patches and not config.retry
    overrides = _parse_set_overrides(config)

    mini_api_key = None
    podman_version = None
    evaluator_dir = config.evaluator_dir
    if not config.dry_run:
        mini_api_key = resolve_api_key(
            env_name=config.api_key_env,
            provider=config.provider,
        )
        try:
            evaluator_dir, _docker_host = evaluation.validate_evaluation_environment(
                evaluator_dir
            )
            podman_version = PodmanCli().version()
        except RuntimeError as exc:
            raise RunExecutionError(f"evaluation preflight failed: {exc}") from exc

    try:
        dataset_path = materialize_dataset_jsonl(config.dataset)
        all_tasks = load_tasks(dataset_path=dataset_path)
    except Exception as exc:
        raise RunExecutionError(f"failed to load SWE-bench Pro dataset: {exc}") from exc

    frozen = load_frozen_sequential_order()
    if config.dataset is None:
        try:
            validate_frozen_sequential_order(all_tasks, frozen)
        except ValueError as exc:
            raise RunExecutionError(str(exc)) from exc

    selected = _select_tasks(config, all_tasks=all_tasks, frozen=frozen, emit=emit)
    selected_task_count = sum(len(plan) for plan in selected.plans)
    canonical_inputs = (
        config.dataset is None
        and config.evaluator_dir is None
        and config.image_prefix == IMAGE_PREFIX_DEFAULT
    )
    complete_suite = (
        canonical_inputs
        and config.all_repos
        and not config.full_dataset
        and config.limit is None
        and not selected.explicit_target_ids
    )
    task_manifest = build_task_manifest(
        task_suite=selected.task_suite,
        cap_per_repo=selected.cap_per_repo,
        limit=config.limit,
        sequential=config.sequential,
        n_runs=config.n_runs,
        image_prefix=config.image_prefix,
        planned_universes=list(selected.plans),
        canonical_inputs=canonical_inputs,
        complete_suite=complete_suite,
    )
    try:
        run_manifest = build_run_manifest(
            experiment_config_source=config.experiment_config_source,
            experiment_config_path=config.experiment_config_path,
            output_dir=config.output_dir,
            dataset_override=config.dataset,
            dataset_path=dataset_path,
            evaluator_dir=config.evaluator_dir,
            provider=config.provider,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            cli_model=config.model,
            config_model=config.experiment_model,
            config_overrides=overrides,
            sequential=config.sequential,
            concurrency=config.concurrency,
            n_runs=config.n_runs,
            overall_timeout=config.overall_timeout,
            context_window=config.context_window,
            compact_after_each_task=config.compact_after_each_task,
            image_prefix=config.image_prefix,
            skip_existing_patches=skip_existing_patches,
            retry=config.retry,
            selected_task_count=selected_task_count,
            selected_repo_count=len(selected.plans),
            podman_version=podman_version,
            canonical_inputs=canonical_inputs,
            complete_suite=complete_suite,
        )
    except ValueError as exc:
        raise RunConfigurationError(str(exc)) from exc

    return _PreparedRun(
        config=config,
        evaluator_dir=evaluator_dir,
        dataset_path=dataset_path,
        selected=selected,
        task_manifest=task_manifest,
        run_manifest=run_manifest,
        mini_api_key=mini_api_key,
        skip_existing_patches=skip_existing_patches,
        overrides=tuple(overrides),
    )


def _validate_config(config: RunConfig) -> None:
    if bool(config.repo) == config.all_repos:
        raise RunConfigurationError("Pass either --repo NAME or --all-repos.")
    if config.concurrency < 1:
        raise RunConfigurationError(f"--concurrency must be >=1, got {config.concurrency}")
    if config.n_runs < 1:
        raise RunConfigurationError(f"--n-runs must be >=1, got {config.n_runs}")
    if config.limit is not None and config.limit < 0:
        raise RunConfigurationError(f"--limit must be >=0, got {config.limit}")
    if config.overall_timeout is not None and config.overall_timeout <= 0:
        raise RunConfigurationError("--overall-timeout must be >0.")
    if config.base_url is not None:
        parsed = urlsplit(config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RunConfigurationError("--base-url must be an absolute HTTP(S) URL.")
    if config.compact_after_each_task and not config.sequential:
        raise RunConfigurationError(
            "--compact-after-each-task only makes sense in --sequential mode."
        )
    if config.retry and not config.sequential:
        raise RunConfigurationError("--retry only makes sense in --sequential mode.")
    if not config.model and config.experiment_model is None:
        raise RunConfigurationError(
            "no model selected: pass --config FILE (with a `model:` block), "
            "or --model NAME."
        )
    if config.provider not in SUPPORTED_PROVIDERS:
        raise RunConfigurationError(f"unsupported provider: {config.provider!r}")


def _parse_set_overrides(config: RunConfig) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in (*config.configured_overrides, *config.set_overrides):
        if "=" not in item:
            raise RunConfigurationError(f"--set expects DOTTED.KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        if not key.strip():
            raise RunConfigurationError(f"--set has an empty key: {item!r}")
        parsed.append((key.strip(), value))
    return parsed


def _select_tasks(
    config: RunConfig,
    *,
    all_tasks: list[Task],
    frozen: dict[str, list[str]],
    emit: MessageSink,
) -> _SelectedTasks:
    cap_per_repo = None if config.full_dataset else DEFAULT_REPO_TASK_CAP
    task_suite = "full" if config.full_dataset else f"repo{DEFAULT_REPO_TASK_CAP}"
    manifest_loaded = False
    plans: list[TaskUniversePlan] | None = None
    if config.retry:
        plans = recovery.load_task_manifest_universes(
            config.output_dir / "task_manifest.json",
            all_tasks,
            repo=None if config.all_repos else config.repo,
        )
        if plans is None:
            raise RunExecutionError(
                f"--retry requires an existing {config.output_dir / 'task_manifest.json'}"
            )
        manifest_loaded = True
        task_suite = "manifest"
        cap_per_repo = None
        if config.limit is not None:
            emit("[swebench-seq] retry loaded task_manifest.json; ignoring --limit")
        emit("[swebench-seq] retry loaded task_manifest.json")
    if plans is None:
        plans = plan_task_universes(
            all_tasks,
            repo=None if config.all_repos else config.repo,
            frozen_order=frozen,
            cap_per_repo=cap_per_repo,
        )
        if config.limit is not None:
            plans = cap_task_universe_plans(plans, config.limit)

    explicit_target_ids = list(config.target_instances)
    if config.target_instances_file is not None:
        try:
            explicit_target_ids.extend(read_instance_ids_file(config.target_instances_file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RunConfigurationError(
                f"failed to read --target-instances-file "
                f"{config.target_instances_file}: {exc}"
            ) from exc
    if explicit_target_ids:
        plans = _filter_task_plans(plans, explicit_target_ids)
        if not plans:
            raise RunConfigurationError(
                "--target-instance(s)/-file matched no tasks in the planned suite."
            )
        emit(
            f"[swebench-seq] instance allowlist: "
            f"{sum(len(plan.tasks) for plan in plans)} task(s) across "
            f"{len(plans)} repo(s)"
        )

    if not plans:
        if config.repo:
            raise RunExecutionError(f"no tasks found for repo={config.repo!r}")
        raise RunExecutionError("no tasks selected")
    return _SelectedTasks(
        plans=tuple(plans),
        task_suite=task_suite,
        cap_per_repo=cap_per_repo,
        manifest_loaded=manifest_loaded,
        explicit_target_ids=tuple(explicit_target_ids),
    )


def _filter_task_plans(
    plans: Iterable[TaskUniversePlan], target_ids: Iterable[str]
) -> list[TaskUniversePlan]:
    allowed = {instance_id.split("instance_")[-1] for instance_id in target_ids}
    filtered: list[TaskUniversePlan] = []
    for plan in plans:
        tasks = tuple(
            task
            for task in plan.tasks
            if task.instance_id.split("instance_")[-1] in allowed
        )
        if tasks:
            filtered.append(replace(plan, tasks=tasks))
    return filtered


def _emit_selection(prepared: _PreparedRun, *, emit: MessageSink) -> None:
    config = prepared.config
    selected = prepared.selected
    selected_task_count = sum(len(plan) for plan in selected.plans)
    repo_scope = "across all repos" if config.all_repos else f"for repo={config.repo}"
    cap_label = "none" if selected.cap_per_repo is None else str(selected.cap_per_repo)
    limit_label = (
        ""
        if config.limit is None or selected.manifest_loaded
        else f", limit={config.limit}"
    )
    emit(
        f"[swebench-seq] task_suite={selected.task_suite} cap_per_repo={cap_label}"
        f"{limit_label}: selected {selected_task_count} task(s) "
        f"in {len(selected.plans)} repo universe(s) {repo_scope}"
    )
    for plan in selected.plans:
        dated = sum(task.commit_date is not None for task in plan.tasks)
        emit(
            f"[swebench-seq] universe {plan.repo}: "
            f"{len(plan)}/{plan.available_task_count} task(s), "
            f"order={plan.order_source}, dated={dated}/{len(plan)}"
        )


def _execute_sequential(
    prepared: _PreparedRun,
    *,
    run_roots: list[Path],
    accumulators: dict[int, PredictionsAccumulator],
    mini_options: MiniSweAgentOptions,
    container_env: dict[str, str],
    emit: MessageSink,
) -> None:
    config = prepared.config
    base_universes = [
        (plan.repo, list(plan.tasks)) for plan in prepared.selected.plans
    ]
    all_universes = [
        (run_id, repo, tasks)
        for run_id in range(config.n_runs)
        for repo, tasks in base_universes
    ]
    agent_home_name = "_agent_home_retry" if config.retry else "_agent_home"
    if config.retry:
        work: list[tuple[int, str, list[Task], bool, str]] = []
        recovered_tasks = 0
        for run_id, repo, tasks in all_universes:
            retry_target = recovery.first_retryable_task(
                run_root=run_roots[run_id],
                universe_repo=repo,
                tasks=tasks,
            )
            if retry_target is None:
                continue
            start_index, failed_status = retry_target
            recovery.restore_sequential_recovery_state(
                run_root=run_roots[run_id],
                universe_repo=repo,
                tasks=tasks,
                start_index=start_index,
                agent_home_name=agent_home_name,
                status=failed_status,
            )
            suffix = tasks[start_index:]
            recovered_tasks += len(suffix)
            work.append((run_id, repo, suffix, start_index > 0, agent_home_name))
            emit(
                f"[swebench-seq] retry run_{run_id} {repo}: "
                f"from {tasks[start_index].instance_id} "
                f"(status={failed_status}, {len(suffix)} task suffix)"
            )
        emit(
            f"[swebench-seq] retry: {len(work)} universe suffix(es), "
            f"{recovered_tasks} task(s)"
        )
    else:
        work = [
            (run_id, repo, tasks, False, agent_home_name)
            for run_id, repo, tasks in all_universes
        ]

    emit(
        f"[swebench-seq] n_runs={config.n_runs} x {len(base_universes)} "
        f"universe(s) = {len(all_universes)} total, shared pool "
        f"concurrency={config.concurrency}"
    )

    def run_one(
        run_id: int,
        repo: str,
        tasks: list[Task],
        memory_seeded: bool,
        current_agent_home_name: str,
    ) -> None:
        execution.run_universe_sequential(
            run_id=run_id,
            universe_repo=repo,
            tasks=tasks,
            output_dir=run_roots[run_id],
            image_prefix=config.image_prefix,
            evaluator_dir=prepared.evaluator_dir,
            dataset_path=prepared.dataset_path,
            env=container_env,
            overall_timeout=config.overall_timeout,
            context_window=config.context_window,
            compact_after_each_task=config.compact_after_each_task,
            mini_options=mini_options,
            accumulator=accumulators[run_id],
            skip_existing_patches=prepared.skip_existing_patches,
            memory_seeded_initial=memory_seeded,
            agent_home_name=current_agent_home_name,
        )

    if config.concurrency == 1 or len(work) == 1:
        for args in work:
            run_one(*args)
        return

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {
            pool.submit(run_one, *args): (args[0], args[1])
            for args in work
        }
        for future in as_completed(futures):
            run_id, repo = futures[future]
            try:
                future.result()
            except Exception as exc:
                emit(
                    f"[swebench-seq] run={run_id} universe {repo} raised: {exc}",
                    err=True,
                )


def _execute_independent(
    prepared: _PreparedRun,
    *,
    run_roots: list[Path],
    accumulators: dict[int, PredictionsAccumulator],
    mini_options: MiniSweAgentOptions,
    container_env: dict[str, str],
    emit: MessageSink,
) -> None:
    config = prepared.config
    tasks = list(prepared.tasks)
    work = [
        (run_id, task)
        for run_id in range(config.n_runs)
        for task in tasks
    ]
    emit(
        f"[swebench-seq] n_runs={config.n_runs} x {len(tasks)} task(s) "
        f"= {len(work)} total, shared pool concurrency={config.concurrency}"
    )
    tag_lookup = evaluation.load_tag_lookup(prepared.dataset_path)
    emit(
        f"[swebench-seq] inline scoring (tag_lookup loaded, "
        f"{len(tag_lookup)} tags)"
    )

    def run_one(run_id: int, task: Task) -> None:
        try:
            task_execution = execution.run_task(
                task=task,
                output_dir=run_roots[run_id],
                image_prefix=config.image_prefix,
                env=container_env,
                overall_timeout=config.overall_timeout,
                mini_options=mini_options,
                accumulator=accumulators[run_id],
                skip_existing_patches=prepared.skip_existing_patches,
                nest_by_repo=config.all_repos,
                run_id=run_id,
            )
        except Exception as exc:
            emit(
                f"[swebench-seq] run={run_id} task {task.instance_id} "
                f"rollout raised: {exc}",
                err=True,
            )
            return

        repo_dir = run_roots[run_id]
        if config.all_repos:
            repo_dir /= task.repo.replace("/", "__")
        if task_execution.rollout_status not in recovery.TERMINAL_EXIT_STATUSES:
            emit(
                f"[swebench-seq] {task.instance_id} → score: INFRA-FAIL "
                f"(rollout status {task_execution.rollout_status or 'missing'})",
                err=True,
            )
            return
        patch_path = repo_dir / task.instance_id / "patch.diff"
        patch = patch_path.read_text() if patch_path.is_file() else ""
        cached = evaluation.get_cached_eval_result(repo_dir, task.instance_id, patch)
        if cached is not None:
            emit(f"[swebench-seq] {task.instance_id} → score: {cached} (cached)")
            return
        try:
            outcome = execution.grade_task(
                repo_dir=repo_dir,
                instance_id=task.instance_id,
                patch=patch,
                image_prefix=config.image_prefix,
                tag_lookup=tag_lookup,
                evaluator_dir=prepared.evaluator_dir,
                dataset_path=prepared.dataset_path,
            )
        except Exception as exc:
            emit(f"[swebench-seq] {task.instance_id} → score raised: {exc}", err=True)
            return
        emit(
            f"[swebench-seq] {task.instance_id} → score: "
            f"{outcome if outcome is not None else 'INFRA-FAIL'}"
        )

    if config.concurrency == 1:
        for args in work:
            run_one(*args)
        return

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {
            pool.submit(run_one, *args): (args[0], args[1].instance_id)
            for args in work
        }
        for future in as_completed(futures):
            run_id, instance_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                emit(
                    f"[swebench-seq] run={run_id} task {instance_id} raised: {exc}",
                    err=True,
                )


def _finalize_runs(
    prepared: _PreparedRun,
    *,
    run_roots: list[Path],
    accumulators: dict[int, PredictionsAccumulator],
    emit: MessageSink,
) -> None:
    config = prepared.config
    completion_errors: list[str] = []
    tasks = list(prepared.tasks)
    for run_id, run_root in enumerate(run_roots):
        final_path = accumulators[run_id].flush()
        write_manifest(
            run_root / "image_manifest.json",
            build_image_manifest(
                run_root=run_root,
                tasks=tasks,
                sequential=config.sequential,
                all_repos=config.all_repos,
            ),
        )
        emit(
            f"[swebench-seq] run_{run_id}: wrote "
            f"{len(accumulators[run_id])} prediction(s) → {final_path}"
        )
        completion_errors.extend(
            execution.run_completion_errors(
                run_root=run_root,
                tasks=tasks,
                sequential=config.sequential,
                all_repos=config.all_repos,
            )
        )
    if completion_errors:
        preview = "\n".join(f"  - {item}" for item in completion_errors[:10])
        remainder = len(completion_errors) - 10
        remainder_line = f"\n  - ... and {remainder} more" if remainder > 0 else ""
        raise RunExecutionError(f"run incomplete:\n{preview}{remainder_line}")
