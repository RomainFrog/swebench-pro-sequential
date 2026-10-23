"""The ``swebench-seq run`` command."""

from __future__ import annotations

from pathlib import Path

import click
from click.core import ParameterSource

from ..constants import IMAGE_PREFIX_DEFAULT
from ..experiment import ExperimentConfigError, experiment_defaults, load_experiment_config
from ..orchestration.run import (
    SUPPORTED_PROVIDERS,
    RunConfig,
    RunConfigurationError,
    RunExecutionError,
    execute_run,
)

_EXPERIMENT_MODEL_KEY = "swebench_seq.experiment_model"
_EXPERIMENT_CONFIG_SOURCE_KEY = "swebench_seq.experiment_config_source"
_EXPERIMENT_CONFIG_PATH_KEY = "swebench_seq.experiment_config_path"
_EXPERIMENT_OVERRIDES_KEY = "swebench_seq.experiment_overrides"


def _apply_experiment_config(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Load experiment values as defaults while preserving explicit CLI flags."""

    del param
    if not value:
        return value
    try:
        path, data = load_experiment_config(value)
        model = data.pop("model", None)
        if model is not None and not isinstance(model, dict):
            raise ExperimentConfigError("experiment-config `model:` must be a mapping")
        allowed = {
            option.name
            for option in ctx.command.params
            if option.name and option.name != "experiment_config"
        }
        defaults = experiment_defaults(data, allowed)
    except ExperimentConfigError as exc:
        raise click.UsageError(str(exc)) from exc

    ctx.meta[_EXPERIMENT_MODEL_KEY] = model
    ctx.meta[_EXPERIMENT_CONFIG_SOURCE_KEY] = value
    ctx.meta[_EXPERIMENT_CONFIG_PATH_KEY] = path
    ctx.meta[_EXPERIMENT_OVERRIDES_KEY] = tuple(defaults.pop("set_overrides", []))
    ctx.default_map = {**(ctx.default_map or {}), **defaults}
    return value


@click.command("run")
@click.option(
    "--config",
    "experiment_config",
    callback=_apply_experiment_config,
    is_eager=True,
    expose_value=False,
    help="Experiment config: a built-in name (see src/swebench_seq/configs/) "
    "or a path to a YAML file defining a full run — task "
    "selection, mode, concurrency, n_runs, an inline `model:` block, timeouts, "
    "output_dir. Its values become defaults; explicit CLI flags override them. "
    "Unknown keys are rejected.",
)
@click.option(
    "--repo",
    help="Single upstream repo to run (e.g. qutebrowser/qutebrowser).",
)
@click.option(
    "--all-repos",
    is_flag=True,
    default=False,
    help="Run every repo in the dataset (mutually exclusive with --repo).",
)
@click.option(
    "--sequential/--independent",
    default=False,
    help="Persist /home/agent/ across tasks within each repo (universe). "
    "First task in a universe runs without bind-mounting agent-home, then "
    "we extract /home/agent/. into a host dir; subsequent tasks mount it. "
    "Tasks within a universe always run serially (sharing /home/agent); "
    "--concurrency controls parallel UNIVERSES instead.",
)
@click.option("--limit", type=int, default=None, help="First N tasks per universe.")
@click.option(
    "--target-instance",
    "target_instances",
    multiple=True,
    help="Restrict the run to an explicit instance id. Repeatable.",
)
@click.option(
    "--target-instances-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Restrict the run using a newline, JSON-list, or JSONL file of instance ids.",
)
@click.option(
    "--full-dataset",
    is_flag=True,
    default=False,
    help="Non-standard run: disable the official 40-task-per-repository cap and "
    "run the full SWE-bench Pro task set.",
)
@click.option(
    "--concurrency",
    type=int,
    default=1,
    help="Parallel tasks (non-sequential mode) OR parallel UNIVERSES "
    "(sequential mode — each universe still runs its tasks serially, "
    "but multiple universes can run in parallel).",
)
@click.option(
    "--n-runs",
    type=int,
    default=1,
    show_default=True,
    help="Run the eval N times (each into output_dir/run_N/). All runs share "
    "ONE concurrency pool of --concurrency workers — so --n-runs 3 "
    "--concurrency 40 still caps at 40 simultaneous containers total. "
    "Use to measure run-to-run variance without multiplying machine load. "
    "Layout is always nested under run_N/, even for N=1.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where artifacts + patches go.",
)
@click.option(
    "--image-prefix",
    default=IMAGE_PREFIX_DEFAULT,
    show_default=True,
    help="Per-task image registry prefix used by task containers and inline "
    "grading. Defaults to the public SWE-bench Pro images.",
)
@click.option(
    "--evaluator-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    envvar="SWEBENCH_PRO_EVALUATOR_DIR",
    help="Optional checkout of the public SWE-bench Pro evaluator. By default "
    "a pinned revision is downloaded and cached automatically.",
)
@click.option(
    "--dataset",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Explicit SWE-bench Pro JSONL/CSV file. By default the pinned public "
    "Hugging Face dataset is downloaded and cached automatically.",
)
@click.option(
    "--overall-timeout",
    type=float,
    default=None,
    help="Per-task agent-rollout wall-time cap in seconds. Container setup and "
    "grading are excluded. If omitted, only per-request and per-command "
    "timeouts apply.",
)
@click.option(
    "--context-window",
    type=int,
    default=200_000,
    show_default=True,
    help="Model context-window in tokens. Sequential mini compaction keeps a "
    "20K-token reserve; raise (e.g. 1_000_000) for the 1M-context variant to "
    "suppress mid-task compactions.",
)
@click.option(
    "--compact-after-each-task/--no-compact-after-each-task",
    default=False,
    show_default=True,
    help="Sequential-only: after each completed task, force a strict "
    "compaction so the next task starts from summarized history instead of "
    "raw previous-task turns.",
)
@click.option("--model", default=None, help="Model id passed to mini-swe-agent / LiteLLM.")
@click.option(
    "--api-key-env",
    default=None,
    metavar="NAME",
    help="Read the model API key from this environment variable. This keeps "
    "provider credentials explicit without storing them in YAML or argv.",
)
@click.option(
    "--provider",
    default="openai",
    type=click.Choice(SUPPORTED_PROVIDERS),
    show_default=True,
    help="Provider id (openai / anthropic / openrouter). Used to resolve the "
    "host-side API key and base URL for mini-swe-agent.",
)
@click.option("--base-url", default=None, help="Override the provider's default base URL.")
@click.option(
    "--set",
    "set_overrides",
    multiple=True,
    metavar="DOTTED.KEY=VALUE",
    help="Override a single model config value, e.g. "
    "--set model.model_kwargs.reasoning_effort=low. Repeatable; the value is "
    "parsed as a YAML scalar and applied after the experiment config.",
)
@click.option(
    "--skip-existing-patches/--rerun-existing-patches",
    default=False,
    help="Reuse tasks with a non-empty patch, or a clean terminal empty patch. "
    "For sequential recovery from a failed task, use --retry.",
)
@click.option(
    "--retry",
    is_flag=True,
    default=False,
    help="Resume an existing sequential run from its first missing or failed "
    "task. Restores the prior conversation and agent-home snapshot, then "
    "reruns that task and the remaining suffix.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Resolve configuration and print the ordered task plan without "
    "starting Podman, calling a model, grading, or writing run artifacts.",
)
def run(
    repo: str | None,
    all_repos: bool,
    sequential: bool,
    limit: int | None,
    target_instances: tuple[str, ...],
    target_instances_file: Path | None,
    full_dataset: bool,
    concurrency: int,
    n_runs: int,
    output_dir: Path,
    image_prefix: str,
    evaluator_dir: Path | None,
    dataset: Path | None,
    overall_timeout: float | None,
    context_window: int,
    compact_after_each_task: bool,
    model: str | None,
    api_key_env: str | None,
    provider: str,
    base_url: str | None,
    set_overrides: tuple[str, ...],
    skip_existing_patches: bool,
    retry: bool,
    dry_run: bool,
) -> None:
    """Run SWE-bench Pro tasks through mini-swe-agent."""

    ctx = click.get_current_context()
    repo, all_repos = _resolve_repository_scope(ctx, repo=repo, all_repos=all_repos)
    config = RunConfig(
        repo=repo,
        all_repos=all_repos,
        sequential=sequential,
        limit=limit,
        target_instances=target_instances,
        target_instances_file=target_instances_file,
        full_dataset=full_dataset,
        concurrency=concurrency,
        n_runs=n_runs,
        output_dir=output_dir,
        image_prefix=image_prefix,
        evaluator_dir=evaluator_dir,
        dataset=dataset,
        overall_timeout=overall_timeout,
        context_window=context_window,
        compact_after_each_task=compact_after_each_task,
        model=model,
        api_key_env=api_key_env,
        provider=provider,
        base_url=base_url,
        set_overrides=set_overrides,
        skip_existing_patches=skip_existing_patches,
        retry=retry,
        dry_run=dry_run,
        experiment_model=ctx.meta.get(_EXPERIMENT_MODEL_KEY),
        experiment_config_source=ctx.meta.get(_EXPERIMENT_CONFIG_SOURCE_KEY),
        experiment_config_path=ctx.meta.get(_EXPERIMENT_CONFIG_PATH_KEY),
        configured_overrides=ctx.meta.get(_EXPERIMENT_OVERRIDES_KEY, ()),
    )
    try:
        execute_run(config, emit=_echo)
    except RunConfigurationError as exc:
        raise click.UsageError(str(exc)) from exc
    except RunExecutionError as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_repository_scope(
    ctx: click.Context, *, repo: str | None, all_repos: bool
) -> tuple[str | None, bool]:
    if not repo or not all_repos:
        return repo, all_repos
    repo_source = ctx.get_parameter_source("repo")
    all_repos_source = ctx.get_parameter_source("all_repos")
    if (
        repo_source is ParameterSource.COMMANDLINE
        and all_repos_source is not ParameterSource.COMMANDLINE
    ):
        return repo, False
    if (
        all_repos_source is ParameterSource.COMMANDLINE
        and repo_source is not ParameterSource.COMMANDLINE
    ):
        return None, all_repos
    raise click.UsageError("--repo and --all-repos are mutually exclusive.")


def _echo(message: str, *, err: bool = False) -> None:
    click.echo(message, err=err)
