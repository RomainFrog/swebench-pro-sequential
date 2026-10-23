"""Top-level command group and small operational commands."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click

from . import __version__
from .commands.run import run as run_command
from .evaluation.dataset import materialize_dataset_jsonl
from .experiment import ExperimentConfigError, experiment_defaults, load_experiment_config
from .orchestration.recovery import TERMINAL_EXIT_STATUSES
from .orchestration.run import SUPPORTED_PROVIDERS
from .report import load_report, validate_run
from .runner import PodmanCli
from .tasks import (
    load_frozen_sequential_order,
    load_tasks,
    validate_frozen_sequential_order,
)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Run SWE-bench Pro tasks through mini-swe-agent, optionally sequentially."""

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


main.add_command(run_command)


@main.command("doctor")
@click.option(
    "--config",
    "experiment_config",
    default=None,
    help="Validate a bundled or local experiment configuration, including its "
    "model and explicitly named API-key environment variable.",
)
@click.option(
    "--evaluator-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    envvar="SWEBENCH_PRO_EVALUATOR_DIR",
    help="Optional evaluator checkout override; the pinned public revision is "
    "downloaded and cached by default.",
)
@click.option(
    "--dataset",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit dataset override; otherwise validate the pinned Hugging Face dataset.",
)
def doctor(
    experiment_config: str | None,
    evaluator_dir: Path | None,
    dataset: Path | None,
) -> None:
    """Check configuration, Podman, evaluator, and data without calling a model."""

    from .evaluation.runner import validate_evaluation_environment

    config: dict[str, Any] = {}
    if experiment_config:
        try:
            _config_path, config = load_experiment_config(experiment_config)
            run_command_params = {
                parameter.name
                for parameter in run_command.params
                if parameter.name and parameter.name != "experiment_config"
            }
            config_without_model = dict(config)
            config_without_model.pop("model", None)
            experiment_defaults(config_without_model, run_command_params)
        except ExperimentConfigError as exc:
            raise click.ClickException(str(exc)) from exc

        model = config.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("model_name"), str):
            raise click.ClickException("experiment config requires model.model_name")
        if not isinstance(model.get("model_class"), str):
            raise click.ClickException("experiment config requires model.model_class")
        provider = config.get("provider", "openai")
        if provider not in SUPPORTED_PROVIDERS:
            raise click.ClickException(
                f"unsupported provider {provider!r}; choose from {SUPPORTED_PROVIDERS}"
            )
        base_url = config.get("base_url")
        if base_url is not None:
            parsed = urlsplit(base_url) if isinstance(base_url, str) else None
            if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise click.ClickException(
                    "experiment config base_url must be an absolute HTTP(S) URL"
                )
        key_env = config.get("api_key_env")
        if key_env is not None:
            if not isinstance(key_env, str) or not key_env:
                raise click.ClickException("experiment config api_key_env must be a name")
            if not os.environ.get(key_env):
                raise click.ClickException(
                    f"model API key environment variable is not set: {key_env}"
                )
        if evaluator_dir is None and isinstance(config.get("evaluator_dir"), str):
            evaluator_dir = Path(config["evaluator_dir"])
        if dataset is None and isinstance(config.get("dataset"), str):
            dataset = Path(config["dataset"])

    try:
        eval_repo, docker_host = validate_evaluation_environment(evaluator_dir)
        dataset_path = materialize_dataset_jsonl(dataset)
        tasks = load_tasks(dataset_path=dataset_path)
        if dataset is None:
            validate_frozen_sequential_order(tasks, load_frozen_sequential_order())
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"preflight failed: {exc}") from exc
    except ValueError as exc:
        raise click.ClickException(f"benchmark validation failed: {exc}") from exc
    podman_version = PodmanCli().version()
    free_gib = shutil.disk_usage(dataset_path.parent).free / (1024**3)
    click.echo(f"[doctor] Podman {podman_version}: ready ({docker_host})")
    click.echo(f"[doctor] evaluator: {eval_repo}")
    click.echo(f"[doctor] dataset: {dataset_path} ({len(tasks)} task(s))")
    click.echo(f"[doctor] cache filesystem free: {free_gib:.1f} GiB")
    if experiment_config:
        provider = str(config.get("provider") or "openai")
        model = config["model"]
        key_env = config.get("api_key_env")
        key_status = f", key={key_env} set" if key_env else ", no key required"
        click.echo(
            f"[doctor] model config: provider={provider}, model={model['model_name']}"
            f"{key_status}"
        )


@main.command("score")
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Run output directory containing predictions.json.",
)
@click.option(
    "--evaluator-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    envvar="SWEBENCH_PRO_EVALUATOR_DIR",
    help="Optional evaluator checkout override; the pinned public revision is "
    "downloaded and cached by default.",
)
@click.option(
    "--dataset",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit SWE-bench Pro JSONL/CSV override. By default the pinned "
    "public Hugging Face dataset is cached automatically.",
)
@click.option("--num-workers", type=int, default=1, show_default=True)
def score(
    run_dir: Path,
    evaluator_dir: Path | None,
    dataset: Path | None,
    num_workers: int,
) -> None:
    """Score a completed run via the upstream SWE-bench Pro evaluator."""

    predictions = run_dir / "predictions.json"
    if not predictions.exists():
        raise click.ClickException(
            f"missing {predictions} — did `swebench-seq run` complete and flush?"
        )
    if not predictions.read_text().strip() or json.loads(predictions.read_text()) == {}:
        raise click.ClickException(f"{predictions} is empty — no predictions to score.")
    from .evaluation.runner import score_predictions, validate_evaluation_environment

    try:
        evaluator_dir, _docker_host = validate_evaluation_environment(evaluator_dir)
    except RuntimeError as exc:
        raise click.ClickException(f"evaluation preflight failed: {exc}") from exc
    try:
        dataset = materialize_dataset_jsonl(dataset)
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"failed to load SWE-bench Pro dataset: {exc}") from exc
    output = run_dir / "score"
    run_report = load_report(run_dir)
    gradeable_instance_ids = {
        task.instance_id
        for task in run_report.tasks
        if task.rollout_status in TERMINAL_EXIT_STATUSES
    }
    skipped = len(run_report.tasks) - len(gradeable_instance_ids)
    if skipped:
        click.echo(
            f"[swebench-seq] skipping {skipped} nonterminal rollout(s); "
            "they remain infrastructure/incomplete outcomes"
        )
    if not gradeable_instance_ids:
        raise click.ClickException("no terminal rollouts are available to score.")

    click.echo(f"[swebench-seq] scoring with local Podman → {output}")
    rc = score_predictions(
        predictions_path=predictions,
        output_dir=output,
        evaluator_dir=evaluator_dir,
        dataset_path=dataset,
        num_workers=num_workers,
        gradeable_instance_ids=gradeable_instance_ids,
    )
    if rc != 0:
        raise click.ClickException(f"scoring failed (exit {rc}); see stderr above.")
    click.echo(f"[swebench-seq] wrote {output}")


@main.command("report")
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Run output directory to summarize.",
)
@click.option("--repo", default=None, help="Restrict the report to one repo.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def report_cmd(run_dir: Path, repo: str | None, fmt: str) -> None:
    """Summarize task outcomes and artifacts for one completed run."""

    report = load_report(run_dir)
    by_repo = report.by_repo()
    if repo:
        if repo not in by_repo:
            raise click.ClickException(
                f"repo {repo!r} has no tasks in {run_dir}. Available: {sorted(by_repo)}"
            )
        by_repo = {repo: by_repo[repo]}

    if fmt == "json":
        payload = {
            "schema_version": 1,
            "benchmark_version": report.benchmark_version,
            "canonical_inputs": report.canonical_inputs,
            "complete_suite": report.complete_suite,
            "run_dir": str(report.run_dir),
            "mode": report.mode,
            "summary": report.summary(),
            "repos": {
                repo_name: {
                    "tasks": [
                        {
                            "instance_id": task.instance_id,
                            "position": task.position,
                            "passed": task.passed,
                            "patch_bytes": task.patch_bytes,
                            "model_calls": task.model_calls,
                            "rollout_status": task.rollout_status,
                            "wall_seconds": task.wall_seconds,
                            "input_tokens": task.input_tokens,
                            "output_tokens": task.output_tokens,
                            "total_tokens": task.total_tokens,
                            "memory_md_bytes": task.memory_md_bytes,
                            "agent_home_bytes": task.agent_home_bytes,
                        }
                        for task in tasks
                    ],
                    "pass_rate_by_position": report.pass_rate_by_position(repo_name),
                }
                for repo_name, tasks in by_repo.items()
            },
        }
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    summary = report.summary()
    rate = summary["pass_rate"]
    rate_text = f"{rate:.1%}" if isinstance(rate, float) else "n/a"
    click.echo(
        f"[report] run_dir={report.run_dir} mode={report.mode} "
        f"benchmark={report.benchmark_version or '?'} "
        f"canonical_inputs={report.canonical_inputs} "
        f"complete_suite={report.complete_suite}"
    )
    click.echo(
        "[report] "
        f"selected={summary['selected']} completed={summary['completed_rollouts']} "
        f"graded={summary['graded']} passed={summary['passed']} "
        f"failed={summary['failed']} infra_or_incomplete="
        f"{summary['infrastructure_or_incomplete']} pass_rate={rate_text}"
    )
    click.echo(
        "[report] "
        f"model_calls={summary['model_calls']} "
        f"compaction_calls={summary['compaction_calls']} "
        f"wall_seconds={summary['wall_seconds']:.1f} "
        f"tokens={summary['total_tokens'] if summary['total_tokens'] is not None else '?'}"
    )
    for repo_name, tasks in by_repo.items():
        click.echo(f"\n=== {repo_name} ({len(tasks)} task(s)) ===")
        scored = [task for task in tasks if task.passed is not None]
        if scored:
            click.echo(
                f"  scored: {sum(1 for task in scored if task.passed)}/{len(scored)} "
                f"({sum(1 for task in scored if task.passed) / len(scored):.1%})"
            )
        for task in tasks:
            score_text = "PASS" if task.passed else ("FAIL" if task.passed is False else "?")
            memory = (
                f" mem={task.memory_md_bytes}B" if task.memory_md_bytes is not None else ""
            )
            home = (
                f" home={task.agent_home_bytes}B" if task.agent_home_bytes is not None else ""
            )
            click.echo(
                f"  pos={task.position} {score_text:4} "
                f"status={task.rollout_status or '?':16} "
                f"calls={task.model_calls:3} patch={task.patch_bytes:6}B"
                f"{memory}{home}  {task.instance_id}"
            )


@main.command("validate-run")
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Run output directory (for example, runs/example/run_0).",
)
def validate_run_cmd(run_dir: Path) -> None:
    """Validate that a run has complete scores, artifacts, and image provenance."""

    errors = validate_run(run_dir)
    if errors:
        rendered = "\n".join(f"  - {error}" for error in errors)
        raise click.ClickException(f"run is incomplete or non-reproducible:\n{rendered}")
    click.echo(f"[validate-run] complete with required provenance: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
