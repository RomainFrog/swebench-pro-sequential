"""Tests for `swebench-seq run --config` self-contained experiment configs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from swebench_seq.cli import main
from swebench_seq.container_network import proxy_env_for
from swebench_seq.experiment import (
    experiment_defaults,
    normalize_set_overrides,
    read_instance_ids_file,
    resolve_experiment_config_path,
)
from swebench_seq.orchestration.run import resolve_api_key

_BUILTINS = [
    "public-smoke-responses-podman",
    "sequential-responses-podman",
    "independent-responses-podman",
]

_RESPONSES_MODEL = "swebench_seq.agent.responses.OpenAIResponsesModel"


def _run_param_names() -> set[str]:
    run_cmd = main.commands["run"]
    return {p.name for p in run_cmd.params if p.name and p.name != "experiment_config"}


# ---- pure helpers -------------------------------------------------------------


def test_experiment_defaults_maps_known_keys() -> None:
    d = experiment_defaults(
        {"all_repos": True, "n_runs": 3, "provider": "openai"},
        {"all_repos", "n_runs", "provider"},
    )
    assert d == {"all_repos": True, "n_runs": 3, "provider": "openai"}


def test_read_instance_ids_file_supports_newlines_json_and_jsonl(tmp_path: Path) -> None:
    newline = tmp_path / "ids.txt"
    newline.write_text("# comment\ntask-1\n\ntask-2\n")
    assert read_instance_ids_file(newline) == ["task-1", "task-2"]

    json_list = tmp_path / "ids.json"
    json_list.write_text(json.dumps(["task-3", "task-4"]))
    assert read_instance_ids_file(json_list) == ["task-3", "task-4"]

    jsonl = tmp_path / "ids.jsonl"
    jsonl.write_text('{"instance_id": "task-5"}\n{"instance_id": "task-6"}\n')
    assert read_instance_ids_file(jsonl) == ["task-5", "task-6"]


def test_experiment_defaults_rejects_unknown_key() -> None:
    with pytest.raises(Exception, match="unknown experiment-config key"):
        experiment_defaults({"conkurrency": 5}, {"concurrency"})


@pytest.mark.parametrize("key", ["config_overrides", "set_overrides"])
def test_experiment_defaults_rejects_internal_override_keys(key: str) -> None:
    with pytest.raises(Exception, match="unknown experiment-config key"):
        experiment_defaults({key: []}, {"concurrency", "set_overrides"})


def test_experiment_defaults_set_mapping_becomes_overrides() -> None:
    d = experiment_defaults(
        {"set": {"model.model_kwargs.reasoning_effort": "low"}}, {"all_repos"}
    )
    assert d["set_overrides"] == ["model.model_kwargs.reasoning_effort=low"]


def test_experiment_defaults_set_list_form() -> None:
    d = experiment_defaults(
        {"set": ["model.model_kwargs.reasoning_effort=high"]}, {"all_repos"}
    )
    assert d["set_overrides"] == ["model.model_kwargs.reasoning_effort=high"]


def test_normalize_set_overrides_rejects_bad_list_item() -> None:
    with pytest.raises(Exception, match="list items must be"):
        normalize_set_overrides(["no-equals-here"])


def test_resolve_experiment_config_path_builtin_and_unknown() -> None:
    assert resolve_experiment_config_path("sequential-responses-podman").is_file()
    with pytest.raises(Exception, match="experiment config not found"):
        resolve_experiment_config_path("does-not-exist-xp")


def test_rollout_proxy_env_is_empty_without_explicit_proxy(monkeypatch) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)

    env = proxy_env_for("rollout")

    assert env == {}


def test_rollout_proxy_env_preserves_proxy_and_merges_no_proxy(monkeypatch) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://custom-http:8080")
    monkeypatch.setenv("https_proxy", "http://custom-https:8080")
    monkeypatch.setenv("NO_PROXY", "example.com,localhost")

    env = proxy_env_for("rollout")

    assert env["HTTP_PROXY"] == "http://custom-http:8080"
    assert env["http_proxy"] == "http://custom-http:8080"
    assert env["HTTPS_PROXY"] == "http://custom-https:8080"
    assert env["https_proxy"] == "http://custom-https:8080"
    no_proxy = env["NO_PROXY"].split(",")
    assert env["no_proxy"] == env["NO_PROXY"]
    assert "example.com" in no_proxy
    assert no_proxy.count("localhost") == 1
    assert "127.0.0.1" in no_proxy


def test_resolve_api_key_from_named_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODEL_API_KEY", "secret")

    assert (
        resolve_api_key(
            env_name="TEST_MODEL_API_KEY",
            provider="openai",
        )
        == "secret"
    )


def test_resolve_api_key_rejects_missing_named_environment(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_MODEL_API_KEY", raising=False)

    with pytest.raises(Exception, match="MISSING_MODEL_API_KEY"):
        resolve_api_key(
            env_name="MISSING_MODEL_API_KEY",
            provider="openai",
        )


def test_doctor_config_rejects_missing_explicit_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEFINITELY_MISSING_KEY", raising=False)
    config = _write(
        tmp_path,
        "provider: openai\n"
        "api_key_env: DEFINITELY_MISSING_KEY\n"
        + _MODEL_BLOCK,
    )

    result = CliRunner().invoke(main, ["doctor", "--config", config])

    assert result.exit_code != 0
    assert "DEFINITELY_MISSING_KEY" in result.output


def test_doctor_config_rejects_invalid_endpoint(tmp_path: Path) -> None:
    config = _write(
        tmp_path,
        "provider: openai\nbase_url: not-a-url\n" + _MODEL_BLOCK,
    )

    result = CliRunner().invoke(main, ["doctor", "--config", config])

    assert result.exit_code != 0
    assert "absolute HTTP(S) URL" in result.output


def test_doctor_config_validates_model_without_calling_it(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text("{}\n")
    config = _write(
        tmp_path,
        "provider: openai\n"
        "api_key_env: TEST_MODEL_API_KEY\n"
        f"dataset: {dataset_path}\n"
        + _MODEL_BLOCK,
    )
    monkeypatch.setenv("TEST_MODEL_API_KEY", "secret")
    monkeypatch.setattr(
        "swebench_seq.evaluation.runner.validate_evaluation_environment",
        lambda evaluator_dir: (tmp_path / "evaluator", "unix:///podman.sock"),
    )
    monkeypatch.setattr("swebench_seq.cli.load_tasks", lambda dataset_path: [])
    monkeypatch.setattr("swebench_seq.cli.PodmanCli.version", lambda self: "5.0.0")

    result = CliRunner().invoke(main, ["doctor", "--config", config])

    assert result.exit_code == 0, result.output
    assert "Podman 5.0.0" in result.output
    assert "model=gpt-5-x" in result.output
    assert "TEST_MODEL_API_KEY set" in result.output
    assert "secret" not in result.output


# ---- built-in templates are valid + self-contained ----------------------------


@pytest.mark.parametrize("name", _BUILTINS)
def test_builtin_experiment_config_is_valid(name: str) -> None:
    path = resolve_experiment_config_path(name)
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    allowed = _run_param_names() | {"set", "model"}
    unknown = set(data) - allowed
    assert not unknown, f"{name} has unknown keys: {unknown}"
    assert "api_key" not in data  # never commit secrets
    # self-contained: an inline model block with a model_class + name
    model = data.get("model")
    assert isinstance(model, dict)
    assert model.get("model_class")
    assert model.get("model_name")
    # Anthropic thinking configs must not also set temperature
    if "thinking" in model.get("model_kwargs", {}):
        assert "temperature" not in model["model_kwargs"]


# ---- CLI integration (config -> Click default_map + ctx.meta model) -----------


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "xp.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


_MODEL_BLOCK = f"model:\n  model_name: gpt-5-x\n  model_class: {_RESPONSES_MODEL}\n"


def test_config_values_reach_run_body(tmp_path: Path) -> None:
    # concurrency=0 from the config must trip the >=1 check in the body, proving
    # output_dir (required), all_repos, and concurrency all applied.
    cfg = _write(
        tmp_path,
        "all_repos: true\noutput_dir: /tmp/xp-test\nconcurrency: 0\n" + _MODEL_BLOCK,
    )
    res = CliRunner().invoke(main, ["run", "--config", cfg])
    assert res.exit_code != 0
    assert "concurrency must be >=1" in res.output


def test_cli_flag_overrides_config(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        "all_repos: true\noutput_dir: /tmp/xp-test\nconcurrency: 0\nn_runs: 0\n"
        + _MODEL_BLOCK,
    )
    res = CliRunner().invoke(main, ["run", "--config", cfg, "--concurrency", "5"])
    assert "concurrency must be >=1" not in res.output  # CLI overrode config 0
    assert "n-runs must be >=1" in res.output


def test_overall_timeout_must_be_positive(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        "all_repos: true\noutput_dir: /tmp/xp-test\noverall_timeout: 0\n"
        + _MODEL_BLOCK,
    )

    res = CliRunner().invoke(main, ["run", "--config", cfg])

    assert res.exit_code != 0
    assert "overall-timeout must be >0" in res.output


def test_retry_is_sequential_only(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        "all_repos: true\noutput_dir: /tmp/xp-test\nretry: true\n"
        + _MODEL_BLOCK,
    )

    res = CliRunner().invoke(main, ["run", "--config", cfg])

    assert res.exit_code != 0
    assert "--retry only makes sense in --sequential mode" in res.output


def test_public_help_excludes_unsafe_or_source_checkout_flags() -> None:
    res = CliRunner().invoke(main, ["run", "--help"])

    assert res.exit_code == 0
    option_names = {
        line.strip().split()[0]
        for line in res.output.splitlines()
        if line.startswith("  --")
    }
    assert "--api-key" not in option_names
    assert "--mini-swe-agent-path" not in option_names
    assert "--sequential-recover-status" not in option_names
    assert "--api-key-env" in option_names
    assert "--compact-after-each-task" in option_names
    assert "--dataset" in option_names
    assert "--dry-run" in option_names
    assert "--retry" in option_names
    assert "--independent" in res.output
    assert "--rerun-existing-patches" in res.output


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "all_repos: true\nconkurrency: 5\n")
    res = CliRunner().invoke(main, ["run", "--config", cfg])
    assert res.exit_code != 0
    assert "unknown experiment-config key" in res.output


def test_inline_model_satisfies_model_requirement(tmp_path: Path) -> None:
    # A model: block means the "no model selected" check passes, so we reach the
    # concurrency check instead.
    cfg = _write(
        tmp_path,
        "all_repos: true\noutput_dir: /tmp/xp-test\nconcurrency: 0\n" + _MODEL_BLOCK,
    )
    res = CliRunner().invoke(main, ["run", "--config", cfg])
    assert "no model selected" not in res.output
    assert "concurrency must be >=1" in res.output


def test_config_can_define_explicit_target_instances(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        "all_repos: true\n"
        "output_dir: /tmp/xp-test\n"
        "target_instances:\n"
        "  - instance-a\n"
        "  - instance-b\n"
        "concurrency: 0\n"
        + _MODEL_BLOCK,
    )
    res = CliRunner().invoke(main, ["run", "--config", cfg])

    assert res.exit_code != 0
    assert "concurrency must be >=1" in res.output


def test_compact_after_each_task_is_sequential_only(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        "all_repos: true\n"
        "output_dir: /tmp/xp-test\n"
        "concurrency: 1\n"
        "compact_after_each_task: true\n"
        + _MODEL_BLOCK,
    )
    res = CliRunner().invoke(main, ["run", "--config", cfg])

    assert res.exit_code != 0
    assert "compact-after-each-task only makes sense" in res.output


def test_missing_model_is_rejected(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "all_repos: true\noutput_dir: /tmp/xp-test\nconcurrency: 2\n")
    res = CliRunner().invoke(main, ["run", "--config", cfg])
    assert "no model selected" in res.output


def test_non_mapping_model_is_rejected(tmp_path: Path) -> None:
    cfg = _write(tmp_path, "all_repos: true\noutput_dir: /tmp/x\nmodel: not-a-mapping\n")
    res = CliRunner().invoke(main, ["run", "--config", cfg])
    assert res.exit_code != 0
    assert "must be a mapping" in res.output


def test_builtin_name_resolves_via_cli(tmp_path: Path) -> None:
    # --concurrency 0 forces the early body failure, proving the built-in loaded.
    res = CliRunner().invoke(
        main,
        ["run", "--config", "sequential-responses-podman", "--concurrency", "0"],
    )
    assert "concurrency must be >=1" in res.output
    assert "not found" not in res.output


def test_run_preflight_fails_before_creating_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SWEBENCH_PRO_EVALUATOR_DIR", raising=False)

    def fail_preflight(_evaluator_dir: Path | None) -> None:
        raise RuntimeError("Podman unavailable")

    monkeypatch.setattr(
        "swebench_seq.evaluation.runner.validate_evaluation_environment",
        fail_preflight,
    )
    output_dir = tmp_path / "must-not-exist"

    res = CliRunner().invoke(
        main,
        [
            "run",
            "--repo",
            "flipt-io/flipt",
            "--model",
            "test-model",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert res.exit_code != 0
    assert "evaluation preflight failed" in res.output
    assert not output_dir.exists()


def test_dry_run_uses_local_dataset_without_writing_or_requiring_key(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "instance_id": "instance_org__repo-1",
                "repo": "org/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "dockerhub_tag": "org.repo-1",
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "must-not-exist"

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--repo",
            "org/repo",
            "--model",
            "test-model",
            "--api-key-env",
            "MISSING_TEST_KEY",
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "no containers, model calls, grading, or artifacts" in result.output
    assert "instance_org__repo-1" in result.output
    assert "docker.io/jefzda/sweap-images:org.repo-1" in result.output
    assert '"api_key_configured": false' in result.output
    assert not output_dir.exists()


def test_explicit_flags_override_boolean_config_defaults(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "instance_id": "instance_org__repo-1",
                "repo": "org/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "dockerhub_tag": "org.repo-1",
            }
        )
        + "\n"
    )
    config = _write(
        tmp_path,
        "all_repos: true\n"
        "sequential: true\n"
        "skip_existing_patches: true\n"
        f"dataset: {dataset_path}\n"
        f"output_dir: {tmp_path / 'output'}\n"
        + _MODEL_BLOCK,
    )

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--config",
            config,
            "--repo",
            "org/repo",
            "--independent",
            "--rerun-existing-patches",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{\n") :])
    assert payload["run_manifest"]["execution"]["mode"] == "independent"
    assert payload["run_manifest"]["execution"]["skip_existing_patches"] is False
    assert payload["run_manifest"]["selection"] == {"repositories": 1, "tasks": 1}


def test_cli_set_layers_over_config_set(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "instance_id": "instance_org__repo-1",
                "repo": "org/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix it",
                "dockerhub_tag": "org.repo-1",
            }
        )
        + "\n"
    )
    config = _write(
        tmp_path,
        "repo: org/repo\n"
        f"dataset: {dataset_path}\n"
        f"output_dir: {tmp_path / 'output'}\n"
        "set:\n"
        "  agent.step_limit: 1\n"
        "  model.model_kwargs.reasoning_effort: medium\n"
        + _MODEL_BLOCK,
    )

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--config",
            config,
            "--set",
            "model.model_kwargs.reasoning_effort=low",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{\n") :])
    assert len(payload["run_manifest"]["experiment_config_sha256"]) == 64
    assert payload["run_manifest"]["model"]["overrides"] == [
        {"key": "agent.step_limit", "value": 1},
        {"key": "model.model_kwargs.reasoning_effort", "value": "medium"},
        {"key": "model.model_kwargs.reasoning_effort", "value": "low"},
    ]
    assert (
        payload["run_manifest"]["model"]["config"]["model_kwargs"][
            "reasoning_effort"
        ]
        == "low"
    )
