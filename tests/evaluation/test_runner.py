"""Tests for swebench_seq.evaluation.runner (inline single-task scoring API).

We don't invoke real podman / the upstream evaluator here — those are
network-and-disk-heavy. We mock _eval_one + _ensure_image and
verify score_one wires the result into score/eval_results.json
correctly, short-circuits empty patches, and reports None on infra fail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from swebench_seq.evaluation.runner import (
    _ensure_image,
    _eval_one,
    _evaluator_environment,
    _has_port_collision,
    _merge_eval_result,
    _podman_docker_host,
    _resolve_evaluator_paths,
    get_cached_eval_result,
    load_tag_lookup,
    score_one,
    score_predictions,
    validate_evaluation_environment,
)
from swebench_seq.runner import PodmanCli


def _make_evaluator_dir(path: Path) -> Path:
    path.mkdir()
    (path / "swe_bench_pro_eval.py").write_text("")
    (path / "run_scripts").mkdir()
    (path / "helper_code").mkdir()
    (path / "helper_code" / "image_uri.py").write_text("")
    (path / "dockerfiles" / "base_dockerfile").mkdir(parents=True)
    (path / "dockerfiles" / "instance_dockerfile").mkdir(parents=True)
    return path


def test_load_tag_lookup_uses_explicit_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.jsonl"
    dataset_path.write_text('{"instance_id":"iid","dockerhub_tag":"tag"}\n')
    load_tag_lookup.cache_clear()
    try:
        assert load_tag_lookup(dataset_path) == {"iid": "tag"}
    finally:
        load_tag_lookup.cache_clear()


# ---- _merge_eval_result ----


def test_resolve_evaluator_paths_uses_resolved_checkout(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "swe_bench_pro_eval.py").write_text("")
    (tmp_path / "run_scripts").mkdir()
    monkeypatch.setattr(
        "swebench_seq.evaluation.runner.resolve_evaluator_dir",
        lambda _path: tmp_path,
    )

    assert _resolve_evaluator_paths(None) == (
        tmp_path,
        tmp_path / "swe_bench_pro_eval.py",
        tmp_path / "run_scripts",
    )


def test_evaluator_environment_does_not_forward_model_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    env = _evaluator_environment("unix:///tmp/podman.sock")

    assert env["DOCKER_HOST"] == "unix:///tmp/podman.sock"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_podman_docker_host_uses_rootless_socket(monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/123")
    assert _podman_docker_host() == "unix:///run/user/123/podman/podman.sock"


def test_validate_evaluation_environment_checks_cli_and_api(tmp_path: Path) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    podman = Mock(spec=PodmanCli)
    podman.version.return_value = "5.0.0"
    api = Mock()
    api.ping.return_value = True
    api.version.return_value = {"Components": [{"Name": "Podman Engine"}]}

    with patch("swebench_seq.evaluation.runner.docker.APIClient", return_value=api) as api_cls:
        result = validate_evaluation_environment(
            evaluator_dir,
            docker_host="unix:///tmp/podman.sock",
            podman=podman,
        )

    assert result == (evaluator_dir, "unix:///tmp/podman.sock")
    podman.version.assert_called_once_with()
    api_cls.assert_called_once_with(base_url="unix:///tmp/podman.sock", timeout=5)
    api.ping.assert_called_once_with()
    api.version.assert_called_once_with()
    api.close.assert_called_once_with()


def test_validate_evaluation_environment_rejects_dead_api(tmp_path: Path) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    podman = Mock(spec=PodmanCli)
    podman.version.return_value = "5.0.0"

    with (
        patch(
            "swebench_seq.evaluation.runner.docker.APIClient",
            side_effect=RuntimeError("connection refused"),
        ),
        pytest.raises(RuntimeError, match="Docker-compatible API is unavailable"),
    ):
        validate_evaluation_environment(evaluator_dir, podman=podman)


def test_validate_evaluation_environment_rejects_docker_daemon(tmp_path: Path) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    podman = Mock(spec=PodmanCli)
    podman.version.return_value = "5.0.0"
    api = Mock()
    api.ping.return_value = True
    api.version.return_value = {
        "Platform": {"Name": "Docker Engine - Community"},
        "Components": [{"Name": "Engine"}],
    }

    with (
        patch("swebench_seq.evaluation.runner.docker.APIClient", return_value=api),
        pytest.raises(RuntimeError, match="does not identify itself as Podman"),
    ):
        validate_evaluation_environment(evaluator_dir, podman=podman)


def test_eval_one_uses_configured_evaluator_and_sanitized_env(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    seen: dict = {}

    def fake_run(cmd, *, cwd, env, capture_output, text):
        seen.update(cmd=cmd, cwd=cwd, env=env)
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (output_dir / "eval_results.json").write_text(json.dumps({"iid": True}))
        workspace = output_dir / "iid" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "output.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fake_run)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    assert (
        _eval_one(
            tmp_path / "run",
            "iid",
            "diff",
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
            docker_host="unix:///tmp/podman.sock",
        )
        is True
    )
    assert seen["cmd"][0] == sys.executable
    assert "--block_network" in seen["cmd"]
    assert seen["cwd"] == evaluator_dir
    assert seen["env"]["DOCKER_HOST"] == "unix:///tmp/podman.sock"
    assert "OPENAI_API_KEY" not in seen["env"]


def test_eval_one_rejects_false_result_without_test_output(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")

    def fake_run(cmd, *, cwd, env, capture_output, text):
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (output_dir / "eval_results.json").write_text(json.dumps({"iid": False}))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="DockerException: connection refused",
            stderr="",
        )

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fake_run)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    assert (
        _eval_one(
            tmp_path / "run",
            "iid",
            "diff",
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
            docker_host="unix:///tmp/podman.sock",
        )
        is None
    )
    diagnostic = tmp_path / "run" / "score" / "iid" / "_eval_stdout.log"
    assert diagnostic.read_text() == "DockerException: connection refused"


def test_score_predictions_invokes_public_evaluator(monkeypatch, tmp_path: Path) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"iid": "diff"}))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    seen: dict = {}

    def fake_run(cmd, *, cwd, env, check):
        seen.update(cmd=cmd, cwd=cwd, env=env, check=check)
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (output_dir / "eval_results.json").write_text(json.dumps({"iid": True}))
        workspace = output_dir / "iid" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "output.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fake_run)

    output_dir = tmp_path / "score"
    assert (
        score_predictions(
            predictions_path=predictions,
            output_dir=output_dir,
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
            num_workers=2,
            docker_host="unix:///tmp/podman.sock",
        )
        == 0
    )
    assert json.loads((output_dir / "patches.json").read_text()) == [
        {"instance_id": "iid", "patch": "diff"}
    ]
    assert "--use_local_docker" in seen["cmd"]
    assert "--block_network" in seen["cmd"]
    assert seen["cwd"] == evaluator_dir
    assert json.loads((output_dir / "eval_results.json").read_text()) == {"iid": True}
    assert (output_dir / "iid" / "workspace" / "output.json").is_file()


def test_score_predictions_rejects_result_without_test_output(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"iid": "diff"}))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    def fake_run(cmd, *, cwd, env, check):
        del cwd, env, check
        output_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (output_dir / "eval_results.json").write_text(json.dumps({"iid": False}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fake_run)

    output_dir = tmp_path / "score"
    assert (
        score_predictions(
            predictions_path=predictions,
            output_dir=output_dir,
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
        )
        == 1
    )
    assert json.loads((output_dir / "eval_results.json").read_text()) == {}


def test_score_predictions_marks_empty_patch_failed_without_evaluator(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"iid": " \n"}))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    def fail_run(*args, **kwargs):
        raise AssertionError("empty patches must not invoke the evaluator")

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fail_run)

    output_dir = tmp_path / "score"
    assert (
        score_predictions(
            predictions_path=predictions,
            output_dir=output_dir,
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
        )
        == 0
    )
    assert json.loads((output_dir / "eval_results.json").read_text()) == {"iid": False}


def test_score_predictions_excludes_nonterminal_rollouts(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator_dir = _make_evaluator_dir(tmp_path / "evaluator")
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps({"terminal": "", "infra": ""}))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    def fail_run(*args, **kwargs):
        raise AssertionError("empty patches must not invoke the evaluator")

    monkeypatch.setattr("swebench_seq.evaluation.runner.subprocess.run", fail_run)

    output_dir = tmp_path / "score"
    assert (
        score_predictions(
            predictions_path=predictions,
            output_dir=output_dir,
            evaluator_dir=evaluator_dir,
            dataset_path=dataset,
            gradeable_instance_ids={"terminal"},
        )
        == 0
    )
    assert json.loads((output_dir / "patches.json").read_text()) == [
        {"instance_id": "terminal", "patch": ""}
    ]
    assert json.loads((output_dir / "eval_results.json").read_text()) == {
        "terminal": False
    }


def test_merge_creates_eval_results_json(tmp_path: Path) -> None:
    _merge_eval_result(tmp_path, "instance_a", True)
    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == {"instance_a": True}


def test_merge_preserves_prior_entries(tmp_path: Path) -> None:
    score = tmp_path / "score"
    score.mkdir()
    (score / "eval_results.json").write_text(
        json.dumps({"instance_old": False, "instance_other": True})
    )
    _merge_eval_result(tmp_path, "instance_new", True)
    data = json.loads((score / "eval_results.json").read_text())
    assert data == {
        "instance_old": False,
        "instance_other": True,
        "instance_new": True,
    }


def test_merge_overwrites_same_iid(tmp_path: Path) -> None:
    _merge_eval_result(tmp_path, "iid", False)
    _merge_eval_result(tmp_path, "iid", True)
    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == {"iid": True}


def test_merge_preserves_parallel_writers(tmp_path: Path) -> None:
    entries = [(f"instance_{i}", bool(i % 2)) for i in range(100)]
    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(lambda item: _merge_eval_result(tmp_path, *item), entries))

    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == dict(entries)


def test_merge_tolerates_corrupt_existing_file(tmp_path: Path) -> None:
    score = tmp_path / "score"
    score.mkdir()
    (score / "eval_results.json").write_text("not valid json")
    _merge_eval_result(tmp_path, "iid", True)
    data = json.loads((score / "eval_results.json").read_text())
    assert data == {"iid": True}


def test_cached_eval_result_requires_matching_patch_hash(tmp_path: Path) -> None:
    _merge_eval_result(
        tmp_path,
        "iid",
        True,
        patch_sha256=(
            "a4895eb44afc336fecbba6e520cd67e178dace0276655d102fceffa8e5f70570"
        ),
    )

    assert get_cached_eval_result(tmp_path, "iid", "patch") is True
    assert get_cached_eval_result(tmp_path, "iid", "different patch") is None


def test_port_collision_detection(tmp_path: Path) -> None:
    task_dir = tmp_path / "iid"
    task_dir.mkdir()
    (task_dir / "_stdout.log").write_text(
        "Error: listen EADDRINUSE: address already in use 0.0.0.0:4567"
    )

    assert _has_port_collision(tmp_path, "iid") is True


# ---- score_one ----


def test_score_one_empty_patch_short_circuits_to_false(tmp_path: Path) -> None:
    """Compaction-thrash tasks produce 0-byte patches. Eval would record
    FAIL anyway — skip the podman cycle and store False directly."""
    out = score_one(tmp_path, "iid", patch="", tag_lookup={"iid": "tag"})
    assert out is False
    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == {"iid": False}
    assert get_cached_eval_result(tmp_path, "iid", "") is False


def test_score_one_whitespace_only_patch_short_circuits(tmp_path: Path) -> None:
    out = score_one(tmp_path, "iid", patch="   \n\n  \n", tag_lookup={"iid": "tag"})
    assert out is False


def test_score_one_no_tag_returns_none(tmp_path: Path) -> None:
    """Iid missing from the dataset → can't score → None."""
    out = score_one(tmp_path, "unknown_iid", patch="diff --git a b", tag_lookup={})
    assert out is None
    # No eval_results.json written on infra-fail.
    assert not (tmp_path / "score" / "eval_results.json").exists()


def test_score_one_pull_failure_returns_none(tmp_path: Path) -> None:
    with patch("swebench_seq.evaluation.runner._ensure_image", return_value=False):
        out = score_one(tmp_path, "iid", patch="real diff", tag_lookup={"iid": "tag"})
    assert out is None
    assert not (tmp_path / "score" / "eval_results.json").exists()


def test_ensure_image_pulls_and_aliases_custom_registry() -> None:
    podman = Mock(spec=PodmanCli)
    podman.image_exists.side_effect = [False, False]

    ok = _ensure_image(
        "tag",
        image_prefix="registry.example.test/swebench-pro",
        podman=podman,
    )

    assert ok is True
    podman.pull.assert_called_once_with(
        "registry.example.test/swebench-pro:tag"
    )
    podman.tag.assert_called_once_with(
        "registry.example.test/swebench-pro:tag",
        "docker.io/jefzda/sweap-images:tag",
    )


def test_score_one_happy_path_pass(tmp_path: Path) -> None:
    with patch("swebench_seq.evaluation.runner._ensure_image", return_value=True), \
         patch("swebench_seq.evaluation.runner._eval_one", return_value=True):
        out = score_one(tmp_path, "iid", patch="real diff", tag_lookup={"iid": "tag"})
    assert out is True
    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == {"iid": True}
    assert get_cached_eval_result(tmp_path, "iid", "real diff") is True


def test_score_one_happy_path_fail(tmp_path: Path) -> None:
    with patch("swebench_seq.evaluation.runner._ensure_image", return_value=True), \
         patch("swebench_seq.evaluation.runner._eval_one", return_value=False):
        out = score_one(tmp_path, "iid", patch="real diff", tag_lookup={"iid": "tag"})
    assert out is False
    data = json.loads((tmp_path / "score" / "eval_results.json").read_text())
    assert data == {"iid": False}


def test_score_one_eval_infra_fail_returns_none_no_merge(tmp_path: Path) -> None:
    """_eval_one returns None on infra failure (evaluator crashed). We
    must NOT write that into eval_results.json — leave the iid absent so
    a future scoring pass can retry without false-fail contamination."""
    with patch("swebench_seq.evaluation.runner._ensure_image", return_value=True), \
         patch("swebench_seq.evaluation.runner._eval_one", return_value=None):
        out = score_one(tmp_path, "iid", patch="real diff", tag_lookup={"iid": "tag"})
    assert out is None
    # No eval_results.json written.
    assert not (tmp_path / "score" / "eval_results.json").exists()


def test_score_one_leaves_cached_image_when_eval_raises(tmp_path: Path) -> None:
    podman = Mock(spec=PodmanCli)
    with patch("swebench_seq.evaluation.runner._ensure_image", return_value=True), \
         patch("swebench_seq.evaluation.runner._eval_one", side_effect=RuntimeError("boom")):
        try:
            score_one(
                tmp_path,
                "iid",
                patch="diff",
                tag_lookup={"iid": "tag"},
                podman=podman,
            )
        except RuntimeError:
            pass
    # The evaluator owns no image deletion lifecycle. Images pulled or tagged
    # for a run remain in Podman's cache for reuse and user-managed cleanup.
    assert not hasattr(podman, "rmi")
