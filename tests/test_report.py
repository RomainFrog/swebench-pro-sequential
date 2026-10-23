"""Tests for reports over the runner's current artifact layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swebench_seq.constants import BENCHMARK_VERSION, SEQUENTIAL_ORDER_PATH
from swebench_seq.report import load_report, validate_run

QUTE_IDS = [
    "instance_qutebrowser__qutebrowser-aaa",
    "instance_qutebrowser__qutebrowser-bbb",
    "instance_qutebrowser__qutebrowser-ccc",
]


def _write_task_artifacts(path: Path, instance_id: str, *, calls: int, patch: str) -> None:
    path.mkdir(parents=True)
    (path / "patch.diff").write_text(patch)
    (path / f"{instance_id}.mini.traj.json").write_text(
        json.dumps({"info": {"model_stats": {"api_calls": calls}}, "messages": []})
    )
    (path / "task_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "instance_id": instance_id,
                "rollout_status": "Submitted",
                "model_calls": calls,
                "wall_seconds": calls * 1.5,
                "usage": {
                    "input_tokens": calls * 100,
                    "output_tokens": calls * 10,
                    "total_tokens": calls * 110,
                },
            }
        )
    )


def _make_independent_run(root: Path) -> None:
    for iid, calls, patch in (
        (QUTE_IDS[0], 2, "diff a"),
        (QUTE_IDS[1], 3, "diff b"),
        ("instance_ansible__ansible-ccc", 4, "diff c\nmore\n"),
    ):
        _write_task_artifacts(root / iid, iid, calls=calls, patch=patch)


def _make_sequential_run(root: Path) -> Path:
    universe = root / "universes" / "qutebrowser__qutebrowser"
    universe.mkdir(parents=True)
    (universe / "_agent_home").mkdir()
    (universe / "_mini_state").mkdir()
    (universe / "score").mkdir()
    for position, iid in enumerate(QUTE_IDS):
        task_dir = universe / iid
        _write_task_artifacts(task_dir, iid, calls=position + 1, patch="x" * 100)
        snapshot = task_dir / "_agent_home_snapshot"
        snapshot.mkdir()
        (snapshot / "MEMORY.md").write_text("x" * (position + 1))
        (snapshot / "notes.txt").write_text("ok")
    (universe / QUTE_IDS[0] / "_agent_home_snapshot" / "bin").mkdir()
    (universe / QUTE_IDS[0] / "_agent_home_snapshot" / "bin" / "ls").symlink_to(
        "/usr/bin/ls"
    )
    return universe


def test_load_independent_mode(tmp_path: Path) -> None:
    _make_independent_run(tmp_path)
    report = load_report(tmp_path)
    assert report.mode == "independent"
    assert len(report.tasks) == 3
    assert set(report.by_repo()) == {"qutebrowser/qutebrowser", "ansible/ansible"}


def test_load_sequential_mode_ignores_runtime_directories(tmp_path: Path) -> None:
    _make_sequential_run(tmp_path)
    tasks = load_report(tmp_path).by_repo()["qutebrowser/qutebrowser"]
    assert [task.instance_id for task in tasks] == QUTE_IDS
    assert all(task.has_agent_home for task in tasks)


def test_sequential_uses_each_tasks_agent_home_snapshot(tmp_path: Path) -> None:
    _make_sequential_run(tmp_path)
    tasks = load_report(tmp_path).by_repo()["qutebrowser/qutebrowser"]
    assert [task.memory_md_bytes for task in tasks] == [1, 2, 3]
    assert tasks[0].agent_home_bytes is not None
    assert tasks[0].agent_home_bytes < 10_000  # the /usr/bin/ls symlink is not followed


def test_task_report_reads_patch_and_model_call_counts(tmp_path: Path) -> None:
    _make_sequential_run(tmp_path)
    tasks = load_report(tmp_path).by_repo()["qutebrowser/qutebrowser"]
    assert [task.patch_bytes for task in tasks] == [100, 100, 100]
    assert [task.model_calls for task in tasks] == [1, 2, 3]
    assert [task.rollout_status for task in tasks] == ["Submitted"] * 3
    assert [task.total_tokens for task in tasks] == [110, 220, 330]


def test_report_summary_separates_scores_from_incomplete_tasks(tmp_path: Path) -> None:
    universe = _make_sequential_run(tmp_path / "run_0")
    (tmp_path / "task_manifest.json").write_text(
        json.dumps(
            {
                "repos": [
                    {"repo": "qutebrowser/qutebrowser", "instance_ids": QUTE_IDS}
                ]
            }
        )
    )
    (universe / "score" / "eval_results.json").write_text(
        json.dumps({QUTE_IDS[0]: True, QUTE_IDS[1]: False})
    )

    summary = load_report(tmp_path / "run_0").summary()

    assert summary["selected"] == 3
    assert summary["completed_rollouts"] == 3
    assert summary["graded"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["infrastructure_or_incomplete"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["total_tokens"] == 660


def test_report_excludes_nonterminal_rollout_even_if_score_exists(tmp_path: Path) -> None:
    universe = _make_sequential_run(tmp_path / "run_0")
    failed_task = universe / QUTE_IDS[0]
    result = json.loads((failed_task / "task_result.json").read_text())
    result["rollout_status"] = "AuthenticationError"
    (failed_task / "task_result.json").write_text(json.dumps(result))
    (universe / "score" / "eval_results.json").write_text(
        json.dumps({QUTE_IDS[0]: False, QUTE_IDS[1]: True})
    )

    report = load_report(tmp_path / "run_0")
    tasks = report.by_repo()["qutebrowser/qutebrowser"]

    assert [task.passed for task in tasks] == [None, True, None]
    assert report.summary()["graded"] == 1
    assert report.summary()["infrastructure_or_incomplete"] == 2


def test_report_summary_includes_sequential_compaction_usage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    universe = _make_sequential_run(run_dir)
    (tmp_path / "task_manifest.json").write_text(
        json.dumps(
            {
                "repos": [
                    {"repo": "qutebrowser/qutebrowser", "instance_ids": QUTE_IDS}
                ]
            }
        )
    )
    usage = [
        {
            "phase": "agent",
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        },
        {
            "phase": "compaction",
            "input_tokens": 80,
            "output_tokens": 10,
            "total_tokens": 90,
        },
    ]
    (universe / "_mini_state" / "usage.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in usage)
    )

    summary = load_report(run_dir).summary()

    assert summary["compaction_calls"] == 1
    assert summary["input_tokens"] == 180
    assert summary["output_tokens"] == 30
    assert summary["total_tokens"] == 210


def test_sequential_reads_inline_scores_and_manifest_order(tmp_path: Path) -> None:
    universe = _make_sequential_run(tmp_path / "run_0")
    ordered = [QUTE_IDS[2], QUTE_IDS[0], QUTE_IDS[1]]
    (tmp_path / "task_manifest.json").write_text(
        json.dumps({"repos": [{"repo": "qutebrowser/qutebrowser", "instance_ids": ordered}]})
    )
    (universe / "score" / "eval_results.json").write_text(
        json.dumps({ordered[0]: True, ordered[1]: False, ordered[2]: True})
    )

    report = load_report(tmp_path / "run_0")
    tasks = report.by_repo()["qutebrowser/qutebrowser"]
    assert [task.instance_id for task in tasks] == ordered
    assert [task.passed for task in tasks] == [True, False, True]
    rates = report.pass_rate_by_position("qutebrowser/qutebrowser")
    assert rates[:2] == [(0, 1.0), (1, 0.5)]
    assert pytest.approx(rates[2][1], abs=1e-9) == 2 / 3


def test_sequential_bulk_scores_override_inline_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    universe = _make_sequential_run(run_dir)
    (universe / "score" / "eval_results.json").write_text(
        json.dumps({QUTE_IDS[0]: False})
    )
    (run_dir / "score").mkdir()
    (run_dir / "score" / "eval_results.json").write_text(
        json.dumps({QUTE_IDS[0]: True, QUTE_IDS[1]: False})
    )

    tasks = load_report(run_dir).by_repo()["qutebrowser/qutebrowser"]

    assert [task.passed for task in tasks] == [True, False, None]


def test_pass_rate_returns_none_when_scoring_data_absent(tmp_path: Path) -> None:
    _make_sequential_run(tmp_path)
    rates = load_report(tmp_path).pass_rate_by_position("qutebrowser/qutebrowser")
    assert all(rate is None for _, rate in rates)


def test_independent_all_repos_layout_uses_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    repo_dir = run_dir / "qutebrowser__qutebrowser"
    _write_task_artifacts(repo_dir / QUTE_IDS[0], QUTE_IDS[0], calls=7, patch="patch")
    (repo_dir / "score").mkdir()
    (repo_dir / "score" / "eval_results.json").write_text(json.dumps({QUTE_IDS[0]: True}))
    (tmp_path / "task_manifest.json").write_text(
        json.dumps(
            {"repos": [{"repo": "qutebrowser/qutebrowser", "instance_ids": [QUTE_IDS[0]]}]}
        )
    )

    task = load_report(run_dir).tasks[0]
    assert task.repo == "qutebrowser/qutebrowser"
    assert task.passed is True
    assert task.model_calls == 7


def test_independent_all_repos_reads_bulk_scores_from_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    repo_dir = run_dir / "qutebrowser__qutebrowser"
    _write_task_artifacts(repo_dir / QUTE_IDS[0], QUTE_IDS[0], calls=1, patch="patch")
    (run_dir / "score").mkdir()
    (run_dir / "score" / "eval_results.json").write_text(
        json.dumps({QUTE_IDS[0]: True})
    )
    (tmp_path / "task_manifest.json").write_text(
        json.dumps(
            {"repos": [{"repo": "qutebrowser/qutebrowser", "instance_ids": [QUTE_IDS[0]]}]}
        )
    )

    assert load_report(run_dir).tasks[0].passed is True


def test_load_report_raises_when_run_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "nope")


def test_validate_run_reports_missing_provenance(tmp_path: Path) -> None:
    _make_sequential_run(tmp_path / "run_0")

    errors = validate_run(tmp_path / "run_0")

    assert "missing or invalid run_manifest.json" in errors
    assert "missing or invalid task_manifest.json" in errors
    assert "missing or invalid image_manifest.json" in errors


def test_validate_run_accepts_complete_failed_scores(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    universe = _make_sequential_run(run_dir)
    order_hash = hashlib.sha256(SEQUENTIAL_ORDER_PATH.read_bytes()).hexdigest()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "swebench_seq_version": "0.1.0",
                "benchmark": {
                    "version": BENCHMARK_VERSION,
                    "sequential_order_sha256": order_hash,
                },
                "runtime": {"packages": {"swebench-seq": "0.1.0"}},
            }
        )
    )
    (tmp_path / "task_manifest.json").write_text(
        json.dumps(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "repos": [
                    {"repo": "qutebrowser/qutebrowser", "instance_ids": QUTE_IDS}
                ]
            }
        )
    )
    (universe / "score" / "eval_results.json").write_text(
        json.dumps({instance_id: False for instance_id in QUTE_IDS})
    )
    (run_dir / "image_manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"instance_id": instance_id, "image": {"id": f"sha256:{index}"}}
                    for index, instance_id in enumerate(QUTE_IDS)
                ]
            }
        )
    )
    (run_dir / "predictions.json").write_text(
        json.dumps({instance_id: "patch" for instance_id in QUTE_IDS})
    )

    assert validate_run(run_dir) == []


def test_validate_run_rejects_tracked_source_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True)
    order_hash = hashlib.sha256(SEQUENTIAL_ORDER_PATH.read_bytes()).hexdigest()
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "swebench_seq_version": "0.1.0",
                "benchmark": {
                    "version": BENCHMARK_VERSION,
                    "sequential_order_sha256": order_hash,
                },
                "runtime": {
                    "source_dirty": True,
                    "packages": {"swebench-seq": "0.1.0"},
                },
            }
        )
    )

    assert "run used a source checkout with tracked modifications" in validate_run(
        run_dir
    )
