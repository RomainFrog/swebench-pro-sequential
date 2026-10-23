"""Tests for dataset row → Task conversion, filtering, grouping, and ordering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swebench_seq.tasks import (
    DEFAULT_REPO_TASK_CAP,
    _load_commit_dates,
    _load_commit_topo_ranks,
    _row_to_task,
    apply_frozen_order,
    cap_task_universe_plans,
    filter_by_repo,
    group_into_universes,
    load_frozen_sequential_order,
    plan_task_universes,
    sort_tasks_chronologically,
    validate_frozen_sequential_order,
)
from swebench_seq.types import Task


def _row(instance_id: str, repo: str, **overrides: object) -> dict:
    base = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "deadbeef",
        "problem_statement": "Some bug.",
        "dockerhub_tag": instance_id,
    }
    base.update(overrides)
    return base


def test_row_to_task_scalars_pass_through() -> None:
    t = _row_to_task(_row("q-1", "qutebrowser"))
    assert t.instance_id == "q-1"
    assert t.repo == "qutebrowser"


def test_filter_by_repo_preserves_order() -> None:
    tasks = [
        _row_to_task(_row("a-1", "ansible")),
        _row_to_task(_row("q-1", "qutebrowser")),
        _row_to_task(_row("a-2", "ansible")),
    ]
    filtered = filter_by_repo(tasks, "ansible")
    assert [t.instance_id for t in filtered] == ["a-1", "a-2"]


def test_group_into_universes_buckets_by_repo() -> None:
    tasks = [
        _row_to_task(_row("a-1", "ansible")),
        _row_to_task(_row("q-1", "qutebrowser")),
        _row_to_task(_row("a-2", "ansible")),
    ]
    universes = {u.repo: u for u in group_into_universes(tasks)}
    assert set(universes) == {"ansible", "qutebrowser"}
    assert [t.instance_id for t in universes["ansible"].tasks] == ["a-1", "a-2"]


def test_task_dataclass_round_trips_through_row() -> None:
    row = _row("q-1", "qutebrowser")
    t = _row_to_task(row)
    assert isinstance(t, Task)
    assert t.problem_statement == "Some bug."


# ───────────────────────── commit-date enrichment ─────────────────────────


def _task(iid: str, base: str, **overrides: object) -> Task:
    return _row_to_task(
        _row(iid, "qutebrowser", base_commit=base),
        overrides.get("dates"),  # type: ignore[arg-type]
    )


def test_row_to_task_attaches_commit_date_when_present_in_cache() -> None:
    dates = {"sha-A": 1_700_000_000, "sha-B": 1_700_000_100}
    t = _row_to_task(_row("q-1", "qutebrowser", base_commit="sha-A"), dates)
    assert t.commit_date == 1_700_000_000


def test_row_to_task_commit_date_none_when_base_absent_from_cache() -> None:
    t = _row_to_task(_row("q-1", "qutebrowser", base_commit="sha-X"), {})
    assert t.commit_date is None


def test_sort_tasks_chronologically_orders_by_commit_date() -> None:
    tasks = [
        _row_to_task(_row("c", "r", base_commit="sha-c"), {"sha-c": 300}),
        _row_to_task(_row("a", "r", base_commit="sha-a"), {"sha-a": 100}),
        _row_to_task(_row("b", "r", base_commit="sha-b"), {"sha-b": 200}),
    ]
    ordered = sort_tasks_chronologically(tasks)
    assert [t.instance_id for t in ordered] == ["a", "b", "c"]


def test_sort_tasks_chronologically_breaks_ties_by_sha() -> None:
    # Two commits at the exact same UNIX second; SHA lex breaks the tie.
    dates = {"aaa": 100, "zzz": 100, "mmm": 200}
    tasks = [
        _row_to_task(_row("Z", "r", base_commit="zzz"), dates),
        _row_to_task(_row("A", "r", base_commit="aaa"), dates),
        _row_to_task(_row("M", "r", base_commit="mmm"), dates),
    ]
    ordered = sort_tasks_chronologically(tasks)
    # Among the two tied (ts=100) commits, 'aaa' < 'zzz' lex; mmm is later.
    assert [t.instance_id for t in ordered] == ["A", "Z", "M"]


def test_sort_tasks_chronologically_undated_tasks_go_last_stable() -> None:
    dates = {"sha-a": 100, "sha-b": 50}
    tasks = [
        _row_to_task(_row("u1", "r", base_commit="ux1"), dates),  # undated
        _row_to_task(_row("a", "r", base_commit="sha-a"), dates),
        _row_to_task(_row("u2", "r", base_commit="ux2"), dates),  # undated
        _row_to_task(_row("b", "r", base_commit="sha-b"), dates),
    ]
    ordered = sort_tasks_chronologically(tasks)
    # Dated: b (50) before a (100). Undated: input order u1, u2.
    assert [t.instance_id for t in ordered] == ["b", "a", "u1", "u2"]


def test_sort_tasks_chronologically_empty_input() -> None:
    assert sort_tasks_chronologically([]) == []


def test_load_commit_dates_reads_json_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "dates.json"
    cache_path.write_text(json.dumps({"sha-1": 100, "sha-2": 200}) + "\n")
    dates = _load_commit_dates(cache_path)
    assert dates == {"sha-1": 100, "sha-2": 200}


def test_load_commit_dates_missing_file_returns_empty() -> None:
    dates = _load_commit_dates(Path("/nonexistent/path/dates.json"))
    assert dates == {}


def test_load_commit_dates_invalid_json_returns_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "broken.json"
    cache_path.write_text("not json at all {{{")
    dates = _load_commit_dates(cache_path)
    assert dates == {}


def test_load_commit_dates_drops_non_numeric_values(tmp_path: Path) -> None:
    cache_path = tmp_path / "dates.json"
    cache_path.write_text(json.dumps({"sha-1": 100, "sha-2": "garbage"}) + "\n")
    dates = _load_commit_dates(cache_path)
    assert dates == {"sha-1": 100}


# ────────────────────── topological-rank enrichment ──────────────────────


def test_row_to_task_attaches_commit_topo_rank_when_present() -> None:
    topo = {"sha-A": 42, "sha-B": 99}
    t = _row_to_task(
        _row("q-1", "qutebrowser", base_commit="sha-A"),
        {"sha-A": 1_700_000_000},
        topo,
    )
    assert t.commit_date == 1_700_000_000
    assert t.commit_topo_rank == 42


def test_row_to_task_commit_topo_rank_none_when_base_absent() -> None:
    t = _row_to_task(_row("q-1", "qutebrowser", base_commit="sha-X"), {}, {})
    assert t.commit_topo_rank is None


def test_row_to_task_topo_independent_of_date() -> None:
    # commit has a topo rank but no date — both fields are independently
    # populated, so commit_topo_rank should still be set.
    t = _row_to_task(
        _row("q-1", "qutebrowser", base_commit="sha-A"),
        dates={},
        topo_ranks={"sha-A": 5},
    )
    assert t.commit_date is None
    assert t.commit_topo_rank == 5


def test_sort_topo_rank_breaks_tie_when_dates_equal() -> None:
    # Two commits at the exact same UNIX second.
    # Without topo rank, SHA lex would put 'zzz' AFTER 'aaa'. But if topo
    # says zzz is an ancestor of aaa (lower rank), zzz must go first.
    dates = {"aaa": 100, "zzz": 100, "mmm": 200}
    topo = {"aaa": 10, "zzz": 5, "mmm": 50}  # zzz earlier than aaa
    tasks = [
        _row_to_task(_row("A", "r", base_commit="aaa"), dates, topo),
        _row_to_task(_row("Z", "r", base_commit="zzz"), dates, topo),
        _row_to_task(_row("M", "r", base_commit="mmm"), dates, topo),
    ]
    ordered = sort_tasks_chronologically(tasks)
    assert [t.instance_id for t in ordered] == ["Z", "A", "M"]


def test_sort_falls_back_to_sha_when_topo_rank_missing() -> None:
    # Same dates, neither commit has a topo rank — should fall back to SHA lex.
    dates = {"zzz": 100, "aaa": 100}
    tasks = [
        _row_to_task(_row("Z", "r", base_commit="zzz"), dates, {}),
        _row_to_task(_row("A", "r", base_commit="aaa"), dates, {}),
    ]
    ordered = sort_tasks_chronologically(tasks)
    assert [t.instance_id for t in ordered] == ["A", "Z"]  # SHA lex


def test_sort_topo_rank_used_only_within_same_date() -> None:
    # Date ordering must NOT be overridden by topo rank.
    # Even if older-rank commit has a much later date, date wins.
    dates = {"early-but-late-rank": 100, "late-but-early-rank": 200}
    topo = {"early-but-late-rank": 999, "late-but-early-rank": 1}
    tasks = [
        _row_to_task(
            _row("LATE_DATE_EARLY_RANK", "r", base_commit="late-but-early-rank"),
            dates,
            topo,
        ),
        _row_to_task(
            _row("EARLY_DATE_LATE_RANK", "r", base_commit="early-but-late-rank"),
            dates,
            topo,
        ),
    ]
    ordered = sort_tasks_chronologically(tasks)
    assert [t.instance_id for t in ordered] == [
        "EARLY_DATE_LATE_RANK",
        "LATE_DATE_EARLY_RANK",
    ]


def test_sort_topo_partial_coverage_known_rank_wins() -> None:
    # When dates tie and ONE commit has a topo rank but the other doesn't,
    # the one with a known rank sorts first (sentinel = n+1).
    dates = {"with-rank": 100, "no-rank": 100}
    topo = {"with-rank": 1}
    tasks = [
        _row_to_task(_row("NO", "r", base_commit="no-rank"), dates, topo),
        _row_to_task(_row("WITH", "r", base_commit="with-rank"), dates, topo),
    ]
    ordered = sort_tasks_chronologically(tasks)
    assert [t.instance_id for t in ordered] == ["WITH", "NO"]


def test_load_commit_topo_ranks_reads_json_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "topo.json"
    cache_path.write_text(json.dumps({"sha-1": 10, "sha-2": 20}) + "\n")
    ranks = _load_commit_topo_ranks(cache_path)
    assert ranks == {"sha-1": 10, "sha-2": 20}


def test_load_commit_topo_ranks_missing_file_returns_empty() -> None:
    ranks = _load_commit_topo_ranks(Path("/nonexistent/topo.json"))
    assert ranks == {}


def test_load_commit_topo_ranks_invalid_json_returns_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "broken.json"
    cache_path.write_text("not json {{")
    ranks = _load_commit_topo_ranks(cache_path)
    assert ranks == {}


def test_load_commit_topo_ranks_drops_non_numeric_values(tmp_path: Path) -> None:
    cache_path = tmp_path / "topo.json"
    cache_path.write_text(json.dumps({"sha-1": 5, "sha-2": "nope"}) + "\n")
    ranks = _load_commit_topo_ranks(cache_path)
    assert ranks == {"sha-1": 5}


# ─────────────────────── frozen sequential order ───────────────────────


def test_load_frozen_order_reads_canonical_file(tmp_path: Path) -> None:
    cache_path = tmp_path / "seq.json"
    cache_path.write_text(
        json.dumps(
            {
                "qutebrowser/qutebrowser": ["q-c", "q-a", "q-b"],
                "tutao/tutanota": ["t-1", "t-2"],
            }
        )
        + "\n"
    )
    order = load_frozen_sequential_order(cache_path)
    assert order == {
        "qutebrowser/qutebrowser": ["q-c", "q-a", "q-b"],
        "tutao/tutanota": ["t-1", "t-2"],
    }


def test_load_frozen_order_missing_file_returns_empty() -> None:
    assert load_frozen_sequential_order(Path("/nonexistent/seq.json")) == {}


def test_load_frozen_order_invalid_json_returns_empty(tmp_path: Path) -> None:
    cache_path = tmp_path / "broken.json"
    cache_path.write_text("[[[")
    assert load_frozen_sequential_order(cache_path) == {}


def test_load_frozen_order_skips_non_list_values(tmp_path: Path) -> None:
    cache_path = tmp_path / "mixed.json"
    cache_path.write_text(
        json.dumps({"good": ["a", "b"], "bad-not-list": "abc", "bad-not-str": [1, 2]})
    )
    order = load_frozen_sequential_order(cache_path)
    assert order == {"good": ["a", "b"]}


def test_validate_frozen_order_accepts_exact_dataset_coverage() -> None:
    tasks = [
        _row_to_task(_row("a", "repo/r")),
        _row_to_task(_row("b", "repo/r")),
        _row_to_task(_row("c", "repo/other")),
    ]

    validate_frozen_sequential_order(
        tasks,
        {"repo/r": ["b", "a"], "repo/other": ["c"]},
    )


def test_validate_frozen_order_rejects_missing_unknown_and_duplicate_ids() -> None:
    tasks = [
        _row_to_task(_row("a", "repo/r")),
        _row_to_task(_row("b", "repo/r")),
    ]

    with pytest.raises(
        ValueError,
        match=r"1 duplicate id\(s\).*1 missing id\(s\).*1 unknown id\(s\)",
    ):
        validate_frozen_sequential_order(
            tasks,
            {"repo/r": ["a", "a", "unknown"]},
        )


def test_apply_frozen_order_reorders_tasks_to_match() -> None:
    tasks = [
        _row_to_task(_row("c", "r")),
        _row_to_task(_row("a", "r")),
        _row_to_task(_row("b", "r")),
    ]
    ordered = apply_frozen_order(tasks, ["a", "b", "c"])
    assert [t.instance_id for t in ordered] == ["a", "b", "c"]


def test_apply_frozen_order_appends_unlisted_tasks_in_input_order() -> None:
    tasks = [
        _row_to_task(_row("z", "r")),
        _row_to_task(_row("a", "r")),
        _row_to_task(_row("new", "r")),  # not in frozen list
        _row_to_task(_row("b", "r")),
    ]
    ordered = apply_frozen_order(tasks, ["a", "b", "z"])
    # Listed tasks first in frozen order, then 'new' at the end
    assert [t.instance_id for t in ordered] == ["a", "b", "z", "new"]


def test_apply_frozen_order_drops_ids_not_in_tasks() -> None:
    tasks = [_row_to_task(_row("a", "r")), _row_to_task(_row("b", "r"))]
    ordered = apply_frozen_order(tasks, ["a", "stale-deleted", "b"])
    assert [t.instance_id for t in ordered] == ["a", "b"]


def test_apply_frozen_order_deduplicates_listed_ids() -> None:
    tasks = [_row_to_task(_row("a", "r")), _row_to_task(_row("b", "r"))]
    ordered = apply_frozen_order(tasks, ["a", "a", "b"])
    assert [t.instance_id for t in ordered] == ["a", "b"]


def test_apply_frozen_order_empty_order_returns_input_order() -> None:
    tasks = [_row_to_task(_row("c", "r")), _row_to_task(_row("a", "r"))]
    ordered = apply_frozen_order(tasks, [])
    assert [t.instance_id for t in ordered] == ["c", "a"]


def test_apply_frozen_order_empty_tasks() -> None:
    assert apply_frozen_order([], ["a", "b"]) == []


def test_plan_task_universes_defaults_to_repo40_after_frozen_order() -> None:
    repo_tasks = [_row_to_task(_row(f"r-{i}", "repo/r")) for i in range(45)]
    other_tasks = [_row_to_task(_row(f"o-{i}", "repo/other")) for i in range(3)]
    frozen = {"repo/r": [f"r-{i}" for i in reversed(range(45))]}

    plans = {
        plan.repo: plan
        for plan in plan_task_universes(repo_tasks + other_tasks, frozen_order=frozen)
    }

    assert len(plans["repo/r"]) == DEFAULT_REPO_TASK_CAP
    assert plans["repo/r"].available_task_count == 45
    assert plans["repo/r"].order_source == "frozen"
    assert [t.instance_id for t in plans["repo/r"].tasks[:3]] == [
        "r-44",
        "r-43",
        "r-42",
    ]
    assert [t.instance_id for t in plans["repo/other"].tasks] == ["o-0", "o-1", "o-2"]


def test_plan_task_universes_full_dataset_disables_repo_cap() -> None:
    tasks = [_row_to_task(_row(f"r-{i}", "repo/r")) for i in range(45)]

    plans = plan_task_universes(tasks, cap_per_repo=None)

    assert len(plans) == 1
    assert len(plans[0]) == 45


def test_cap_task_universe_plans_is_additional_per_repo_limit() -> None:
    tasks = [_row_to_task(_row(f"r-{i}", "repo/r")) for i in range(5)]
    plans = plan_task_universes(tasks, cap_per_repo=None)

    limited = cap_task_universe_plans(plans, 2)

    assert len(limited) == 1
    assert [t.instance_id for t in limited[0].tasks] == ["r-0", "r-1"]
