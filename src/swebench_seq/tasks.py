"""SWE-bench Pro dataset loading + grouping into universes (= per-repo task batches)."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import COMMIT_DATES_PATH, COMMIT_TOPO_RANKS_PATH, SEQUENTIAL_ORDER_PATH
from .evaluation.dataset import load_swebench_pro_rows
from .types import Task, Universe

DEFAULT_REPO_TASK_CAP = 40


@dataclass(frozen=True)
class TaskUniversePlan:
    """Ordered task selection for one repo universe."""

    repo: str
    tasks: tuple[Task, ...]
    available_task_count: int
    order_source: str

    def __len__(self) -> int:
        return len(self.tasks)


def load_frozen_sequential_order(path: Path | None = None) -> dict[str, list[str]]:
    """Read the canonical per-repo task ordering from disk.

    Format: `{"<repo_name>": ["instance_id_1", "instance_id_2", ...], ...}`

    The committed file is the reproducible source of truth. Returns `{}` on a
    missing or invalid file so callers can fall back to dynamic chronological
    ordering.
    """
    target = path or SEQUENTIAL_ORDER_PATH
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[str(k)] = list(v)
    return out


def validate_frozen_sequential_order(
    tasks: Iterable[Task],
    frozen_order: Mapping[str, Sequence[str]],
) -> None:
    """Require exact frozen-order coverage for the pinned official dataset."""

    by_repo: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        by_repo[task.repo].add(task.instance_id)

    errors: list[str] = []
    missing_repos = sorted(set(by_repo) - set(frozen_order))
    extra_repos = sorted(set(frozen_order) - set(by_repo))
    if missing_repos:
        errors.append(f"missing repositories: {', '.join(missing_repos)}")
    if extra_repos:
        errors.append(f"unknown repositories: {', '.join(extra_repos)}")

    for repo, expected_ids in sorted(by_repo.items()):
        ordered = list(frozen_order.get(repo, ()))
        ordered_ids = set(ordered)
        duplicates = len(ordered) - len(ordered_ids)
        missing = expected_ids - ordered_ids
        extra = ordered_ids - expected_ids
        details: list[str] = []
        if duplicates:
            details.append(f"{duplicates} duplicate id(s)")
        if missing:
            details.append(f"{len(missing)} missing id(s)")
        if extra:
            details.append(f"{len(extra)} unknown id(s)")
        if details:
            errors.append(f"{repo}: {', '.join(details)}")

    if errors:
        raise ValueError(
            "canonical sequential order does not match the pinned dataset: "
            + "; ".join(errors)
        )


def apply_frozen_order(tasks: Iterable[Task], order: Sequence[str]) -> list[Task]:
    """Reorder a single repo's tasks according to a frozen instance_id list.

    Tasks present in `tasks` but missing from `order` are appended in their
    original input order — keeps things robust if the cache is stale (e.g.
    dataset added new instances since the freeze).

    Tasks present in `order` but missing from `tasks` are silently dropped —
    we only run what's in `tasks` (the caller may have already filtered).
    """
    by_id = {t.instance_id: t for t in tasks}
    seen: set[str] = set()
    ordered: list[Task] = []
    for iid in order:
        if iid in by_id and iid not in seen:
            ordered.append(by_id[iid])
            seen.add(iid)
    for t in tasks:
        if t.instance_id not in seen:
            ordered.append(t)
    return ordered


def plan_task_universes(
    tasks: Iterable[Task],
    *,
    repo: str | None = None,
    frozen_order: Mapping[str, Sequence[str]] | None = None,
    cap_per_repo: int | None = DEFAULT_REPO_TASK_CAP,
) -> list[TaskUniversePlan]:
    """Build the benchmark task selection, ordered and capped per repo.

    The raw dataset loader intentionally remains uncapped. This helper is the
    single place where benchmark suite selection happens.
    """
    if cap_per_repo is not None and cap_per_repo < 0:
        raise ValueError(f"cap_per_repo must be >= 0, got {cap_per_repo}")

    selected_tasks = filter_by_repo(tasks, repo) if repo is not None else list(tasks)
    frozen_order = frozen_order or {}
    plans: list[TaskUniversePlan] = []
    for universe in group_into_universes(selected_tasks):
        if universe.repo in frozen_order:
            ordered = apply_frozen_order(universe.tasks, frozen_order[universe.repo])
            order_source = "frozen"
        else:
            ordered = sort_tasks_chronologically(universe.tasks)
            order_source = "dynamic-sort"
        if cap_per_repo is not None:
            ordered = ordered[:cap_per_repo]
        if ordered:
            plans.append(
                TaskUniversePlan(
                    repo=universe.repo,
                    tasks=tuple(ordered),
                    available_task_count=len(universe),
                    order_source=order_source,
                )
            )
    return plans


def cap_task_universe_plans(
    plans: Iterable[TaskUniversePlan],
    cap_per_repo: int,
) -> list[TaskUniversePlan]:
    """Apply an additional per-repo debug cap to already-planned universes."""
    if cap_per_repo < 0:
        raise ValueError(f"cap_per_repo must be >= 0, got {cap_per_repo}")
    capped: list[TaskUniversePlan] = []
    for plan in plans:
        tasks = plan.tasks[:cap_per_repo]
        if tasks:
            capped.append(
                TaskUniversePlan(
                    repo=plan.repo,
                    tasks=tasks,
                    available_task_count=plan.available_task_count,
                    order_source=plan.order_source,
                )
            )
    return capped


def _load_commit_dates(path: Path | None) -> dict[str, int]:
    """Load {base_commit_sha: unix_ts} from the cache file. Empty dict if missing."""
    target = path or COMMIT_DATES_PATH
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, (int, float))}


def _load_commit_topo_ranks(path: Path | None) -> dict[str, int]:
    """Load {base_commit_sha: topo_rank} from the cache file.

    Topo rank = position in `git rev-list --topo-order --reverse --all` for
    that commit's repo. Smaller = earlier ancestor. Missing file or invalid
    JSON returns empty dict — sort falls back to date+SHA only.
    """
    target = path or COMMIT_TOPO_RANKS_PATH
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): int(v) for k, v in data.items() if isinstance(v, (int, float))}


def load_tasks(
    *,
    dataset_path: Path | None = None,
    commit_dates_path: Path | None = None,
    commit_topo_ranks_path: Path | None = None,
) -> list[Task]:
    """Load the pinned SWE-bench Pro split and convert rows to tasks."""

    dates = _load_commit_dates(commit_dates_path)
    topo = _load_commit_topo_ranks(commit_topo_ranks_path)
    return [
        _row_to_task(row, dates, topo)
        for row in load_swebench_pro_rows(dataset_path)
    ]


def _row_to_task(
    row: dict[str, Any],
    dates: dict[str, int] | None = None,
    topo_ranks: dict[str, int] | None = None,
) -> Task:
    dates = dates or {}
    topo_ranks = topo_ranks or {}
    base = row["base_commit"]
    return Task(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=base,
        problem_statement=row["problem_statement"],
        dockerhub_tag=row["dockerhub_tag"],
        requirements=row.get("requirements"),
        interface=row.get("interface"),
        commit_date=dates.get(base),
        commit_topo_rank=topo_ranks.get(base),
    )


def sort_tasks_chronologically(tasks: Iterable[Task]) -> list[Task]:
    """Return tasks ordered by base_commit timestamp.

    Sort key, in priority order:
      1. `commit_date` (UNIX ts) — primary chronological signal
      2. `commit_topo_rank` if both peers have one — ancestor-before-descendant
         when committer-timestamps tie (common for batch-rebased or imported
         commits that share a second)
      3. `base_commit` SHA lex — deterministic last-resort tiebreaker

    Tasks with `commit_date is None` are sorted LAST in their input order
    (preserves stability when the date cache is partially populated). This
    lets sequential mode degrade gracefully — missing dates mean those tasks
    run after all dated ones, which is no worse than today's behavior.

    `commit_topo_rank` defaults to `len(tasks) + 1` (i.e. last) when missing,
    so a task with a known rank always sorts before one without.
    """
    dated: list[tuple[int, Task]] = []
    undated: list[tuple[int, Task]] = []
    n = 0
    for i, t in enumerate(tasks):
        n += 1
        (dated if t.commit_date is not None else undated).append((i, t))
    sentinel_rank = n + 1
    dated.sort(
        key=lambda it: (
            it[1].commit_date,
            it[1].commit_topo_rank
            if it[1].commit_topo_rank is not None
            else sentinel_rank,
            it[1].base_commit,
        )
    )
    undated.sort(key=lambda it: it[0])  # stable
    return [t for _, t in dated] + [t for _, t in undated]


def filter_by_repo(tasks: Iterable[Task], repo: str) -> list[Task]:
    """Return tasks belonging to one upstream repo, preserving input order."""
    return [t for t in tasks if t.repo == repo]


def group_into_universes(tasks: Iterable[Task]) -> list[Universe]:
    """Group tasks by repo. Each universe preserves the input ordering."""
    by_repo: dict[str, list[Task]] = defaultdict(list)
    for t in tasks:
        by_repo[t.repo].append(t)
    return [Universe(repo=repo, tasks=tuple(ts)) for repo, ts in by_repo.items()]
