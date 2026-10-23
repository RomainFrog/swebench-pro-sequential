"""Core dataclasses shared across the package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One SWE-bench Pro task — a single bug to fix in one repo at one commit."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    dockerhub_tag: str
    requirements: str | None = None
    interface: str | None = None
    # UNIX timestamp of the base_commit's committer date. None when the date
    # cache (data/repo_commit_dates.json) hasn't been populated. Used for
    # chronological ordering in --sequential mode so the agent's carried-over
    # MEMORY.md doesn't contain knowledge of future fixes.
    commit_date: int | None = None
    # Position of the base_commit in the repo's `git rev-list --topo-order
    # --reverse --all` output. Smaller = earlier ancestor. Used to tiebreak
    # when two commits share the same committer-second — ancestor goes first
    # regardless of SHA. Populated from data/repo_topo_ranks.json.
    commit_topo_rank: int | None = None


@dataclass(frozen=True)
class Universe:
    """A group of tasks that share a `/home/agent/` lineage in sequential mode.

    For SWE-bench Pro V1 a universe == one upstream repo.
    """

    repo: str
    tasks: tuple[Task, ...]

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass
class RunResult:
    """Outcome of running a single task."""

    patch: str  # unified diff produced by `git diff HEAD` inside the container
    turns_used: int
    wall_seconds: float
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
