# Canonical sequence metadata

These files define the task order for `swebench-pro-sequential-v1`:

- `repo_commit_dates.json`: base commit SHA to committer timestamp.
- `repo_topo_ranks.json`: topological rank used to resolve commits sharing a
  timestamp.
- `sequential_order.json`: the frozen instance-ID order for each repository.

They were derived from the pinned `ScaleAI/SWE-bench_Pro` dataset and the
public Git histories of its 11 repositories. Commits are content-addressed, so
these timestamps match the histories embedded in the task images. Every task's
`base_commit` has a committer timestamp. Topological ranks are recorded for
repositories in which two base commits share a timestamp. Ordering uses commit
time first, topological rank for timestamp ties (ancestor first), and commit
SHA as the deterministic final tiebreaker; `sequential_order.json` is the output
of `swebench_seq.tasks.sort_tasks_chronologically` over these files. The
resulting order was frozen so benchmark selection never depends on network
state or a mutable repository checkout.

`sequential_order.json` contains all 731 instance IDs from the pinned test
split exactly once across its 11 repositories. Runtime validation enforces
that coverage for canonical runs. These files contain identifiers and Git
metadata only; they do not contain gold patches, test patches, issue text, or
model output.

The current SHA-256 hashes are recorded automatically in every run manifest.
Changing any of these files requires a new benchmark version and an explanation
in `CHANGELOG.md`.
