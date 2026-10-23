# Canonical sequence metadata

These files define the task order for `swebench-pro-sequential-v1`:

- `repo_commit_dates.json`: base commit SHA to committer timestamp.
- `repo_topo_ranks.json`: topological rank used to resolve commits sharing a
  timestamp.
- `sequential_order.json`: the frozen instance-ID order for each repository.

They were derived from the pinned `ScaleAI/SWE-bench_Pro` dataset and the Git
histories embedded in its public task images. Ordering uses commit time first,
topological rank for timestamp ties, and commit SHA as the deterministic final
tiebreaker. The resulting order was frozen so benchmark selection never
depends on network state or a mutable repository checkout.

`sequential_order.json` contains all 731 instance IDs from the pinned test
split exactly once across its 11 repositories. Runtime validation enforces
that coverage for canonical runs. These files contain identifiers and Git
metadata only; they do not contain gold patches, test patches, issue text, or
model output.

The current SHA-256 hashes are recorded automatically in every run manifest.
Changing any of these files requires a new benchmark version and an explanation
in `CHANGELOG.md`.
