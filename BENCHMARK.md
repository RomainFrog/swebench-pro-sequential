# Benchmark contract

This document defines `swebench-pro-sequential-v1`, the canonical benchmark
implemented by this repository. A run is canonical only when it uses the
pinned assets and protocol below. Custom datasets, image prefixes, models, and
debug limits remain useful experiments, but must be reported as non-canonical.
Run manifests expose `canonical_inputs` and `complete_suite` separately so a
canonical single-repository subset cannot be confused with the full benchmark.

## Pinned inputs

- Dataset: `ScaleAI/SWE-bench_Pro`, test split, revision
  `7ab5114912baf22bb098818e604c02fe7ad2c11f` (731 tasks in 11 repositories).
- Evaluator: `scaleapi/SWE-bench_Pro-os`, revision
  `0c64e26f00b9c190432de7fc520c8ceed5c25518`; its downloaded archive and
  extracted tree are checksum-verified.
- Task images: `docker.io/jefzda/sweap-images:<dockerhub_tag>` from each dataset
  row. The immutable image ID and registry digest observed by Podman are saved
  in every task directory and aggregated in `image_manifest.json`.
- Sequence: [`data/sequential_order.json`](data/sequential_order.json), which
  covers every task in the pinned dataset exactly once. Official runs fail
  closed if that coverage changes.
- Agent runtime: `mini-swe-agent==1.17.5` in the committed `uv.lock`.

The standard suite takes the first 40 chronologically ordered tasks from each
repository (or every task when a repository has fewer than 40), for 420 tasks
total. `--full-dataset`, `--limit`, target filters, local datasets, and local
evaluator directories produce non-standard runs whose resolved inputs are
recorded in their manifests.

## Sequential protocol

Each repository is one independent sequence. Within a sequence:

1. Tasks execute serially in the frozen order.
2. A fresh task container is started from the row's image and reset to its
   specified `base_commit`.
3. The image's existing Git history is replaced with a clean one-commit
   baseline before the agent starts, preventing later commits from being read.
4. Conversation messages, compaction state, `/home/agent`, and files under that
   home persist to the next task in the same repository. Nothing is shared
   across repositories or repeated runs.
5. The agent is explicitly told that the repository advanced to a new commit.
6. The produced `git diff HEAD` is graded immediately by the pinned public
   evaluator. Evaluation failures are recorded as infrastructure failures and
   do not stop unrelated tasks.
7. The task container is removed before the next task starts. Benchmark images
   remain in Podman's cache.

Independent mode selects the same ordered tasks but creates a fresh
conversation and agent home for each task.

Compaction is triggered when estimated or provider-reported context use reaches
`context_window - 20,000` tokens. `--compact-after-each-task` additionally
forces a boundary summary. Summaries are structurally validated, repaired once
when invalid, and replaced by a deterministic fallback if model summarization
still fails. Every compaction has an audit record.

## Failure and retry policy

`Submitted` and `LimitsExceeded` are terminal rollout statuses. Their patches,
including legitimate empty patches, are gradeable results. API errors,
timeouts, missing trajectories, missing scores, and evaluator failures make the
overall command exit non-zero after other independent work has finished.

`--retry` is recovery, not retry-until-pass. It restores the conversation and
agent-home snapshot immediately before the first incomplete task, reruns that
task once, and continues through the remaining suffix. It never retries a
correctly completed task merely because the patch failed tests.

## Leakage boundary

The agent receives the issue description and the repository at `base_commit`.
Gold patches, test patches, evaluator files, score files, and later Git commits
are not mounted into the rollout container. Model credentials remain on the
host. Persisted configuration, trajectories, compaction audits, and manifests
are recursively redacted before writing.

Rollout containers use Podman's default network unless the operator configures
an explicit network mode. Evaluator containers always run with network access
blocked. Proxy values forwarded to a rollout are visible inside that container;
operators should not embed reusable credentials in proxy URLs.

## Reproducibility and reporting

Every run records the exact task plan, dataset hash and revision, evaluator
revision, frozen-order hashes, harness version and source commit, dependency
versions, lockfile hash when available, Python/platform/Podman versions, model
configuration with credentials removed, and resolved task-image identities.

Model sampling and external provider implementations may still be
nondeterministic. Comparable results must publish the complete sanitized run
manifest, task manifest, image manifest, predictions, scores, and JSON report.
See [`docs/RESULTS.md`](docs/RESULTS.md) for artifact and metric semantics.
