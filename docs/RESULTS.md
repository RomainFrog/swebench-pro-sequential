# Results and artifacts

A run writes immutable provenance at the output root and execution artifacts
under one directory per repetition:

```text
<output>/
├── run_manifest.json
├── task_manifest.json
└── run_0/
    ├── predictions.json
    ├── image_manifest.json
    ├── score/eval_results.json        # bulk scoring, when requested
    └── universes/<repo>/              # sequential mode
        ├── _mini_state/
        │   ├── usage.jsonl
        │   ├── compactions.jsonl
        │   └── compactions/*.json
        ├── <instance-id>/
        │   ├── container_image.json
        │   ├── patch.diff
        │   ├── task_result.json
        │   └── <instance-id>.mini.traj.json
        └── score/eval_results.json
```

Independent mode uses the same per-task files without shared conversation or
agent-home state. With multiple repositories, task directories are nested by
repository.

## Status and denominator rules

- `passed: true`: the official evaluator accepted the patch.
- `passed: false`: grading completed and the patch did not pass.
- `passed: null`: no trustworthy score exists. This is an infrastructure or
  incomplete-run outcome and is never counted as a benchmark failure.

Pass rate is `passed / graded`, where `graded` includes only boolean scores.
Reports always show the separate selected, completed, graded, passed, failed,
and infrastructure-failure counts so an incomplete run cannot look complete.
`Submitted` and `LimitsExceeded` are completed rollout statuses; other rollout
statuses are surfaced separately. Both inline grading and `swebench-seq score`
exclude nonterminal rollouts from `eval_results.json`; reports also enforce this
boundary when reading older or manually edited artifacts.

Token counts are provider-reported when available and `null` otherwise.
Compaction calls are included in run totals and labeled separately from agent
calls. Costs are not normalized across providers; publish provider billing
data separately when cost comparisons matter.

## Portable result set

For a reproducible result, retain and share:

1. `run_manifest.json`
2. `task_manifest.json`
3. every `run_N/image_manifest.json`
4. every `run_N/predictions.json`
5. evaluator score files
6. `swebench-seq report --format json` output

`validate-run` rejects source checkouts with tracked modifications. Untracked
files are recorded but do not invalidate a run because output directories and
local notes commonly coexist with a checkout.

Trajectories, agent-home snapshots, and compaction audits are useful for
research but may contain model-generated or repository-derived text. Review
them before publication even though credential-shaped configuration fields are
redacted automatically.

JSON Schemas for the run, task, image, per-task result, and report artifacts
are published in [`schemas/`](../schemas/). `swebench-seq validate-run` performs
the additional cross-file completeness checks that JSON Schema alone cannot.
