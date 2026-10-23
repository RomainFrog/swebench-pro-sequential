# Expected smoke-test result

Run:

```bash
export OPENAI_API_KEY=...
uv run swebench-seq doctor --config public-smoke-responses-podman
uv run swebench-seq run --config public-smoke-responses-podman
uv run swebench-seq validate-run --run-dir /tmp/swebench-seq-public-smoke/run_0
uv run swebench-seq report --run-dir /tmp/swebench-seq-public-smoke/run_0
```

A successful infrastructure smoke selects two ordered `flipt-io/flipt` tasks,
starts and removes a different Podman container for each base commit, records
one prediction and one official score per task, carries `/home/agent` into the
second task, and ends with:

```text
[swebench-seq] run_0: wrote 2 prediction(s) → .../run_0/predictions.json
[validate-run] complete with required provenance: .../run_0
```

The smoke limits each rollout to one model step. `LimitsExceeded`, an empty
patch, and a `false` score are therefore valid smoke outcomes: this command
tests wiring and isolation, not solve quality. A missing prediction, missing
score, API error, evaluator failure, or leaked/live task container is not a
successful smoke.

Exact timings, token counts, image IDs, and digests differ by machine and
provider. The sanitized JSON report is the portable summary:

```json
{
  "schema_version": 1,
  "benchmark_version": "swebench-pro-sequential-v1",
  "canonical_inputs": true,
  "complete_suite": false,
  "mode": "sequential",
  "summary": {
    "selected": 2,
    "artifacts": 2,
    "completed_rollouts": 2,
    "graded": 2,
    "passed": 0,
    "failed": 2,
    "infrastructure_or_incomplete": 0,
    "pass_rate": 0.0,
    "model_calls": 2,
    "compaction_calls": 0
  }
}
```
