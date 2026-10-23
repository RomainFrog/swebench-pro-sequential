# Contributing

Thank you for improving swebench-seq. Changes should preserve the benchmark
contract in [`BENCHMARK.md`](BENCHMARK.md); behavior that changes task
selection, ordering, state carryover, leakage boundaries, or scoring requires a
new benchmark version and a changelog entry.

## Development setup

```bash
git clone https://github.com/RomainFrog/swebench-pro-sequential.git
cd swebench-pro-sequential
uv sync --frozen --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src
uv build
```

CI tests Python 3.11 and 3.12, the provider-free workflow, live rootless
Podman, and installation from the built wheel.

Use `uv run pytest -m integration` for the provider-free workflow test. Before
submitting changes to Podman, model-provider, or evaluator boundaries, also run
the public two-task smoke described in the README when credentials and disk
space are available.

The live Podman fixture is opt-in locally:

```bash
SWEBENCH_RUN_PODMAN_TESTS=1 uv run pytest -m podman
```

Pull requests should be focused, include tests for behavior changes, avoid
committing run artifacts or credentials, and explain any reproducibility
impact. Run `uv lock` only when dependencies intentionally change, and commit
the resulting lockfile.

## Architecture

The runtime path is:

```text
CLI → experiment config → run orchestration → task execution → Podman + agent/evaluator
      experiment.py       orchestration/run.py  execution.py    runner.py
                                                            agent/  evaluation/
```

`runner.py` owns task-container isolation. `agent/` owns mini-swe-agent,
conversation state, compaction, provider adapters, and persistent agent home.
`evaluation/` owns the pinned public dataset and evaluator boundary.

```text
src/swebench_seq/
├── cli.py               # operational commands
├── commands/run.py      # run options and CLI-to-service boundary
├── experiment.py        # experiment YAML and task-selector parsing
├── runner.py            # task-container lifecycle and Podman wrapper
├── agent/               # mini-swe-agent integration and persistent state
├── orchestration/       # planning, task execution, and recovery
├── evaluation/          # public benchmark data and grading boundary
├── tasks.py             # dataset loading, ordering, and grouping
├── manifest.py          # provenance and exact task plans
├── redaction.py         # credential and URL sanitization
├── predictions.py       # crash-safe prediction accumulation
└── configs/             # complete runnable configurations
```

## Reporting problems

Use a GitHub issue for reproducible bugs and feature proposals. Do not include
API keys, private model traces, proprietary source, or unreviewed agent-home
artifacts. Report security issues privately as described in
[`SECURITY.md`](SECURITY.md).
