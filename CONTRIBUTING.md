# Contributing

Thank you for improving swebench-seq. Changes should preserve the benchmark
contract in [`BENCHMARK.md`](BENCHMARK.md); behavior that changes task
selection, ordering, state carryover, leakage boundaries, or scoring requires a
new benchmark version and a changelog entry.

## Development setup

```bash
git clone https://github.com/RomainFrog/swebenchpro-seq.git
cd swebenchpro-seq
uv sync --frozen --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src
```

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

## Reporting problems

Use a GitHub issue for reproducible bugs and feature proposals. Do not include
API keys, private model traces, proprietary source, or unreviewed agent-home
artifacts. Report security issues privately as described in
[`SECURITY.md`](SECURITY.md).
