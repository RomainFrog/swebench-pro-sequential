# Experiment configurations

Each YAML file defines one complete `swebench-seq run`. Pass either a bundled
file stem or the path to your own file:

```bash
uv run swebench-seq run --config sequential-responses-podman
uv run swebench-seq run --config ./my-experiment.yaml
```

Explicit CLI flags override values from the file. Unknown top-level keys fail
validation instead of being ignored. Repeated CLI `--set` values are applied
after the file's `set` block; overriding one nested key does not discard the
other configured overrides.

Validate a configuration and its explicitly named credential without making a
model request:

```bash
uv run swebench-seq doctor --config sequential-responses-podman
```

## Bundled configurations

- `public-smoke-responses-podman`: two sequential tasks, each bounded to one
  model call and 120 seconds, for installation, state carryover, container,
  evaluator, and API connectivity checks.
- `sequential-responses-podman`: the full sequential protocol.
- `independent-responses-podman`: a matching full independent baseline.

The files intentionally contain no credentials. Set `api_key_env` to the name
of an environment variable, or rely on the provider default
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`).

## Model configuration

The `model` block replaces mini-swe-agent's base model section:

```yaml
provider: openai
api_key_env: OPENAI_API_KEY
base_url: https://api.openai.com/v1

model:
  model_name: gpt-5
  model_class: swebench_seq.agent.responses.OpenAIResponsesModel
  cost_tracking: ignore_errors
  model_kwargs:
    reasoning_effort: medium
    max_output_tokens: 16000
    bash_tool: true
    replay_encrypted_reasoning: true
```

For a compatible gateway, change `base_url`, `model_name`, and `api_key_env`.
For a LiteLLM-supported chat model, set `model_class: litellm` and use the
provider's supported model options.

Responses API runs automatically send a stable `prompt_cache_key` for each
repository, so a sequential universe keeps the same prompt-cache routing key
across all of its tasks. OpenAI-compatible gateways that require the same key
in a custom affinity header can name that header without provider-specific
code:

```yaml
model:
  model_kwargs:
    prompt_cache_key_header: X-Sticky-Routing-Key
```

`prompt_cache_key_header` is consumed by the harness and is not included in
the Responses request body. The generated value is sent both as the standard
`prompt_cache_key` field and as the configured header.

TLS certificate verification is enabled by default. Setting
`model_kwargs.tls_verify` to `false` is intended only for isolated testing
against an endpoint you control; the runtime emits a warning whenever it is
used.

For example, Anthropic and OpenRouter use mini-swe-agent's LiteLLM model:

```yaml
# Anthropic
provider: anthropic
api_key_env: ANTHROPIC_API_KEY
model:
  model_name: anthropic/YOUR_MODEL_ID
  model_class: litellm
  cost_tracking: ignore_errors
```

```yaml
# OpenRouter
provider: openrouter
api_key_env: OPENROUTER_API_KEY
model:
  model_name: openrouter/YOUR_PROVIDER/YOUR_MODEL_ID
  model_class: litellm
  cost_tracking: ignore_errors
```

## Run keys

Top-level keys are the snake-case names shown by
`uv run swebench-seq run --help`. Common keys include:

- task selection: `repo`, `all_repos`, `full_dataset`, `limit`;
- execution: `sequential`, `concurrency`, `n_runs`, `overall_timeout`;
- paths: `output_dir`, `evaluator_dir`, `image_prefix`;
- data: `dataset` for an explicit local JSONL/CSV file;
- model access: `provider`, `api_key_env`, `base_url`;
- context: `context_window`, `compact_after_each_task`;
- recovery: `skip_existing_patches`, `retry`;
- planning: `dry_run` prints the resolved configuration and ordered tasks
  without starting containers or writing artifacts.

Use `set` to change nested model values without copying the whole model block:

```yaml
set:
  model.model_kwargs.reasoning_effort: low
```
