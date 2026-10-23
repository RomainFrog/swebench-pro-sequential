"""Tests for mini-swe-agent config preparation."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from swebench_seq.agent.compaction import (
    coalesce_consecutive_messages,
    estimate_message_tokens,
    history_compaction_messages,
    locate_current_task_index,
    recent_keep_start_index,
    split_messages_for_compaction,
    validate_compaction_summary,
)
from swebench_seq.agent.config import (
    MiniSweAgentOptions,
    normalize_model_name,
    prepare_config,
    prepare_sequential_config,
    prompt_cache_key_for_repo,
    with_repo_prompt_cache_key,
)
from swebench_seq.agent.driver import (
    MiniConversationSession,
    _bound_memory_content,
    _format_system_prompt_mode_task_text,
    _inject_system_memory_context,
    _run_agent_with_deadline,
    extract_mini_usage,
    render_agent_home_memory_context,
)
from swebench_seq.agent.environment import ContainerBackedEnvironment, DeadlineModel
from swebench_seq.agent.providers import (
    sanitize_openai_chat_messages,
    wrap_model_for_provider,
)
from swebench_seq.runner import ContainerError
from swebench_seq.types import Task


def test_prepare_config_injects_runtime_secrets_for_responses_model() -> None:
    prepared = prepare_config(
        {
            "model": {
                "model_class": "swebench_seq.agent.responses.OpenAIResponsesModel",
                "model_kwargs": {"reasoning_effort": "high"},
            }
        },
        MiniSweAgentOptions(
            provider="openai",
            api_key="sk-test",
            base_url="https://api.example.test/v1",
        ),
    )

    kwargs = prepared["model"]["model_kwargs"]
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "https://api.example.test/v1"
    # reasoning effort comes from the config file, untouched by the driver.
    assert kwargs["reasoning_effort"] == "high"
    # Responses model must NOT receive litellm-only chat defaults.
    assert "drop_params" not in kwargs
    assert "temperature" not in kwargs
    assert "custom_llm_provider" not in kwargs
    assert "extra_headers" not in kwargs


def test_repo_prompt_cache_key_is_stable() -> None:
    options = MiniSweAgentOptions()

    first = with_repo_prompt_cache_key(options, "NodeBB/NodeBB")
    repeated = with_repo_prompt_cache_key(options, "NodeBB/NodeBB")
    other = with_repo_prompt_cache_key(options, "qutebrowser/qutebrowser")

    assert first.prompt_cache_key == repeated.prompt_cache_key
    assert first.prompt_cache_key != other.prompt_cache_key
    assert first.prompt_cache_key == prompt_cache_key_for_repo("NodeBB/NodeBB")


def test_prepare_config_adds_standard_and_gateway_cache_routing_keys() -> None:
    cache_key = prompt_cache_key_for_repo("NodeBB/NodeBB")
    prepared = prepare_config(
        {
            "model": {
                "model_class": "swebench_seq.agent.responses.OpenAIResponsesModel",
                "model_kwargs": {
                    "prompt_cache_key_header": "X-Sticky-Routing-Key",
                },
            }
        },
        MiniSweAgentOptions(prompt_cache_key=cache_key),
    )

    kwargs = prepared["model"]["model_kwargs"]
    assert kwargs["prompt_cache_key"] == cache_key
    assert kwargs["extra_headers"] == {"X-Sticky-Routing-Key": cache_key}
    assert "prompt_cache_key_header" not in kwargs


def test_prepare_config_keeps_thinking_from_config_and_skips_temperature() -> None:
    prepared = prepare_config(
        {
            "model": {
                "model_class": "litellm",
                "model_kwargs": {
                    "drop_params": False,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"},
                },
            }
        },
        MiniSweAgentOptions(provider="anthropic"),
    )

    kwargs = prepared["model"]["model_kwargs"]
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}
    # thinking present → driver must not inject a sampling temperature.
    assert "temperature" not in kwargs


def test_prepare_config_drops_inherited_temperature_with_thinking() -> None:
    # Regression: mini's base config sets model_kwargs.temperature, which merges
    # into a thinking overlay; Anthropic rejects temperature with thinking, so
    # the driver must drop the inherited temperature (not just skip adding it).
    prepared = prepare_config(
        {
            "model": {
                "model_class": "litellm",
                "model_kwargs": {
                    "temperature": 0.0,  # inherited from the base swebench.yaml
                    "thinking": {"type": "adaptive"},
                },
            }
        },
        MiniSweAgentOptions(provider="anthropic"),
    )
    assert "temperature" not in prepared["model"]["model_kwargs"]


def test_prepare_config_normalizes_anthropic_api_base_to_messages() -> None:
    prepared = prepare_config(
        {"model": {"model_class": "litellm", "model_kwargs": {}}},
        MiniSweAgentOptions(
            provider="anthropic",
            base_url="https://api.example.test/v1",
        ),
    )

    assert prepared["model"]["model_kwargs"]["api_base"].endswith("/messages")


def test_openrouter_model_name_gets_provider_prefix() -> None:
    assert normalize_model_name("anthropic/claude-sonnet", "openrouter") == (
        "openrouter/anthropic/claude-sonnet"
    )
    assert normalize_model_name("openrouter/openai/gpt-5", "openrouter") == (
        "openrouter/openai/gpt-5"
    )


def test_prepare_config_sets_openai_chat_timeout() -> None:
    prepared = prepare_config(
        {"model": {"model_kwargs": {}}},
        MiniSweAgentOptions(
            provider="openai",
            base_url="https://api.example.test/v1",
        ),
    )

    kwargs = prepared["model"]["model_kwargs"]
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["timeout"] == 180


def test_prepare_config_preserves_explicit_openai_chat_timeout() -> None:
    prepared = prepare_config(
        {"model": {"model_kwargs": {"timeout": 42}}},
        MiniSweAgentOptions(
            provider="openai",
            base_url="https://api.example.test/v1",
        ),
    )

    assert prepared["model"]["model_kwargs"]["timeout"] == 42


def test_prepare_sequential_config_honors_explicit_step_limit() -> None:
    prepared = prepare_sequential_config(
        {"agent": {"system_template": "system", "step_limit": 250}},
        MiniSweAgentOptions(),
    )

    assert prepared["agent"]["step_limit"] == 250


def test_prepare_sequential_config_does_not_lower_step_limit() -> None:
    prepared = prepare_sequential_config(
        {"agent": {"system_template": "system", "step_limit": 750}},
        MiniSweAgentOptions(),
    )

    assert prepared["agent"]["step_limit"] == 750


def test_prepare_sequential_config_can_promote_task_instructions_to_system() -> None:
    prepared = prepare_sequential_config(
        {
            "agent": {
                "system_template": "base system",
                "instance_template": (
                    "<pr_description>\n{{task}}\n</pr_description>\n\n"
                    "<instructions>\n# Task Instructions\nDo not edit tests.\n"
                    "</instructions>"
                ),
                "step_limit": 750,
            }
        },
        MiniSweAgentOptions(),
    )

    agent_cfg = prepared["agent"]
    assert "Task-invariant SWE-bench instructions:" in agent_cfg["system_template"]
    assert "# Task Instructions" in agent_cfg["system_template"]
    assert "Sequential mode:" in agent_cfg["system_template"]
    assert agent_cfg["instance_template"] == "{{task}}"


def test_system_prompt_mode_task_text_contains_only_pr_description() -> None:
    task = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="fix the thing",
        dockerhub_tag="tag",
    )

    text = _format_system_prompt_mode_task_text(task)

    assert "<pr_description>" in text
    assert "fix the thing" in text
    assert "<instructions>" not in text
    assert "# Task Instructions" not in text


class FakeRunner:
    def __init__(
        self,
        result: subprocess.CompletedProcess[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.cmd = []
        self.kwargs = {}
        self.result = result or subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="out",
            stderr="err",
        )
        self.error = error

    def exec_in_container(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


def test_container_backed_environment_applies_config_and_forwarded_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FORWARDED", "host-value")
    monkeypatch.setenv("OVERRIDE", "host-value")
    runner = FakeRunner()
    env = ContainerBackedEnvironment(
        runner,
        env={"PAGER": "cat", "VALUE": "space value", "OVERRIDE": "config-value"},
        forward_env=["FORWARDED", "OVERRIDE", "MISSING"],
    )

    env.execute("echo $PAGER:$VALUE")

    assert runner.cmd == [
        "bash",
        "-c",
        "export FORWARDED=host-value OVERRIDE=config-value PAGER=cat "
        "VALUE='space value'; timeout -k 5 60 "
        "bash -c 'echo $PAGER:$VALUE'",
    ]
    assert runner.kwargs["workdir"] == "/app"
    assert runner.kwargs["timeout"] == 75


def test_container_backed_environment_returns_normal_nonzero_result() -> None:
    runner = FakeRunner(
        subprocess.CompletedProcess(args=[], returncode=2, stdout="out", stderr="err")
    )
    env = ContainerBackedEnvironment(runner, timeout=9)

    result = env.execute("exit 2")

    assert result == {"output": "outerr", "returncode": 2}
    assert runner.kwargs["timeout"] == 24


def test_container_backed_environment_caps_command_to_task_deadline(
    monkeypatch,
) -> None:
    monkeypatch.setattr("swebench_seq.agent.environment.time.monotonic", lambda: 100.0)
    runner = FakeRunner()
    env = ContainerBackedEnvironment(runner, timeout=60, deadline=102.5)

    env.execute("echo bounded")

    assert "timeout -k 5 2.5 bash" in runner.cmd[-1]
    assert runner.kwargs["timeout"] == 17.5


def test_deadline_model_caps_model_request_to_remaining_time(monkeypatch) -> None:
    monkeypatch.setattr("swebench_seq.agent.environment.time.monotonic", lambda: 40.0)

    class Model:
        def __init__(self) -> None:
            self.timeout = None

        def query(self, _messages, **kwargs):
            self.timeout = kwargs["timeout"]
            return {"content": "ok"}

    inner = Model()
    model = DeadlineModel(inner, deadline=43.0)

    assert model.query([]) == {"content": "ok"}
    assert inner.timeout == 3.0


def test_fresh_agent_deadline_stops_before_first_step() -> None:
    class Agent:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                system_template="system",
                instance_template="{{task}}",
            )
            self.extra_template_vars = {}
            self.messages = []
            self.steps = 0

        def render_template(self, template):
            return template.replace("{{task}}", self.extra_template_vars.get("task", ""))

        def add_message(self, role, content):
            self.messages.append({"role": role, "content": content})

        def step(self):
            self.steps += 1

    agent = Agent()
    status, _ = _run_agent_with_deadline(
        agent,
        {
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
        },
        task_text="fix it",
        deadline=0.0,
    )

    assert status == "AgentTimeout"
    assert agent.steps == 0


def test_container_backed_environment_bounds_large_normal_output() -> None:
    large_output = ("H" * 4_000) + ("M" * 8_000) + ("T" * 4_000)
    runner = FakeRunner(
        subprocess.CompletedProcess(args=[], returncode=0, stdout=large_output, stderr="")
    )
    env = ContainerBackedEnvironment(runner)

    result = env.execute("cat huge.log")

    output = result["output"]
    assert len(output) < 10_000
    assert "Command output was truncated by swebench-seq" in output
    assert "Original output length: 16000 characters" in output
    assert "HHHH" in output
    assert "TTTT" in output
    assert "MMMM" not in output


def test_container_backed_environment_preserves_submit_sentinel_when_bounding() -> None:
    large_diff = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n" + (
        ("H" * 4_000) + ("M" * 8_000) + ("T" * 4_000)
    )
    runner = FakeRunner(
        subprocess.CompletedProcess(args=[], returncode=0, stdout=large_diff, stderr="")
    )
    env = ContainerBackedEnvironment(runner)

    result = env.execute("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff")

    output = result["output"]
    assert output.startswith("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")
    assert "Command output was truncated by swebench-seq" in output
    assert "HHHH" in output
    assert "TTTT" in output
    assert "MMMM" not in output


@pytest.mark.parametrize("returncode", [124, 137, 143])
def test_container_backed_environment_translates_timeout_return_codes(
    returncode: int,
) -> None:
    runner = FakeRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout="partial",
            stderr="stderr",
        )
    )
    env = ContainerBackedEnvironment(runner, timeout=3)

    with pytest.raises(subprocess.TimeoutExpired) as exc:
        env.execute("sleep 999")

    assert exc.value.cmd == "sleep 999"
    assert exc.value.timeout == 3
    assert exc.value.output == b"partialstderr"


def test_container_backed_environment_bounds_timeout_output() -> None:
    large_output = ("H" * 4_000) + ("M" * 8_000) + ("T" * 4_000)
    runner = FakeRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=124,
            stdout=large_output,
            stderr="",
        )
    )
    env = ContainerBackedEnvironment(runner, timeout=3)

    with pytest.raises(subprocess.TimeoutExpired) as exc:
        env.execute("grep huge")

    output = exc.value.output.decode()
    assert len(output) < 10_000
    assert "Command output was truncated by swebench-seq" in output
    assert "Original output length: 16000 characters" in output
    assert "HHHH" in output
    assert "TTTT" in output
    assert "MMMM" not in output


def test_container_backed_environment_translates_podman_timeout() -> None:
    runner = FakeRunner(error=ContainerError("podman timed out: podman exec ..."))
    env = ContainerBackedEnvironment(runner, timeout=3)

    with pytest.raises(subprocess.TimeoutExpired) as exc:
        env.execute("sleep 999")

    assert exc.value.cmd == "sleep 999"
    assert exc.value.timeout == 3
    assert b"podman timed out" in exc.value.output


def test_container_backed_environment_propagates_non_timeout_container_error() -> None:
    runner = FakeRunner(error=ContainerError("container not started"))
    env = ContainerBackedEnvironment(runner)

    with pytest.raises(ContainerError, match="container not started"):
        env.execute("echo nope")


def test_render_agent_home_memory_context_includes_exact_memory_files(
    tmp_path,
) -> None:
    home = tmp_path / "home"
    (home / "memory").mkdir(parents=True)
    (home / "MEMORY.md").write_text("# memory\n- keep me\n")
    (home / "memory" / "2026-05-26.md").write_text("daily note\n")

    rendered = render_agent_home_memory_context(home)

    assert '<file path="/home/agent/MEMORY.md">' in rendered
    assert "# memory\n- keep me\n" in rendered
    assert "/home/agent/memory/2026-05-26.md" not in rendered
    assert "daily note\n" not in rendered


def test_inject_system_memory_context_places_memory_after_instructions() -> None:
    system = (
        "base\n\n"
        "Task-invariant SWE-bench instructions:\nRules\n\n"
        "Sequential mode:\nUse memory."
    )
    memory = "<sequential_memory>\nmem\n</sequential_memory>"

    rendered = _inject_system_memory_context(system, memory)

    assert rendered.index("Rules") < rendered.index("<sequential_memory>")
    assert rendered.index("</sequential_memory>") < rendered.index("Sequential mode:")


def test_inject_system_memory_context_replaces_prior_memory() -> None:
    system = (
        "base\n\n"
        "<sequential_memory>\nold\n</sequential_memory>\n\n"
        "Sequential mode:\nUse memory."
    )
    memory = "<sequential_memory>\nnew\n</sequential_memory>"

    rendered = _inject_system_memory_context(system, memory)

    assert "old" not in rendered
    assert "new" in rendered
    assert rendered.count("<sequential_memory>") == 1


def test_extract_mini_usage_reads_litellm_usage_shape() -> None:
    usage = extract_mini_usage(
        {
            "content": "ok",
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 7,
                        "total_tokens": 107,
                    }
                }
            },
        }
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 7
    assert usage.total_tokens == 107


def test_extract_mini_usage_counts_anthropic_cache_tokens() -> None:
    usage = extract_mini_usage(
        {
            "extra": {
                "response": {
                    "usage": {
                        "input_tokens": 100,
                        "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 30,
                        "output_tokens": 5,
                    }
                }
            }
        }
    )

    assert usage.input_tokens == 150
    assert usage.output_tokens == 5
    assert usage.cache_creation_input_tokens == 20
    assert usage.cache_read_input_tokens == 30


def test_extract_mini_usage_does_not_double_count_litellm_prompt_cache() -> None:
    usage = extract_mini_usage(
        {
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 7,
                        "total_tokens": 107,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 90,
                    }
                }
            }
        }
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 7
    assert usage.total_tokens == 107
    assert usage.cache_creation_input_tokens == 5
    assert usage.cache_read_input_tokens == 90


def test_extract_mini_usage_reads_responses_cache_details_without_double_counting() -> None:
    usage = extract_mini_usage(
        {
            "extra": {
                "response": {
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {
                            "cached_tokens": 80,
                            "cache_write_tokens": 10,
                        },
                        "output_tokens": 5,
                        "total_tokens": 105,
                    }
                }
            }
        }
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 5
    assert usage.total_tokens == 105
    assert usage.cache_creation_input_tokens == 10
    assert usage.cache_read_input_tokens == 80


def test_estimate_message_tokens_counts_content_only() -> None:
    assert estimate_message_tokens([{"role": "user", "content": "x" * 40}]) >= 10


def test_sanitize_openai_chat_messages_drops_mini_bookkeeping() -> None:
    original = [
        {
            "role": "assistant",
            "content": "THOUGHT: done",
            "extra": {"response": {"id": "chatcmpl"}},
            "cache_control": {"type": "ephemeral"},
            "tool_calls": [{"id": "call_1"}],
        }
    ]

    sanitized = sanitize_openai_chat_messages(original)

    assert sanitized == [
        {
            "role": "assistant",
            "content": "THOUGHT: done",
            "tool_calls": [{"id": "call_1"}],
        }
    ]
    assert "extra" in original[0]


def test_anthropic_wrapper_promotes_signed_thinking_blocks() -> None:
    raw_blocks = [
        {"type": "thinking", "thinking": "plan", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]

    class Model:
        n_calls = 0
        cost = 0.0

        def query(self, messages, **_kwargs):
            self.messages = messages
            return {
                "content": "THOUGHT: visible\n```bash\ntrue\n```",
                "extra": {
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "thinking_blocks": raw_blocks,
                                    "reasoning_content": "plan",
                                }
                            }
                        ]
                    }
                },
            }

    messages = [{"role": "user", "content": "task"}]
    response = wrap_model_for_provider(Model(), "anthropic").query(messages)

    assert response["content"].startswith("THOUGHT:")
    assert response["thinking_blocks"] == raw_blocks
    assert response["thinking_blocks"] is not raw_blocks
    response["thinking_blocks"][0]["thinking"] = "mutated"
    assert raw_blocks[0]["thinking"] == "plan"


def test_anthropic_wrapper_does_not_replay_unsigned_reasoning_content() -> None:
    class Model:
        n_calls = 0
        cost = 0.0

        def query(self, _messages, **_kwargs):
            return {
                "content": "visible",
                "extra": {
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "reasoning_content": "unsigned",
                                    "thinking_blocks": [
                                        {
                                            "type": "thinking",
                                            "thinking": "missing signature",
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                },
            }

    response = wrap_model_for_provider(Model(), "anthropic").query(
        [{"role": "user", "content": "task"}]
    )

    assert "thinking_blocks" not in response


def test_anthropic_wrapper_reads_provider_specific_thinking_blocks() -> None:
    raw_blocks = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]

    class Model:
        n_calls = 0
        cost = 0.0

        def query(self, _messages, **_kwargs):
            return {
                "content": "visible",
                "extra": {
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "provider_specific_fields": {
                                        "thinking_blocks": raw_blocks,
                                    }
                                }
                            }
                        ]
                    }
                },
            }

    response = wrap_model_for_provider(Model(), "anthropic").query(
        [{"role": "user", "content": "task"}]
    )

    assert response["thinking_blocks"] == raw_blocks


class Submitted(Exception):
    pass


class NonTerminatingException(Exception):
    pass


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.n_calls = 0
        self.cost = 0.0
        self.seen_messages: list[list[dict]] = []

    def query(self, messages):
        self.n_calls += 1
        self.seen_messages.append(json.loads(json.dumps(messages)))
        return {
            "content": self.content,
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 5,
                        "total_tokens": 47,
                    }
                }
            },
        }

    def get_template_vars(self):
        return {}


class FakeDefaultAgent:
    def __init__(self, model, env, **kwargs) -> None:
        self.model = model
        self.env = env
        self.config = SimpleNamespace(**kwargs)
        self.messages = []
        self.extra_template_vars = {}

    def render_template(self, template: str, **_kwargs) -> str:
        return template.replace("{{task}}", self.extra_template_vars.get("task", ""))

    def add_message(self, role: str, content: str, **kwargs) -> None:
        self.messages.append({"role": role, "content": content, **kwargs})

    def query(self):
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, _response):
        raise Submitted("done")


class FakeSequentialRunner:
    container_name = "fake-container"
    podman = object()

    def __init__(self, patch: str = "diff --git a/x b/x\n") -> None:
        self.patch = patch
        self.execs = []

    def exec_in_container(self, cmd, **kwargs):
        self.execs.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def extract_patch(self):
        return self.patch


VALID_COMPACTION_SUMMARY = """\
## Goal
- Continue the SWE-bench task.

## Constraints & Preferences
- Preserve the mini command protocol.

## Progress
### Done
- [x] Inspected the previous task context.

### In Progress
- [ ] Continue the active task.

### Blocked
- (none)

## Key Decisions
- **Structured compaction**: Preserve only facts needed to continue.

## Next Steps
1. Continue from the retained recent messages.

## Critical Context
- Compact summary.
"""


def test_mini_conversation_session_resumes_messages_across_tasks(
    tmp_path,
    monkeypatch,
) -> None:
    models: list[FakeModel] = []

    def fake_get_model(*, config):
        model = FakeModel("THOUGHT: done\n```bash\necho done\n```")
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "<pr_description>\n{{task}}\n</pr_description>\n\n"
                    "<instructions>\nRules\n</instructions>"
                ),
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=tmp_path / "home",
        context_window=1_000_000,
    )
    first = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="first issue",
        dockerhub_tag="tag",
    )
    second = Task(
        instance_id="i2",
        repo="repo/repo",
        base_commit="b",
        problem_statement="second issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=first,
        artifacts_dir=tmp_path / "i1",
    )
    session.run_task(
        runner=FakeSequentialRunner(),
        task=second,
        artifacts_dir=tmp_path / "i2",
    )

    assert len(models) == 2
    first_task_text = (tmp_path / "i1" / "mini_task.txt").read_text()
    second_task_text = (tmp_path / "i2" / "mini_task.txt").read_text()
    boundary = (
        "The repository has advanced to a new commit. "
        "Continue with the next task below."
    )
    assert boundary not in first_task_text
    assert boundary in second_task_text
    assert second_task_text.index(boundary) < second_task_text.index("second issue")
    second_call_messages = models[1].seen_messages[0]
    assert any("first issue" in msg.get("content", "") for msg in second_call_messages)
    assert any("second issue" in msg.get("content", "") for msg in second_call_messages)
    saved = json.loads((tmp_path / "state" / "messages.json").read_text())
    assert len(saved["messages"]) >= 5
    assert not any(
        msg.get("role") == "user" and msg.get("content") == "done"
        for msg in saved["messages"]
    )


def test_mini_conversation_puts_memory_in_system_not_task_messages(
    tmp_path,
    monkeypatch,
) -> None:
    models: list[FakeModel] = []

    def fake_get_model(*, config):
        model = FakeModel("THOUGHT: done\n```bash\necho done\n```")
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": (
                    "<pr_description>\n{{task}}\n</pr_description>\n\n"
                    "<instructions>\nRules\n</instructions>"
                ),
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    home = tmp_path / "home"
    home.mkdir()
    (home / "MEMORY.md").write_text("# repo memory\n- durable fact\n")
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=home,
        context_window=1_000_000,
    )
    first = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="first issue",
        dockerhub_tag="tag",
    )
    second = Task(
        instance_id="i2",
        repo="repo/repo",
        base_commit="b",
        problem_statement="second issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=first,
        artifacts_dir=tmp_path / "i1",
    )
    session.run_task(
        runner=FakeSequentialRunner(),
        task=second,
        artifacts_dir=tmp_path / "i2",
    )

    first_task_text = (tmp_path / "i1" / "mini_task.txt").read_text()
    second_task_text = (tmp_path / "i2" / "mini_task.txt").read_text()
    assert '<file path="/home/agent/MEMORY.md">' not in first_task_text
    assert "- durable fact" not in first_task_text
    assert first_task_text.index("<pr_description>") < first_task_text.index(
        "first issue"
    )
    assert '<file path="/home/agent/MEMORY.md">' not in second_task_text
    assert "- durable fact" not in second_task_text
    assert "already injected earlier" not in second_task_text
    assert "The repository has advanced to a new commit." in second_task_text
    assert len(models) == 2
    first_system = models[0].seen_messages[0][0]["content"]
    second_system = models[1].seen_messages[0][0]["content"]
    assert '<file path="/home/agent/MEMORY.md">' in first_system
    assert "- durable fact" in first_system
    assert '<file path="/home/agent/MEMORY.md">' in second_system
    assert "- durable fact" in second_system


def test_mini_conversation_session_keeps_submitted_payload_out_of_live_context(
    tmp_path,
    monkeypatch,
) -> None:
    huge_patch = "diff --git a/huge b/huge\n" + ("x" * 20_000)
    models: list[FakeModel] = []

    class HugeSubmitAgent(FakeDefaultAgent):
        def get_observation(self, _response):
            raise Submitted(huge_patch)

    def fake_get_model(*, config):
        model = FakeModel(
            "THOUGHT: submit\n```bash\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff --cached\n```"
        )
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": HugeSubmitAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": "TASK={{task}}",
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=tmp_path / "home",
        context_window=1_000_000,
    )
    task = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=task,
        artifacts_dir=tmp_path / "i1",
    )

    traj = json.loads((tmp_path / "i1" / "i1.mini.traj.json").read_text())
    saved = json.loads((tmp_path / "state" / "messages.json").read_text())
    assert traj["exit_status"] == "Submitted"
    assert traj["result"] == huge_patch
    assert "diff --git a/huge b/huge" not in json.dumps(traj["messages"])
    assert "diff --git a/huge b/huge" not in json.dumps(saved["messages"])


def test_mini_conversation_session_writes_compaction_audit(
    tmp_path,
    monkeypatch,
) -> None:
    models: list[FakeModel] = []

    def fake_get_model(*, config):
        content = (
            VALID_COMPACTION_SUMMARY
            if len(models) == 2
            else "THOUGHT: done\n```bash\necho done\n```"
        )
        model = FakeModel(content)
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": "TASK={{task}}",
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    home = tmp_path / "home"
    home.mkdir()
    (home / "MEMORY.md").write_text("# memory\nold fact\n")
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=home,
        context_window=2_000,
        reserve_tokens=1_500,
        keep_recent_messages=1,
    )
    first = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="first issue " + ("x" * 3_000),
        dockerhub_tag="tag",
    )
    second = Task(
        instance_id="i2",
        repo="repo/repo",
        base_commit="b",
        problem_statement="second issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=first,
        artifacts_dir=tmp_path / "i1",
    )
    (home / "MEMORY.md").write_text("# memory\nnew fact\n")
    session.run_task(
        runner=FakeSequentialRunner(),
        task=second,
        artifacts_dir=tmp_path / "i2",
    )

    compaction_lines = (tmp_path / "state" / "compactions.jsonl").read_text().splitlines()
    assert len(compaction_lines) == 1
    compaction_meta = json.loads(compaction_lines[0])
    audit_path = tmp_path / "state" / compaction_meta["audit_path"]
    audit = json.loads(audit_path.read_text())

    assert audit["compaction_id"] == compaction_meta["compaction_id"]
    assert audit["reason"] == "before_task"
    assert audit["method"] == "model_summary"
    assert audit["summary"] == VALID_COMPACTION_SUMMARY.strip()
    assert audit["summary_retry_count"] == 0
    assert audit["summary_validation_error"] is None
    assert "first issue" in json.dumps(audit["messages_before"])
    assert "first issue" in json.dumps(audit["messages_compacted"])
    assert "first issue" in audit["transcript_sent_to_compactor"]
    compaction_messages = models[2].seen_messages[0]
    assert compaction_messages[0]["role"] == "system"
    assert "Do NOT continue the conversation" in compaction_messages[0]["content"]
    assert "<conversation>" in compaction_messages[1]["content"]
    assert "<mini_sequential_compaction_summary>" in json.dumps(
        audit["messages_after"]
    )
    system_after = next(
        msg["content"] for msg in audit["messages_after"] if msg["role"] == "system"
    )
    assert "new fact" in system_after
    assert "old fact" not in system_after
    assert audit["estimated_tokens_after"] < audit["estimated_tokens_before"]
    # compaction output must stay provider-valid: no consecutive same-role turns
    roles_after = [m.get("role") for m in audit["messages_after"]]
    assert all(
        roles_after[i] != roles_after[i - 1] for i in range(1, len(roles_after))
    ), roles_after


def test_compact_completed_history_strictly_summarizes_before_next_task(
    tmp_path,
    monkeypatch,
) -> None:
    models: list[FakeModel] = []

    def fake_get_model(*, config):
        content = (
            VALID_COMPACTION_SUMMARY
            if len(models) == 1
            else "THOUGHT: done\n```bash\necho done\n```"
        )
        model = FakeModel(content)
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": "TASK={{task}}",
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    home = tmp_path / "home"
    home.mkdir()
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=home,
        context_window=1_000_000,
    )
    first = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="first issue",
        dockerhub_tag="tag",
    )
    second = Task(
        instance_id="i2",
        repo="repo/repo",
        base_commit="b",
        problem_statement="second issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=first,
        artifacts_dir=tmp_path / "i1",
    )
    assert session.compact_completed_history(reason="after_task") is True

    first_traj = json.loads((tmp_path / "i1" / "i1.mini.traj.json").read_text())
    saved_after_compaction = json.loads(
        (tmp_path / "state" / "messages.json").read_text()
    )
    compaction_meta = json.loads(
        (tmp_path / "state" / "compactions.jsonl").read_text().splitlines()[0]
    )
    audit = json.loads(
        (tmp_path / "state" / compaction_meta["audit_path"]).read_text()
    )

    assert "first issue" in json.dumps(first_traj["messages"])
    assert "first issue" not in json.dumps(saved_after_compaction["messages"])
    assert "<mini_sequential_compaction_summary>" in json.dumps(
        saved_after_compaction["messages"]
    )
    assert saved_after_compaction["messages"][-1] == {
        "role": "assistant",
        "content": "Ready for the next task.",
    }
    assert audit["reason"] == "after_task"
    assert audit["messages_kept"] == []
    assert audit["keep_recent_messages"] == 0
    assert audit["keep_recent_tokens"] == 0
    assert audit["min_recent_messages"] == 0

    session.run_task(
        runner=FakeSequentialRunner(),
        task=second,
        artifacts_dir=tmp_path / "i2",
    )

    second_query_messages = models[2].seen_messages[0]
    assert "first issue" not in json.dumps(second_query_messages)
    assert "second issue" in json.dumps(second_query_messages)
    assert "<mini_sequential_compaction_summary>" in json.dumps(second_query_messages)


def test_compaction_trigger_uses_20k_reserve(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": lambda *, config: FakeModel("unused"),
            "save_traj": lambda agent, path, **kwargs: path.write_text("{}"),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": "TASK={{task}}",
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=tmp_path / "home",
        context_window=200_000,
    )
    calls = []

    def fake_compact_now(agent, reason, force):
        calls.append(reason)
        return True

    session._compact_now = fake_compact_now
    agent = SimpleNamespace(messages=[{"role": "system", "content": "sys"}])

    session._last_observed_input_tokens = 170_000
    assert session._compact_if_needed(agent, reason="pre_query") is False
    assert calls == []

    session._last_observed_input_tokens = 181_000
    assert session._compact_if_needed(agent, reason="pre_query") is True
    assert calls == ["pre_query"]
    assert session.reserve_tokens == 20_000
    assert session.keep_recent_tokens == 20_000


def test_compaction_repairs_transcript_like_model_summary(
    tmp_path,
    monkeypatch,
) -> None:
    models: list[FakeModel] = []
    contents = [
        "THOUGHT: done\n```bash\necho done\n```",
        "THOUGHT: done\n```bash\necho done\n```",
        (
            "Let me inspect the file.\n\n```bash\ncat /app/file.py\n```\n\n"
            "[user]\n<returncode>0</returncode>"
        ),
        VALID_COMPACTION_SUMMARY,
    ]

    def fake_get_model(*, config):
        model = FakeModel(contents[min(len(models), len(contents) - 1)])
        models.append(model)
        return model

    monkeypatch.setattr(
        "swebench_seq.agent.driver.load_mini_imports",
        lambda: {
            "DefaultAgent": FakeDefaultAgent,
            "NonTerminatingException": NonTerminatingException,
            "TerminatingException": Submitted,
            "get_model": fake_get_model,
            "save_traj": lambda agent, path, **kwargs: path.write_text(
                json.dumps({"messages": agent.messages, **kwargs})
            ),
        },
    )
    monkeypatch.setattr(
        "swebench_seq.agent.config.load_config",
        lambda _imports: {
            "agent": {
                "system_template": "system",
                "instance_template": "TASK={{task}}",
                "step_limit": 10,
                "cost_limit": 0,
            },
            "environment": {"env": {}},
            "model": {},
        },
    )
    session = MiniConversationSession(
        options=MiniSweAgentOptions(model="fake-model"),
        state_dir=tmp_path / "state",
        agent_home_dir=tmp_path / "home",
        context_window=2_000,
        reserve_tokens=1_500,
        keep_recent_messages=1,
    )
    first = Task(
        instance_id="i1",
        repo="repo/repo",
        base_commit="a",
        problem_statement="first issue " + ("x" * 3_000),
        dockerhub_tag="tag",
    )
    second = Task(
        instance_id="i2",
        repo="repo/repo",
        base_commit="b",
        problem_statement="second issue",
        dockerhub_tag="tag",
    )

    session.run_task(
        runner=FakeSequentialRunner(),
        task=first,
        artifacts_dir=tmp_path / "i1",
    )
    session.run_task(
        runner=FakeSequentialRunner(),
        task=second,
        artifacts_dir=tmp_path / "i2",
    )

    compaction_meta = json.loads(
        (tmp_path / "state" / "compactions.jsonl").read_text().splitlines()[0]
    )
    audit = json.loads(
        (tmp_path / "state" / compaction_meta["audit_path"]).read_text()
    )
    assert audit["method"] == "model_summary"
    assert audit["summary_retry_count"] == 1
    assert "contains markdown fenced code block" in audit["summary_validation_error"]
    assert audit["summary"] == VALID_COMPACTION_SUMMARY.strip()
    assert validate_compaction_summary(audit["summary"]) is None
    summary_message = next(
        msg["content"]
        for msg in audit["messages_after"]
        if "<mini_sequential_compaction_summary>" in msg["content"]
    )
    assert "```bash" not in summary_message
    assert "[user]" not in summary_message.lower()
    repair_messages = models[3].seen_messages[0]
    assert "previous compaction summary failed validation" in repair_messages[1][
        "content"
    ].lower()


def test_history_compaction_messages_use_previous_summary_for_updates() -> None:
    messages = history_compaction_messages(
        "[user]\nnew work",
        previous_summary=VALID_COMPACTION_SUMMARY,
    )

    assert messages[0]["role"] == "system"
    assert "ONLY output the structured summary" in messages[0]["content"]
    assert "<previous-summary>" in messages[1]["content"]
    assert "NEW conversation messages" in messages[1]["content"]


def test_bound_memory_content_passes_through_small() -> None:
    text = "short memory\n"
    assert _bound_memory_content(text, max_chars=1000) == text


def test_bound_memory_content_truncates_large() -> None:
    bounded = _bound_memory_content("y" * 50_000, max_chars=1_000)
    assert len(bounded) < 1_500
    assert "truncated by swebench-seq" in bounded


def test_render_agent_home_memory_context_bounds_huge_memory(tmp_path) -> None:
    (tmp_path / "MEMORY.md").write_text("z" * 100_000)
    rendered = render_agent_home_memory_context(tmp_path)
    assert "truncated by swebench-seq" in rendered
    # bounded well below the raw 100k, even with the wrapping tags
    assert len(rendered) < 20_000


def test_recent_keep_start_index_token_bounds_large_messages() -> None:
    # 10 large messages (~2500 tokens each); a 6000-token budget keeps only a
    # couple, well under the message-count cap of 8.
    messages = [{"role": "system", "content": "s"}]
    messages += [{"role": "user", "content": "x" * 10_000} for _ in range(10)]
    start = recent_keep_start_index(messages, start=1, max_messages=8, max_tokens=6_000)
    assert 1 <= len(messages) - start <= 3  # token budget bites before the count cap


def test_recent_keep_start_index_count_bounds_small_messages() -> None:
    messages = [{"role": "system", "content": "s"}]
    messages += [{"role": "user", "content": "ok"} for _ in range(20)]
    start = recent_keep_start_index(
        messages, start=1, max_messages=5, max_tokens=1_000_000
    )
    assert len(messages) - start == 5  # count cap bites before the token budget


def test_recent_keep_start_index_always_keeps_at_least_one() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "x" * 100_000},
    ]
    start = recent_keep_start_index(messages, start=1, max_messages=5, max_tokens=10)
    assert len(messages) - start == 1  # one oversized message is still kept


def test_coalesce_merges_consecutive_same_role() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "summary"},
        {"role": "user", "content": "task"},  # adjacent user (the risky case)
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "obs"},
    ]
    out = coalesce_consecutive_messages(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "user"]
    assert all(roles[i] != roles[i - 1] for i in range(1, len(roles)))  # alternates
    assert out[1]["content"] == "summary\n\ntask"  # content preserved


def test_coalesce_is_noop_when_already_alternating() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert coalesce_consecutive_messages(msgs) == msgs


def testlocate_current_task_index_by_identity_and_content() -> None:
    target = {"role": "user", "content": "TASK-XYZ"}
    msgs = [{"role": "system", "content": "s"}, target, {"role": "assistant", "content": "a"}]
    assert locate_current_task_index(msgs, target) == 1
    # after coalesce the object changes, but its content is still contained
    merged = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "summary\n\nTASK-XYZ"},
    ]
    assert locate_current_task_index(merged, target) == 1
    assert locate_current_task_index(msgs, None) is None


def test_split_compaction_token_budget_compacts_large_recent_messages() -> None:
    # System + 6 large messages. A tight token budget makes the keep window
    # small even though the message-count cap is generous, so most get compacted.
    messages: list[dict] = [{"role": "system", "content": "sys"}]
    messages += [
        {"role": "user", "content": f"msg{i} " + "x" * 8_000} for i in range(6)
    ]
    split = split_messages_for_compaction(
        messages,
        current_task_start_index=None,
        keep_recent_messages=24,
        keep_recent_tokens=3_000,
    )
    assert split["system"] is messages[0]
    assert len(split["compact"]) >= 4  # most of the large messages compacted
    assert len(split["keep"]) <= 2


def test_split_compaction_can_disable_recent_keep_tail() -> None:
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "THOUGHT\n```bash\ncat a.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>a</output>"},
    ]

    split = split_messages_for_compaction(
        messages,
        current_task_start_index=None,
        keep_recent_messages=0,
        keep_recent_tokens=0,
        min_recent_messages=0,
    )

    assert split["compact"] == messages[1:]
    assert split["history_compact"] == messages[1:]
    assert split["turn_prefix"] == []
    assert split["keep"] == []


def test_split_compaction_does_not_keep_bare_mini_observation() -> None:
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "TASK"},
        {"role": "assistant", "content": "THOUGHT\n```bash\ncat a.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>a</output>"},
        {"role": "assistant", "content": "THOUGHT\n```bash\ncat b.py\n```"},
        {"role": "user", "content": "<returncode>0</returncode>\n<output>b</output>"},
    ]

    split = split_messages_for_compaction(
        messages,
        current_task_start_index=1,
        keep_recent_messages=1,
        keep_recent_tokens=1_000,
    )

    assert split["keep"] == [messages[1], messages[4], messages[5]]
    assert split["turn_prefix"] == [messages[2], messages[3]]
    assert split["history_compact"] == []
