"""Tests for the OpenAI Responses model + config override/merge helpers."""

from __future__ import annotations

import json

from swebench_seq.agent.config import apply_override, is_responses_model_class
from swebench_seq.agent.driver import extract_mini_usage
from swebench_seq.agent.providers import wrap_model_for_provider
from swebench_seq.agent.responses import (
    OpenAIResponsesModel,
    _messages_to_responses_input,
    _render_bash_tool_call,
)

# ---- config override machinery ------------------------------------------------


def test_apply_override_sets_dotted_key_parsed_as_yaml_scalar() -> None:
    cfg: dict = {"model": {"model_kwargs": {"reasoning_effort": "high"}}}
    apply_override(cfg, "model.model_kwargs.reasoning_effort", "low")
    apply_override(cfg, "agent.step_limit", "300")
    assert cfg["model"]["model_kwargs"]["reasoning_effort"] == "low"
    assert cfg["agent"]["step_limit"] == 300  # int, not the string "300"


def test_apply_override_creates_nested_path() -> None:
    cfg: dict = {}
    apply_override(cfg, "model.model_kwargs.reasoning_effort", "medium")
    assert cfg == {"model": {"model_kwargs": {"reasoning_effort": "medium"}}}


def test_is_responses_model_class() -> None:
    assert is_responses_model_class(
        "swebench_seq.agent.responses.OpenAIResponsesModel"
    )
    assert not is_responses_model_class("litellm")
    assert not is_responses_model_class(None)


# ---- message conversion -------------------------------------------------------


def test_messages_to_input_splits_leading_system_into_instructions() -> None:
    instructions, items = _messages_to_responses_input(
        [
            {"role": "system", "content": "sys-a"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    )
    assert instructions == "sys-a"
    assert items == [
        {"role": "user", "content": "u1"},
        # An assistant turn with no replayable native items must be sent as an
        # `output_text` block; a bare string becomes `input_text`, which the
        # Responses API rejects on assistant items.
        {"role": "assistant", "content": [{"type": "output_text", "text": "a1"}]},
        {"role": "user", "content": "u2"},
    ]


def test_unreplayable_assistant_turn_uses_output_text_not_input_text() -> None:
    """A truncated turn (reasoning item, no message) must not poison the history.

    Regression test for tasks dying mid-run with:
      Invalid value: 'input_text'. Supported values are: 'output_text' and 'refusal'.
    Some compatible gateways wrap that 400 as a 500, causing pointless retries
    with an empty patch that is then scored False.
    """
    truncated = {
        "role": "assistant",
        "content": "partial answer",
        "extra": {
            "response": {
                "status": "incomplete",
                # reasoning only -- max_output_tokens ran out before any message
                "output": [{"id": "rs_1", "type": "reasoning", "summary": []}],
            }
        },
    }
    _, items = _messages_to_responses_input([truncated, {"role": "user", "content": "next"}])
    assert items == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "partial answer"}]},
        {"role": "user", "content": "next"},
    ]
    assert not any(
        block.get("type") == "input_text"
        for item in items
        if isinstance(item.get("content"), list)
        for block in item["content"]
    )


def test_empty_unreplayable_assistant_turn_is_dropped() -> None:
    """A fully-truncated turn carries no content, so it is omitted entirely."""
    empty = {
        "role": "assistant",
        "content": "",
        "extra": {"response": {"status": "incomplete", "output": [{"type": "reasoning"}]}},
    }
    _, items = _messages_to_responses_input([empty, {"role": "user", "content": "go"}])
    assert items == [{"role": "user", "content": "go"}]


def test_messages_to_input_maps_unknown_roles_to_user() -> None:
    _, items = _messages_to_responses_input([{"role": "tool", "content": "obs"}])
    assert items == [{"role": "user", "content": "obs"}]


def test_messages_to_input_replays_encrypted_reasoning_items() -> None:
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "encrypted_content": "opaque",
        "status": None,
        "summary": [],
    }
    assistant_message = {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "42"}],
    }

    _, items = _messages_to_responses_input(
        [
            {"role": "user", "content": "what is 17+25?"},
            {
                "role": "assistant",
                "content": "42",
                "extra": {"response": {"output": [reasoning, assistant_message]}},
            },
            {"role": "user", "content": "repeat it"},
        ]
    )

    assert items == [
        {"role": "user", "content": "what is 17+25?"},
        {
            "id": "rs_1",
            "type": "reasoning",
            "encrypted_content": "opaque",
            "summary": [],
        },
        assistant_message,
        {"role": "user", "content": "repeat it"},
    ]


def test_replay_drops_reasoning_item_trailing_the_message() -> None:
    # Some compatible models emit a reasoning item AFTER the
    # final assistant message: [reasoning, message, reasoning]. Replaying the
    # trailing reasoning before the next user turn yields a 400 "Invalid
    # conversation structure: reasoning items must be followed by an assistant
    # message...". The replayed block must end with the message.
    pre = {"id": "rs_a", "type": "reasoning", "encrypted_content": "opaque", "summary": []}
    msg = {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ls"}],
    }
    trailing = {"id": "rs_b", "type": "reasoning", "encrypted_content": "more", "summary": []}

    _, items = _messages_to_responses_input(
        [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "ls",
                "extra": {"response": {"output": [pre, msg, trailing]}},
            },
            {"role": "user", "content": "now cat a.txt"},
        ]
    )

    # pre-message reasoning kept, message kept, trailing reasoning dropped; the
    # assistant block ends with the message right before the next user item.
    assert items == [
        {"role": "user", "content": "list files"},
        {"id": "rs_a", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
        msg,
        {"role": "user", "content": "now cat a.txt"},
    ]


def test_messages_to_input_drops_unencrypted_reasoning_items() -> None:
    assistant_message = {
        "id": "msg_1",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "42"}],
    }

    _, items = _messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "42",
                "extra": {
                    "response": {
                        "output": [
                            {
                                "id": "rs_1",
                                "type": "reasoning",
                                "encrypted_content": None,
                                "summary": [],
                            },
                            assistant_message,
                        ]
                    }
                },
            }
        ]
    )

    assert items == [assistant_message]


# ---- OpenAIResponsesModel -----------------------------------------------------


class _FakeResponse:
    output_text = "```bash\necho hi\n```"

    def model_dump(self) -> dict:
        return {
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "reasoning": {"effort": "high"},
        }


class _FakeResponsesAPI:
    def __init__(self) -> None:
        self.captured: dict = {}

    def create(self, **kwargs):
        self.captured = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponsesAPI()


def test_responses_model_query_applies_reasoning_and_parses_output() -> None:
    model = OpenAIResponsesModel(
        model_name="gpt-5-x",
        model_kwargs={
            "api_key": "k",
            "api_base": "http://x/openai/v1",
            "extra_headers": {"x-sticky-routing-key": "repo-key"},
            "prompt_cache_key": "repo-key",
            "reasoning_effort": "high",
            "max_output_tokens": 1234,
        },
    )
    fake = _FakeClient()
    model._client = fake  # inject

    res = model.query(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "go"}]
    )

    assert res["content"] == "```bash\necho hi\n```"
    assert model.n_calls == 1
    cap = fake.responses.captured
    assert cap["model"] == "gpt-5-x"
    assert cap["reasoning"] == {"effort": "high"}
    assert cap["max_output_tokens"] == 1234
    assert cap["instructions"] == "s"
    assert cap["input"] == [{"role": "user", "content": "go"}]
    assert cap["include"] == ["reasoning.encrypted_content"]
    assert cap["extra_headers"] == {"x-sticky-routing-key": "repo-key"}
    assert cap["prompt_cache_key"] == "repo-key"
    assert cap["store"] is False
    usage = extract_mini_usage(res)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5


def test_responses_model_forwards_deadline_timeout(monkeypatch) -> None:
    monkeypatch.setattr("swebench_seq.agent.responses.time.monotonic", lambda: 10.0)
    model = OpenAIResponsesModel(model_name="gpt-5-x", model_kwargs={})
    fake = _FakeClient()
    model._client = fake

    model.query([{"role": "user", "content": "go"}], timeout=12.0)

    assert fake.responses.captured["timeout"] == 12.0


def test_responses_model_drops_litellm_only_kwargs() -> None:
    model = OpenAIResponsesModel(
        model_name="gpt-5-x",
        model_kwargs={
            "drop_params": True,
            "temperature": 0.5,
            "custom_llm_provider": "openai",
            "allowed_openai_params": ["thinking"],
            "reasoning_effort": "low",
        },
    )
    assert model.reasoning_effort == "low"
    for key in ("drop_params", "temperature", "custom_llm_provider", "allowed_openai_params"):
        assert key not in model._extra_request


def test_wrap_model_for_provider_skips_responses_model() -> None:
    model = OpenAIResponsesModel(model_name="gpt-5-x", model_kwargs={})
    assert wrap_model_for_provider(model, "openai") is model


def test_tls_verify_default_uses_no_custom_http_client(monkeypatch) -> None:
    import openai

    captured: dict = {}

    class _Cli:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openai, "OpenAI", _Cli)
    model = OpenAIResponsesModel(model_name="x", model_kwargs={"api_key": "k"})
    assert model.tls_verify is True
    model._client_or_create()
    assert "http_client" not in captured


def test_tls_verify_false_builds_insecure_http_client(monkeypatch, caplog) -> None:
    import httpx
    import openai

    captured: dict = {}

    class _Cli:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(openai, "OpenAI", _Cli)
    model = OpenAIResponsesModel(
        model_name="tool-capable-reasoning-model",
        model_kwargs={
            "api_key": "test",
            "api_base": "https://host:20009/v1",
            "tls_verify": False,
        },
    )
    assert model.tls_verify is False
    model._client_or_create()
    assert isinstance(captured.get("http_client"), httpx.Client)
    assert "TLS certificate verification is disabled" in caplog.text
    # tls_verify is consumed, not forwarded to responses.create
    assert "tls_verify" not in model._extra_request


def test_replay_keeps_plaintext_reasoning_when_enabled() -> None:
    # vLLM-style reasoning item: plaintext content, NO encrypted_content.
    assistant = {
        "role": "assistant",
        "content": "```bash\nls\n```",
        "extra": {
            "response": {
                "output": [
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "think.."}],
                        "encrypted_content": None,
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "```bash\nls\n```"}],
                    },
                ]
            }
        },
    }
    msgs = [{"role": "user", "content": "go"}, assistant, {"role": "user", "content": "next"}]

    # Default (encrypted-only) replay drops the plaintext reasoning item.
    _, items_enc = _messages_to_responses_input(msgs, replay_encrypted_reasoning=True)
    assert not any(it.get("type") == "reasoning" for it in items_enc)

    # Plaintext replay carries it forward.
    _, items_plain = _messages_to_responses_input(
        msgs,
        replay_encrypted_reasoning=False,
        replay_plaintext_reasoning=True,
    )
    reasoning = [it for it in items_plain if it.get("type") == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["content"] == [{"type": "reasoning_text", "text": "think.."}]
    assert "encrypted_content" not in reasoning[0]  # None dropped
    assert any(it.get("type") == "message" for it in items_plain)


def test_responses_model_passes_extra_body_and_skips_replay() -> None:
    # A compatible gateway configuration: medium effort, no encrypted replay
    # (vLLM doesn't emit it), and vLLM-extension params via extra_body.
    model = OpenAIResponsesModel(
        model_name="tool-capable-reasoning-model",
        model_kwargs={
            "api_key": "test",
            "reasoning_effort": "medium",
            "replay_encrypted_reasoning": False,
            "extra_body": {"skip_special_tokens": False},
        },
    )
    fake = _FakeClient()
    model._client = fake  # inject
    model.query([{"role": "user", "content": "go"}])
    cap = fake.responses.captured
    assert cap["reasoning"] == {"effort": "medium"}
    assert cap["extra_body"] == {"skip_special_tokens": False}
    assert "include" not in cap  # replay off -> no encrypted-content include


# ---- tool-protocol (bash_tool) mode -------------------------------------------
#
# Some tool-trained checkpoints route shell actions to a dedicated tool channel.
# A server-side tool parser may interpret that channel as a tool call whether or
# not the request declared tools. These tests pin the translation layer that
# keeps mini's text agent loop unchanged.


def _bash_call_response(command: str, call_id: str = "call_1") -> dict:
    return {
        "output": [
            {"id": "rs_1", "type": "reasoning", "content": [{"text": "thinking"}]},
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": call_id,
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            },
        ]
    }


def test_render_bash_tool_call_produces_a_parseable_bash_block() -> None:
    rendered = _render_bash_tool_call(_bash_call_response("ls -la /app"))
    assert rendered is not None
    assert "```bash\nls -la /app\n```" in rendered
    assert rendered.startswith("THOUGHT:")


def test_render_bash_tool_call_returns_none_without_a_bash_call() -> None:
    plain = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ]
    }
    assert _render_bash_tool_call(plain) is None


def test_tool_protocol_replays_function_call_and_pairs_the_observation() -> None:
    """The observation must come back as function_call_output with the call_id.

    Replaying it as a plain user message works for one or two turns and then the
    model silently stops emitting tool calls, which is exactly the failure this
    pairing exists to prevent.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "THOUGHT: x\n\n```bash\nls\n```",
            "extra": {"response": _bash_call_response("ls", call_id="call_42")},
        },
        {"role": "user", "content": "<returncode>0</returncode>"},
    ]
    instructions, items = _messages_to_responses_input(
        messages, replay_plaintext_reasoning=True, tool_protocol=True
    )
    assert instructions == "sys"
    assert items[0] == {"role": "user", "content": "task"}
    assert items[1]["type"] == "reasoning"
    assert items[2]["type"] == "function_call"
    assert items[3] == {
        "type": "function_call_output",
        "call_id": "call_42",
        "output": "<returncode>0</returncode>",
    }


def test_tool_protocol_disabled_keeps_user_observation_and_drops_function_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "text",
            "extra": {"response": _bash_call_response("ls")},
        },
        {"role": "user", "content": "obs"},
    ]
    _, items = _messages_to_responses_input(
        messages, replay_plaintext_reasoning=True, tool_protocol=False
    )
    # No assistant message item and function_call replay is off, so the turn falls
    # back to plain text and the observation stays a user message.
    assert not any(i.get("type") == "function_call" for i in items)
    assert items[-1] == {"role": "user", "content": "obs"}


def test_bash_tool_declares_the_tool_on_the_request() -> None:
    captured: dict = {}

    class _API:
        def create(self, **request):
            captured.update(request)
            return _FakeResponse()

    class _Client:
        responses = _API()

    model = OpenAIResponsesModel(
        model_name="m",
        model_kwargs={"api_key": "k", "bash_tool": True},
    )
    model._client = _Client()
    model.query([{"role": "user", "content": "hi"}])
    assert [t["name"] for t in captured["tools"]] == ["bash"]
    assert captured["tools"][0]["parameters"]["required"] == ["command"]


def test_bash_tool_off_by_default() -> None:
    captured: dict = {}

    class _API:
        def create(self, **request):
            captured.update(request)
            return _FakeResponse()

    class _Client:
        responses = _API()

    model = OpenAIResponsesModel(model_name="m", model_kwargs={"api_key": "k"})
    model._client = _Client()
    model.query([{"role": "user", "content": "hi"}])
    assert "tools" not in captured
