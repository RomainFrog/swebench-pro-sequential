"""OpenAI Responses API model for mini-swe-agent.

mini-swe-agent's default ``LitellmModel`` drives ``/chat/completions``. The
gpt-5 reasoning models are better driven through the OpenAI **Responses API**
(``/responses``): native ``reasoning.effort`` control and reasoning-token
accounting. This class implements mini's model interface (``query`` /
``n_calls`` / ``cost`` / ``get_template_vars``) on top of the Responses API via
the OpenAI SDK. A custom Responses-compatible endpoint can be selected with
``api_base``.

It is selected from a mini config file via ``model_class``::

    model:
      model_name: gpt-5
      model_class: swebench_seq.agent.responses.OpenAIResponsesModel
      model_kwargs:
        reasoning_effort: high

``api_key`` / ``api_base`` are injected at runtime (secrets / env), not stored
in the config file.

Reasoning persistence uses encrypted reasoning replay when the provider returns
``reasoning.encrypted_content``. The model asks for those opaque blobs and
replays prior response ``output`` items from mini's saved assistant messages.
It does not rely on ``previous_response_id`` chaining, which only works for
stored responses and is less robust for this stateless per-turn agent loop.

Tool-protocol mode (``bash_tool: true``)
----------------------------------------
Some Responses-compatible models emit structured tool calls rather than a
Markdown shell block. Setting ``bash_tool: true`` declares a single ``bash``
function, renders the returned call back into the ```` ```bash ```` block
mini's agent expects, and replays history natively as ``reasoning`` +
``function_call`` + ``function_call_output``. The agent loop is unchanged; only
the wire format differs.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_any,
    stop_before_delay,
    wait_exponential,
)

logger = logging.getLogger("responses_model")

# Keys we consume from ``model_kwargs`` (everything else is forwarded to the
# Responses API request body verbatim, after dropping the litellm-only keys
# below which the Responses API does not accept).
_LITELLM_ONLY_KEYS = frozenset(
    {
        "drop_params",
        "temperature",
        "custom_llm_provider",
        "allowed_openai_params",
        "thinking",
        "output_config",
        "set_cache_control",
        "litellm_model_registry",
    }
)

_DEFAULT_MAX_OUTPUT_TOKENS = 16_000
_DEFAULT_TIMEOUT_SECONDS = 300
_DEFAULT_REASONING_EFFORT = "medium"

# Tool declaration used by ``bash_tool`` mode. See the module docstring section
# "Tool-protocol mode" for why this exists.
_BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "bash",
    "description": "Run a bash command in the shell and return its output.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to run"}
        },
        "required": ["command"],
    },
}


@dataclass
class OpenAIResponsesModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    # Accepted for parity with mini's LitellmModel config; provider-specific
    # billing means this adapter does not compute cost locally.
    cost_tracking: str = "ignore_errors"


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient transport / rate-limit / server errors only."""
    name = type(exc).__name__
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "APIStatusError",
    }:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return bool(status == 429 or status >= 500)
        return True
    return False


class OpenAIResponsesModel:
    """mini-swe-agent model backed by the OpenAI Responses API."""

    # Lets the driver skip the chat-message sanitizing/replay wrappers, which
    # only apply to litellm-backed models.
    _is_responses_model = True

    def __init__(
        self,
        *,
        config_class: type = OpenAIResponsesModelConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        mk = dict(self.config.model_kwargs)
        self.api_key = mk.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        self.base_url = mk.pop("api_base", None) or mk.pop("base_url", None)
        self.reasoning_effort = mk.pop("reasoning_effort", _DEFAULT_REASONING_EFFORT)
        # "auto" / "concise" / "detailed" → request a visible reasoning summary
        # (useful for trace analysis); None → no summary.
        self.reasoning_summary = mk.pop("reasoning_summary", None)
        self.replay_encrypted_reasoning = bool(
            mk.pop("replay_encrypted_reasoning", True)
        )
        # Some self-hosted reasoning models emit chain-of-thought as plaintext
        # reasoning-item ``content`` rather than an opaque ``encrypted_content``
        # blob. This opt-in round-trips those items when the endpoint supports it.
        self.replay_plaintext_reasoning = bool(
            mk.pop("replay_plaintext_reasoning", False)
        )
        self.max_output_tokens = (
            mk.pop("max_output_tokens", None) or _DEFAULT_MAX_OUTPUT_TOKENS
        )
        self.timeout = mk.pop("timeout", None) or _DEFAULT_TIMEOUT_SECONDS
        # Self-hosted HTTPS endpoints may use a private CA. ``tls_verify: false``
        # is available for isolated test environments; verification stays on by
        # default.
        self.tls_verify = bool(mk.pop("tls_verify", True))
        # Tool-protocol mode. Some models route shell actions through structured
        # function calls instead of Markdown fences. Declaring the tool makes
        # the model emit proper JSON arguments, which
        # the parser accepts. We keep mini's text agent loop untouched by
        # rendering the returned call back into a ```bash block.
        self.bash_tool = bool(mk.pop("bash_tool", False))
        self.max_attempts = int(mk.pop("max_retries", 6))
        extra_headers = mk.pop("extra_headers", None) or {}
        if not isinstance(extra_headers, dict):
            raise TypeError("model_kwargs.extra_headers must be a mapping")
        self.extra_headers = dict(extra_headers)
        for key in _LITELLM_ONLY_KEYS:
            mk.pop(key, None)
        # Remaining keys are passed through to responses.create (e.g. top_p,
        # service_tier). Keep this conservative — unknown params raise 400.
        self._extra_request: dict[str, Any] = mk

        self._client: Any | None = None

    def _client_or_create(self) -> Any:
        if self._client is None:
            import openai

            client_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.base_url,
                "timeout": self.timeout,
                "max_retries": 0,  # we own retries via tenacity
            }
            if not self.tls_verify:
                # Custom httpx client with TLS verification disabled for
                # self-signed sidecar certs (the SDK has no env/flag for this).
                import httpx

                logger.warning(
                    "TLS certificate verification is disabled for the model endpoint"
                )
                client_kwargs["http_client"] = httpx.Client(
                    verify=False, timeout=self.timeout
                )
            self._client = openai.OpenAI(**client_kwargs)
        return self._client

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        client = self._client_or_create()
        instructions, input_items = _messages_to_responses_input(
            messages,
            replay_encrypted_reasoning=self.replay_encrypted_reasoning,
            replay_plaintext_reasoning=self.replay_plaintext_reasoning,
            tool_protocol=self.bash_tool,
        )
        query_extra_headers = kwargs.pop("extra_headers", None) or {}
        if not isinstance(query_extra_headers, dict):
            raise TypeError("extra_headers must be a mapping")
        request: dict[str, Any] = {
            "model": self.config.model_name,
            "input": input_items,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            **self._extra_request,
            **kwargs,
        }
        if instructions:
            request["instructions"] = instructions
        if self.bash_tool:
            request.setdefault("tools", [copy.deepcopy(_BASH_TOOL)])
        reasoning: dict[str, Any] = {}
        if self.reasoning_effort and self.reasoning_effort != "none":
            reasoning["effort"] = self.reasoning_effort
        if self.reasoning_summary:
            reasoning["summary"] = self.reasoning_summary
        if reasoning:
            request["reasoning"] = reasoning
        if self.replay_encrypted_reasoning:
            include = request.get("include")
            if include is None:
                include_items: list[str] = []
            elif isinstance(include, str):
                include_items = [include]
            elif isinstance(include, list):
                include_items = [item for item in include if isinstance(item, str)]
            else:
                raise TypeError("model_kwargs.include must be a string or list")
            if "reasoning.encrypted_content" not in include_items:
                include_items.append("reasoning.encrypted_content")
            request["include"] = include_items
        if self.extra_headers or query_extra_headers:
            request["extra_headers"] = {
                **self.extra_headers,
                **query_extra_headers,
            }

        response = self._create_with_retry(client, request)
        self.n_calls += 1
        dumped = response.model_dump()
        content = response.output_text or ""
        if self.bash_tool:
            # Render the tool call back into the ```bash block mini's agent
            # parses, so nothing downstream of this method has to change.
            rendered = _render_bash_tool_call(dumped)
            if rendered is not None:
                content = rendered
        return {
            "content": content,
            "extra": {"response": dumped},
        }

    def _create_with_retry(self, client: Any, request: dict[str, Any]) -> Any:
        requested_timeout = request.get("timeout")
        if (
            isinstance(requested_timeout, (int, float))
            and not isinstance(requested_timeout, bool)
            and requested_timeout > 0
        ):
            has_budget = True
            budget = float(requested_timeout)
        else:
            has_budget = False
            budget = float("inf")
        deadline = time.monotonic() + budget

        @retry(
            stop=stop_any(
                stop_after_attempt(self.max_attempts),
                stop_before_delay(budget),
            ),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _do() -> Any:
            attempt_request = request
            if has_budget:
                attempt_request = {
                    **request,
                    "timeout": max(0.001, deadline - time.monotonic()),
                }
            return client.responses.create(**attempt_request)

        return _do()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }


def _render_bash_tool_call(raw_response: dict[str, Any]) -> str | None:
    """Render a ``bash`` function call as the ```bash block mini's agent parses.

    Returns None when the response carries no bash call, so the caller can fall
    back to ``output_text`` (e.g. a plain final message).
    """
    output = raw_response.get("output")
    if not isinstance(output, list):
        return None
    command: str | None = None
    thought_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == "bash":
            args = item.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = None
            if isinstance(args, dict) and isinstance(args.get("command"), str):
                command = args["command"]
        elif item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content") or []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    thought_parts.append(block["text"])
    if command is None:
        return None
    thought = "\n".join(t for t in thought_parts if t.strip()).strip()
    if not thought:
        thought = "Running the next command."
    return f"THOUGHT: {thought}\n\n```bash\n{command}\n```"


def _messages_to_responses_input(
    messages: list[dict[str, Any]],
    *,
    replay_encrypted_reasoning: bool = True,
    replay_plaintext_reasoning: bool = False,
    tool_protocol: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert mini's chat-style messages into Responses API ``input``.

    Leading system messages are concatenated into the ``instructions`` field;
    the remaining user/assistant messages become input items. When assistant
    messages carry a prior Responses API payload with encrypted reasoning, replay
    that provider-native output item sequence instead of flattening the turn to
    plaintext. mini always uses string ``content`` for fallback messages.
    """
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    # In tool-protocol mode the observation that follows a function_call must be
    # returned as a function_call_output carrying the matching call_id, not as a
    # user message. Replaying it as user text works for one or two turns and then
    # the model silently stops emitting tool calls, so this pairing is required.
    pending_call_id: str | None = None
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not isinstance(content, str):
            # mini uses string content; be defensive for tool/observation dicts.
            content = str(content)
        if role in ("system", "developer") and not input_items:
            instructions_parts.append(content)
            continue
        # Map any non-user/assistant roles to user so the API accepts them.
        api_role = role if role in ("user", "assistant") else "user"
        if api_role == "assistant":
            if replay_encrypted_reasoning or replay_plaintext_reasoning:
                replay_items = _responses_output_items_for_replay(
                    message,
                    keep_encrypted=replay_encrypted_reasoning,
                    keep_plaintext=replay_plaintext_reasoning,
                    allow_function_call=tool_protocol,
                )
                if replay_items:
                    input_items.extend(replay_items)
                    pending_call_id = None
                    if tool_protocol:
                        for item in reversed(replay_items):
                            if item.get("type") == "function_call":
                                call_id = item.get("call_id")
                                if isinstance(call_id, str):
                                    pending_call_id = call_id
                                break
                    continue
            # Fallback for an assistant turn we could NOT replay natively -- most
            # often a truncated response (status=incomplete: a reasoning item and
            # no assistant message, because max_output_tokens ran out mid-
            # reasoning).
            #
            # A bare string here is encoded by the SDK as an `input_text` content
            # block, which the Responses API rejects on an assistant item:
            #   Invalid value: 'input_text'. Supported values are: 'output_text'
            #   and 'refusal'.
            # Some gateways wrap that 400 as a retryable server error. The bad
            # turn would then remain in history and poison every later call. Emit
            # the correct assistant content block and drop empty turns instead.
            pending_call_id = None
            if not content:
                continue
            input_items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            )
            continue
        if pending_call_id is not None:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": pending_call_id,
                    "output": content,
                }
            )
            pending_call_id = None
            continue
        input_items.append({"role": api_role, "content": content})
    return "\n\n".join(p for p in instructions_parts if p), input_items


def _responses_output_items_for_replay(
    message: dict[str, Any],
    *,
    keep_encrypted: bool = True,
    keep_plaintext: bool = False,
    allow_function_call: bool = False,
) -> list[dict[str, Any]]:
    """Return provider-native Responses output items safe to replay as input.

    OpenAI Responses accepts prior response ``output`` items directly in a later
    ``input`` array. We always keep the assistant message item. For reasoning:

    - ``keep_encrypted`` keeps items carrying an opaque ``encrypted_content`` blob.
    - ``keep_plaintext`` keeps items whose chain-of-thought lives in plaintext
      ``content`` (some self-hosted reasoning models never populate
      ``encrypted_content``).

    A reasoning item with neither cannot carry hidden reasoning forward.
    """

    raw_response = message.get("extra", {}).get("response", {})
    if not isinstance(raw_response, dict):
        return []
    output = raw_response.get("output")
    if not isinstance(output, list):
        return []

    # The Responses API requires every reasoning item to be followed by an
    # assistant message / function_call before any user/system/developer message.
    # Some compatible models emit a trailing reasoning item
    # *after* the final assistant message; replaying it as the last item before the
    # next user turn yields a 400 "Invalid conversation structure". So we replay
    # only up to and including the last assistant message and drop anything after
    # it — guaranteeing the replayed block ends with the message.
    # In tool-protocol mode a turn usually ends in a function_call and carries no
    # assistant message at all, so a function_call is an equally valid terminator.
    last_msg_idx = -1
    for i, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            last_msg_idx = i
        elif allow_function_call and item.get("type") == "function_call":
            last_msg_idx = i
    if last_msg_idx < 0:
        return []

    replay_items: list[dict[str, Any]] = []
    for item in output[: last_msg_idx + 1]:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            has_encrypted = isinstance(encrypted, str) and bool(encrypted)
            has_plaintext = bool(item.get("content"))
            if (keep_encrypted and has_encrypted) or (
                keep_plaintext and has_plaintext
            ):
                replay_items.append(_drop_none_values(copy.deepcopy(item)))
        elif item_type == "message" and item.get("role") == "assistant":
            replay_items.append(_drop_none_values(copy.deepcopy(item)))
        elif allow_function_call and item_type == "function_call":
            replay_items.append(_drop_none_values(copy.deepcopy(item)))

    return replay_items


def _drop_none_values(value: Any) -> Any:
    """Drop null fields that OpenAI's Python SDK includes in ``model_dump()``.

    Responses-compatible endpoints accept raw output items for replay, but some
    reject null-only fields such as ``input[1].status`` on reasoning items. Raw
    HTTP responses omit those fields, so mirror that shape.
    """

    if isinstance(value, dict):
        return {
            key: _drop_none_values(inner)
            for key, inner in value.items()
            if inner is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    return value
