"""Provider-specific message normalization around mini-swe-agent models."""

from __future__ import annotations

import copy
from typing import Any, cast

_OPENAI_CHAT_MESSAGE_KEYS = frozenset(
    {
        "role",
        "content",
        "name",
        "tool_calls",
        "tool_call_id",
        "function_call",
    }
)


class _OpenAIChatMessageSanitizingModel:
    """Send OpenAI-chat-safe messages without mutating mini's audit trail."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            self._model.query(sanitize_openai_chat_messages(messages), **kwargs),
        )


class _AnthropicThinkingReplayModel:
    """Preserve signed Anthropic thinking blocks in mini's live history."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        response: dict[str, Any] = self._model.query(messages, **kwargs)
        thinking_blocks = extract_anthropic_replay_thinking_blocks(response)
        if not thinking_blocks:
            return response
        response = dict(response)
        response["thinking_blocks"] = thinking_blocks
        return response


def sanitize_openai_chat_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop mini/LiteLLM bookkeeping fields rejected by OpenAI chat APIs."""

    return [
        {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key in _OPENAI_CHAT_MESSAGE_KEYS
        }
        for message in messages
    ]


def extract_anthropic_replay_thinking_blocks(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract provider-valid thinking blocks for Anthropic history replay."""

    raw_response = response.get("extra", {}).get("response", {})
    if not isinstance(raw_response, dict):
        return []
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return []
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return []

    thinking_blocks = normalize_replay_thinking_blocks(message.get("thinking_blocks"))
    if thinking_blocks:
        return thinking_blocks
    provider_fields = message.get("provider_specific_fields")
    if isinstance(provider_fields, dict):
        return normalize_replay_thinking_blocks(provider_fields.get("thinking_blocks"))
    return []


def normalize_replay_thinking_blocks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    replayable: list[dict[str, Any]] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            thinking = block.get("thinking")
            signature = block.get("signature")
            if isinstance(thinking, str) and thinking and isinstance(signature, str) and signature:
                replayable.append(copy.deepcopy(block))
        elif block_type == "redacted_thinking":
            data = block.get("data")
            if isinstance(data, str) and data:
                replayable.append(copy.deepcopy(block))
    return replayable


def wrap_model_for_provider(model: Any, provider: str) -> Any:
    """Apply only the history adapter required by the selected provider."""

    if getattr(model, "_is_responses_model", False):
        return model
    if provider == "openai":
        return _OpenAIChatMessageSanitizingModel(model)
    if provider == "anthropic":
        return _AnthropicThinkingReplayModel(model)
    return model
