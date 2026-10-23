"""mini-swe-agent configuration, provider credentials, and artifact redaction."""

from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..redaction import redact_secrets
from .environment import DEFAULT_WORKDIR

_AGENT_HOME = "/home/agent"
_DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS = 180
_PROMPT_CACHE_NAMESPACE = uuid.UUID("3c285279-6a78-4ff8-b4b1-27d072723a61")
_SEQUENTIAL_SYSTEM_ADDITION = """

Sequential mode:
Your HOME directory /home/agent persists across related tasks in this
repository, so you can build up reusable knowledge of this codebase.

- /home/agent/MEMORY.md may be included below in this system prompt. Treat it
  as a concise, durable cheat-sheet: how to build and run the tests, the repo
  layout, recurring gotchas, and fixes that generalize. CURATE it — rewrite
  and prune rather than endlessly appending.
- Files under /home/agent/memory/ also persist but are NOT injected — use them
  for bulkier scratch notes you can re-read on demand.
- Each task is a different commit of this repo, so exact line numbers and some
  paths may drift between tasks; prefer durable facts (commands, structure,
  symbol names) over brittle ones.
"""
_TASK_INSTRUCTIONS_SYSTEM_HEADER = "Task-invariant SWE-bench instructions:"


@dataclass(frozen=True)
class MiniSweAgentOptions:
    """Runtime options layered onto mini-swe-agent's benchmark config."""

    config_model: dict[str, Any] | None = None
    config_overrides: tuple[tuple[str, str], ...] = ()
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    provider: str = "openai"
    cost_tracking: str = "ignore_errors"
    workdir: str = DEFAULT_WORKDIR
    prompt_cache_key: str | None = None


def prompt_cache_key_for_repo(repo: str) -> str:
    """Return the stable cache-routing key shared by a repository's tasks."""

    return str(uuid.uuid5(_PROMPT_CACHE_NAMESPACE, repo))


def with_repo_prompt_cache_key(
    options: MiniSweAgentOptions,
    repo: str,
) -> MiniSweAgentOptions:
    """Scope Responses API prompt caching to one repository universe."""

    return replace(options, prompt_cache_key=prompt_cache_key_for_repo(repo))


def load_mini_imports() -> dict[str, Any]:
    """Import the pinned mini-swe-agent package."""

    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    try:
        from minisweagent.agents.default import (
            DefaultAgent,
            NonTerminatingException,
            TerminatingException,
        )
        from minisweagent.config import builtin_config_dir, get_config_path
        from minisweagent.models import get_model
        from minisweagent.run.utils.save import save_traj
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mini-swe-agent is not importable; install the locked environment "
            "with `uv sync`."
        ) from exc

    return {
        "DefaultAgent": DefaultAgent,
        "NonTerminatingException": NonTerminatingException,
        "TerminatingException": TerminatingException,
        "builtin_config_dir": builtin_config_dir,
        "get_config_path": get_config_path,
        "get_model": get_model,
        "save_traj": save_traj,
    }


def load_config(imports: dict[str, Any]) -> dict[str, Any]:
    target = imports["builtin_config_dir"] / "extra" / "swebench.yaml"
    resolved = imports["get_config_path"](target)
    data = yaml.safe_load(Path(resolved).read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"mini-swe-agent config did not load as a mapping: {resolved}")
    return data


def build_mini_config(
    imports: dict[str, Any],
    options: MiniSweAgentOptions,
) -> dict[str, Any]:
    """Build mini's base config plus the experiment's model and overrides."""

    config = load_config(imports)
    if options.config_model is not None:
        config["model"] = copy.deepcopy(options.config_model)
    for dotted, raw_value in options.config_overrides:
        apply_override(config, dotted, raw_value)
    return config


def apply_override(config: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a dotted config override after parsing its value as YAML."""

    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = yaml.safe_load(raw_value)


def is_responses_model_class(model_class: Any) -> bool:
    return isinstance(model_class, str) and model_class.endswith("OpenAIResponsesModel")


def prepare_config(config: dict[str, Any], options: MiniSweAgentOptions) -> dict[str, Any]:
    """Inject runtime-only paths, credentials, endpoint, and model overrides."""

    config = copy.deepcopy(config)
    if options.workdir != "/testbed":
        config["agent"] = _replace_string_values(
            config.get("agent", {}), old="/testbed", new=options.workdir
        )

    env_config = config.setdefault("environment", {})
    env_config["cwd"] = options.workdir
    env_config.pop("environment_class", None)

    model_config = config.setdefault("model", {})
    if options.model:
        model_config["model_name"] = normalize_model_name(options.model, options.provider)
    model_config.setdefault("cost_tracking", options.cost_tracking)

    model_kwargs = model_config.setdefault("model_kwargs", {})
    if options.api_key:
        model_kwargs["api_key"] = options.api_key
    if options.base_url:
        model_kwargs["api_base"] = normalize_api_base(options.base_url, provider=options.provider)
    if is_responses_model_class(model_config.get("model_class")):
        for key in ("drop_params", "temperature", "custom_llm_provider"):
            model_kwargs.pop(key, None)
        cache_key_header = model_kwargs.pop("prompt_cache_key_header", None)
        if cache_key_header is not None and (
            not isinstance(cache_key_header, str) or not cache_key_header.strip()
        ):
            raise TypeError(
                "model.model_kwargs.prompt_cache_key_header must be a non-empty string"
            )
        cache_key = model_kwargs.get("prompt_cache_key") or options.prompt_cache_key
        if cache_key is not None:
            if not isinstance(cache_key, str):
                raise TypeError("model.model_kwargs.prompt_cache_key must be a string")
            model_kwargs["prompt_cache_key"] = cache_key
            if cache_key_header:
                extra_headers = model_kwargs.setdefault("extra_headers", {})
                if not isinstance(extra_headers, dict):
                    raise TypeError("model.model_kwargs.extra_headers must be a mapping")
                extra_headers.setdefault(cache_key_header, cache_key)
    elif "thinking" in model_kwargs:
        model_kwargs.pop("temperature", None)
        model_kwargs.setdefault("drop_params", True)
    else:
        model_kwargs.setdefault("drop_params", True)
        model_kwargs.setdefault("temperature", 0.0)
        if options.provider == "openai" and options.base_url:
            model_kwargs.setdefault("custom_llm_provider", "openai")
            model_kwargs.setdefault("timeout", _DEFAULT_OPENAI_CHAT_TIMEOUT_SECONDS)
    return config


def prepare_sequential_config(
    config: dict[str, Any],
    options: MiniSweAgentOptions,
) -> dict[str, Any]:
    config = prepare_config(config, options)
    env_config = config.setdefault("environment", {})
    env = env_config.setdefault("env", {})
    env["HOME"] = _AGENT_HOME
    agent_config = config.setdefault("agent", {})
    _promote_task_instructions_to_system(agent_config)
    system_template = agent_config.get("system_template") or ""
    if "Sequential mode:" not in system_template:
        agent_config["system_template"] = system_template.rstrip() + _SEQUENTIAL_SYSTEM_ADDITION
    return config


def normalize_model_name(model: str, provider: str) -> str:
    """Add the provider prefix expected by LiteLLM when it is absent."""

    if provider == "anthropic" and "/" not in model:
        return f"anthropic/{model}"
    if provider == "openrouter" and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def normalize_api_base(base_url: str, *, provider: str) -> str:
    """Account for LiteLLM's provider-specific URL suffix handling."""

    base = base_url.rstrip("/")
    if provider == "anthropic" and base.endswith("/v1"):
        return f"{base}/messages"
    return base


def redact_json_file(path: Path) -> None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    path.write_text(json.dumps(redact_secrets(data), indent=2))


def _promote_task_instructions_to_system(agent_config: dict[str, Any]) -> None:
    """Move mini's invariant SWE-bench instructions out of the task template."""

    instance_template = str(agent_config.get("instance_template") or "")
    system_template = str(agent_config.get("system_template") or "")
    instructions = _extract_tagged_block(instance_template, "instructions")
    if instructions and _TASK_INSTRUCTIONS_SYSTEM_HEADER not in system_template:
        agent_config["system_template"] = (
            system_template.rstrip()
            + "\n\n"
            + _TASK_INSTRUCTIONS_SYSTEM_HEADER
            + "\n"
            + instructions.strip()
        )
    agent_config["instance_template"] = "{{task}}"


def _extract_tagged_block(text: str, tag: str) -> str | None:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = text.find(start_token)
    if start < 0:
        return None
    end = text.find(end_token, start + len(start_token))
    if end < 0:
        return None
    return text[start : end + len(end_token)]


def _replace_string_values(obj: Any, *, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_replace_string_values(value, old=old, new=new) for value in obj]
    if isinstance(obj, dict):
        return {
            key: _replace_string_values(value, old=old, new=new)
            for key, value in obj.items()
        }
    return obj
