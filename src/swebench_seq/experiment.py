"""Experiment-file loading and task-selection input parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


class ExperimentConfigError(ValueError):
    """Raised when an experiment file cannot be resolved or validated."""


def resolve_experiment_config_path(name_or_path: str) -> Path:
    """Resolve a bundled experiment name or an explicit YAML path."""

    named = EXPERIMENT_CONFIG_DIR / f"{name_or_path}.yaml"
    if named.is_file():
        return named
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate
    available = sorted(path.stem for path in EXPERIMENT_CONFIG_DIR.glob("*.yaml"))
    raise ExperimentConfigError(
        f"experiment config not found: {name_or_path!r} "
        f"(looked for {named} and {candidate}). Built-ins: {available}"
    )


def normalize_set_overrides(value: Any) -> list[str]:
    """Normalize a config ``set`` block into ``dotted.key=value`` entries."""

    if isinstance(value, dict):
        return [f"{key}={item}" for key, item in value.items()]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str) or "=" not in item:
                raise ExperimentConfigError(
                    "experiment-config `set` list items must be "
                    f"'dotted.key=value', got {item!r}"
                )
            items.append(item)
        return items
    raise ExperimentConfigError("experiment-config `set` must be a mapping or a list")


def load_experiment_config(name_or_path: str) -> tuple[Path, dict[str, Any]]:
    """Load one bundled or explicit experiment YAML mapping."""

    path = resolve_experiment_config_path(name_or_path)
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(f"failed to read experiment config {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ExperimentConfigError("experiment config must be a YAML mapping")
    return path, data


def read_instance_ids_file(path: Path) -> list[str]:
    """Read newline, JSON-list, or JSONL task instance IDs."""

    text = path.read_text()
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError(f"{path} must contain a JSON list of strings")
        return list(data)

    instance_ids: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            data = json.loads(line)
            if not isinstance(data, dict) or not isinstance(data.get("instance_id"), str):
                raise ValueError(
                    f"JSONL line in {path} must contain string field instance_id"
                )
            instance_ids.append(data["instance_id"])
        else:
            instance_ids.append(line)
    return instance_ids


def experiment_defaults(data: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    """Translate an experiment mapping into defaults for the run command."""

    allowed = allowed - {"set_overrides"}
    defaults: dict[str, Any] = {}
    for key, value in data.items():
        if key == "set":
            defaults["set_overrides"] = normalize_set_overrides(value)
            continue
        if key not in allowed:
            raise ExperimentConfigError(
                f"unknown experiment-config key: {key!r}. "
                f"Allowed keys: {', '.join(sorted(allowed | {'set'}))}"
            )
        defaults[key] = value
    return defaults
