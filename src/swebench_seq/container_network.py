"""Podman networking configuration for rollout containers."""

from __future__ import annotations

import os

DEFAULT_NO_PROXY = "localhost,127.0.0.1,::1"


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def network_mode_for(kind: str) -> str | None:
    """Return an explicitly configured Podman network mode, if any."""

    kind_key = kind.upper()
    return _env_value(
        f"SWEBENCH_{kind_key}_NETWORK_MODE",
        "SWEBENCH_CONTAINER_NETWORK_MODE",
    )


def proxy_env_for(kind: str) -> dict[str, str]:
    """Build standard proxy variables for a task container.

    The caller may use ordinary host proxy variables or the scoped
    ``SWEBENCH_<KIND>_PROXY`` / ``SWEBENCH_CONTAINER_PROXY`` overrides.
    No proxy is configured when none was supplied explicitly.
    """

    kind_key = kind.upper()
    proxy = _env_value(
        f"SWEBENCH_{kind_key}_PROXY",
        "SWEBENCH_CONTAINER_PROXY",
    )
    http_proxy = _env_value("HTTP_PROXY", "http_proxy") or proxy
    https_proxy = _env_value("HTTPS_PROXY", "https_proxy") or proxy
    if http_proxy is None and https_proxy is None:
        return {}
    http_proxy = http_proxy or https_proxy
    https_proxy = https_proxy or http_proxy
    assert http_proxy is not None and https_proxy is not None

    pieces: list[str] = []
    for value in (_env_value("NO_PROXY", "no_proxy"), DEFAULT_NO_PROXY):
        if not value:
            continue
        for piece in (item.strip() for item in value.split(",") if item.strip()):
            if piece not in pieces:
                pieces.append(piece)
    no_proxy = ",".join(pieces)
    return {
        "HTTP_PROXY": http_proxy,
        "HTTPS_PROXY": https_proxy,
        "http_proxy": http_proxy,
        "https_proxy": https_proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }
