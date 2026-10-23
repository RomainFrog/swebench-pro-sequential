"""Credential redaction shared by persisted configs and run manifests."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def key_is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in {
        "api_key",
        "apikey",
        "access_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "credentials",
        "password",
        "private_key",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "set_cookie",
    } or normalized.endswith(
        ("_api_key", "_auth", "_credential", "_key", "_password", "_secret", "_token")
    )


def path_is_sensitive(dotted_key: str) -> bool:
    return any(
        key_is_sensitive(part)
        or part.lower().replace("-", "_").endswith("headers")
        for part in dotted_key.split(".")
    )


def redact_secrets(value: Any, *, redact_values: bool = False) -> Any:
    """Recursively redact credentials, authentication headers, and URL auth."""

    if redact_values:
        if isinstance(value, dict):
            return {key: "<redacted>" for key in value}
        if isinstance(value, list):
            return ["<redacted>" for _ in value]
        return "<redacted>"
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_base", "base_url"} and isinstance(nested, str):
                redacted[key] = sanitize_url(nested)
            else:
                redacted[key] = redact_secrets(
                    nested,
                    redact_values=(
                        key_is_sensitive(str(key)) or normalized.endswith("headers")
                    ),
                )
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def sanitize_url(value: str | None) -> str | None:
    """Remove userinfo, query parameters, and fragments from a recorded URL."""

    if value is None:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        if not parsed.scheme or hostname is None:
            return value.split("?", 1)[0].split("#", 1)[0]
        host = f"[{hostname}]" if ":" in hostname else hostname
        netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<redacted-url>"
