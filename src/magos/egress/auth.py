"""API-key resolution and per-provider auth-header injection.

See ``docs/architecture/headers-and-auth.md``.
"""

from __future__ import annotations

import os

from magos.egress.errors import DispatchError
from magos.routing.schema import Action

__all__ = ["maybe_inject_api_key", "resolve_api_key"]

_ANTHROPIC_OAUTH_TOKEN_PREFIX = "sk-ant-oat"  # noqa: S105
_ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"


def maybe_inject_api_key(headers: dict[str, str], action: Action) -> dict[str, str]:
    """In passthrough mode, inject the env-resolved API key when absent.

    See module docstring for shape rules. Skipped entirely when the
    inbound request already carries ``Authorization`` or ``x-api-key``.
    """
    if action.mode != "passthrough" or not action.api_key_env:
        return headers
    if "authorization" in headers or "x-api-key" in headers:
        return headers
    value = os.environ.get(action.api_key_env)
    if not value:
        raise DispatchError(f"env var {action.api_key_env!r} is not set")
    if action.provider == "anthropic" and value.startswith(_ANTHROPIC_OAUTH_TOKEN_PREFIX):
        return {
            **headers,
            "authorization": f"Bearer {value}",
            "anthropic-beta": _ANTHROPIC_OAUTH_BETA,
        }
    shape = action.auth_header or _default_auth_header(action.provider)
    if shape == "x-api-key":
        return {**headers, "x-api-key": value}
    return {**headers, "authorization": f"Bearer {value}"}


def _default_auth_header(provider: str) -> str:
    """Pick the auth-header shape for a provider when no override is set."""
    return "x-api-key" if provider == "anthropic" else "bearer"


def resolve_api_key(api_key_env: str | None) -> str | None:
    """Translate-mode helper: read ``api_key_env`` from the environment."""
    if not api_key_env:
        return None
    value = os.environ.get(api_key_env)
    if not value:
        raise DispatchError(f"env var {api_key_env!r} is not set")
    return value
