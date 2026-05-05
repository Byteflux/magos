"""API-key resolution and per-provider auth-header injection.

Two seams the dispatcher relies on:

- ``resolve_api_key(env)``: translate-mode helper. Reads the env var
  named by ``action.api_key_env`` and hands the value off to LiteLLM
  as ``api_key=``. Lets one provider host multiple keys (e.g. tier
  routing) by declaring separate rules with different env vars.

- ``maybe_inject_api_key(headers, action)``: passthrough-mode helper.
  When the inbound request carries no ``Authorization`` or
  ``x-api-key`` header and ``action.api_key_env`` is set, injects the
  env value in the shape that provider expects:

  - ``provider: anthropic`` → ``x-api-key: <env>`` (Anthropic's
    convention).
  - everything else → ``Authorization: Bearer <env>`` (the
    openai-compatible convention used by openai, openrouter, vultr,
    etc.).
  - ``action.auth_header`` overrides the default.
  - Anthropic OAuth tokens (``sk-ant-oat...``) override both: they
    always go out as a Bearer plus an ``anthropic-beta:
    oauth-2025-04-20`` opt-in header. api.anthropic.com 401s on
    ``x-api-key`` for that credential class.

Headers are not part of the prompt-cache hash, so injection here does
not break Anthropic's byte-exact billing.

``DispatchError`` is raised when the named env var is missing; the
ingress layer turns it into the standard 503 ``dispatch_error``
envelope.
"""

from __future__ import annotations

import os

from magos.routing.schema import Action

_ANTHROPIC_OAUTH_TOKEN_PREFIX = "sk-ant-oat"  # noqa: S105
_ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"


class DispatchError(Exception):
    """Raised when a runtime config invariant fails (e.g., missing env var)."""


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
