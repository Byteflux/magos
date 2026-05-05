"""Auth-header builders for discovery adapters.

Each function reads the API key from the environment variable named by
``config.api_key_env`` and returns the appropriate header dict.
"""

from __future__ import annotations

import os

from magos.registry.discovery.base import DiscoveryError
from magos.registry.schema import ProviderConfig

_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_TOKEN_PREFIX = "sk-ant-oat"  # noqa: S105
_OAUTH_BETA = "oauth-2025-04-20"


def bearer_auth(provider_name: str, config: ProviderConfig) -> dict[str, str]:
    """Return ``Authorization: Bearer <key>`` if ``api_key_env`` is set.

    An unset env var raises ``DiscoveryError``; a missing ``api_key_env``
    returns an empty dict (adapter allows anonymous access).
    """
    if not config.api_key_env:
        return {}
    key = os.environ.get(config.api_key_env)
    if not key:
        raise DiscoveryError(f"provider {provider_name!r}: env var {config.api_key_env} unset")
    return {"Authorization": f"Bearer {key}"}


def anthropic_auth(provider_name: str, config: ProviderConfig) -> dict[str, str]:
    """Return headers for Anthropic's ``/v1/models`` endpoint.

    ``sk-ant-oat...`` OAuth tokens use ``Authorization: Bearer`` plus
    ``anthropic-beta: oauth-2025-04-20``; everything else uses ``x-api-key``.
    All requests require ``anthropic-version: 2023-06-01``.

    ``api_key_env`` is required; raises ``DiscoveryError`` if absent or unset.
    """
    if not config.api_key_env:
        raise DiscoveryError(
            f"provider {provider_name!r}: api_key_env required for anthropic adapter"
        )
    key = os.environ.get(config.api_key_env)
    if not key:
        raise DiscoveryError(f"provider {provider_name!r}: env var {config.api_key_env} unset")
    if key.startswith(_OAUTH_TOKEN_PREFIX):
        return {
            "authorization": f"Bearer {key}",
            "anthropic-beta": _OAUTH_BETA,
            "anthropic-version": _ANTHROPIC_VERSION,
        }
    return {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
