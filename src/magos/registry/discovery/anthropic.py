"""Anthropic ``GET /v1/models`` adapter.

Endpoint returns just ``id``; merge fills the rest. ``sk-ant-oat...``
OAuth tokens use ``Authorization: Bearer`` + ``anthropic-beta:
oauth-2025-04-20``; everything else uses ``x-api-key``. All requests
require ``anthropic-version: 2023-06-01``.
"""

from __future__ import annotations

from magos.registry.discovery._auth import anthropic_auth
from magos.registry.discovery.base import JsonListAdapter
from magos.registry.schema import ProviderConfig

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_LITELLM_PROVIDER = "anthropic"


class AnthropicAdapter(JsonListAdapter):
    """Calls ``GET {base_url}/v1/models`` and maps ``data[*].id`` to entries."""

    name = "anthropic"
    # LiteLLM's anthropic provider already knows the host; ``None`` signals
    # that to the refresher. Discovery falls back to _DEFAULT_BASE_URL below.
    default_base_url: str | None = None

    _path_suffix = "/v1/models"
    _data_field = "data"
    _default_litellm_provider = _DEFAULT_LITELLM_PROVIDER
    _auth_headers = staticmethod(anthropic_auth)

    def _build_url(self, config: ProviderConfig) -> str:
        # Use the canonical api.anthropic.com when operator omits base_url,
        # even though default_base_url is None (the LiteLLM-hint field).
        base = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        return base + self._path_suffix
