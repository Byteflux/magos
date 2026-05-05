"""Anthropic ``GET /v1/models`` adapter.

Anthropic's models endpoint returns ``{data: [{id, display_name, type,
created_at}, ...]}``. No context window, no pricing, no modality flags.
We pass through ``id``; field-precedence merging in
``magos.registry.merge`` fills the rest from LiteLLM's bundled registry
or operator overrides.

Auth uses the ``x-api-key`` header (Anthropic's convention) plus the
``anthropic-version`` header — both required by the API. Claude-Code-
style OAuth tokens (``sk-ant-oat...``) are detected and sent as
``Authorization: Bearer ...`` with the ``anthropic-beta: oauth-2025-04-20``
opt-in header instead, which is what api.anthropic.com expects for that
credential class.
"""

from __future__ import annotations

import os

import httpx

from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.litellm_lookup import PartialEntry
from magos.registry.schema import ProviderConfig

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_LITELLM_PROVIDER = "anthropic"
_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_TOKEN_PREFIX = "sk-ant-oat"  # noqa: S105
_OAUTH_BETA = "oauth-2025-04-20"


class AnthropicAdapter:
    name = "anthropic"
    # LiteLLM's anthropic provider already knows api.anthropic.com; no
    # need to override unless a future deployment uses an alternate host
    # (Bedrock, Vertex, internal proxy).
    default_base_url: str | None = None

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        base = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        url = f"{base}/v1/models"
        headers = _auth_headers(provider_name, config)
        try:
            response = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"transport error from {url}: {exc}") from exc
        if response.is_error:
            raise DiscoveryError(
                f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise DiscoveryError(f"non-JSON response from {url}: {exc}") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise DiscoveryError(f"{url}: missing or non-list 'data' field")
        litellm_provider = config.litellm_provider or _DEFAULT_LITELLM_PROVIDER
        models: list[DiscoveredModel] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            raw_id = raw.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                continue
            litellm_id = f"{litellm_provider}/{raw_id}"
            models.append(
                DiscoveredModel(
                    raw_id=raw_id,
                    litellm_id=litellm_id,
                    # Stamp litellm_id on the partial so merge records
                    # 'discovery' in sources even though Anthropic's models
                    # endpoint returns no enrichable fields.
                    partial=PartialEntry(litellm_id=litellm_id),
                )
            )
        return DiscoveryResult(models=tuple(models))


def _auth_headers(provider_name: str, config: ProviderConfig) -> dict[str, str]:
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
