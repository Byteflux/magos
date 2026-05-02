"""OpenAI-shape ``GET /v1/models`` adapter.

Used by OpenAI proper, vLLM, SGLang, LM Studio, and any other server that
implements the OpenAI models endpoint. The endpoint returns very little:
``{id, created, owned_by, object}`` per model. We pass through ``id`` and
let the merge layer fill in everything else from ``magos.yaml`` overrides
or LiteLLM's bundled registry.

``litellm_provider`` on the provider config controls the LiteLLM dispatch
prefix; if unset, the adapter uses its own default of ``openai``. Local
inference servers (vLLM, SGLang) typically want this set to ``hosted_vllm``
or similar so LiteLLM points at the right driver.
"""

from __future__ import annotations

import os

import httpx

from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.schema import ProviderConfig

_DEFAULT_LITELLM_PROVIDER = "openai"


class OpenAIModelsAdapter:
    """Calls ``GET {base_url}/v1/models`` and maps ``data[*].id`` to entries."""

    name = "openai_models"

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        if not config.base_url:
            raise DiscoveryError(
                f"provider {provider_name!r}: base_url required for openai_models adapter"
            )
        url = config.base_url.rstrip("/") + "/v1/models"
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
            models.append(
                DiscoveredModel(
                    raw_id=raw_id,
                    litellm_id=f"{litellm_provider}/{raw_id}",
                )
            )
        return DiscoveryResult(models=tuple(models))


def _auth_headers(provider_name: str, config: ProviderConfig) -> dict[str, str]:
    """Read the API key from env if ``api_key_env`` is set; bearer-format it."""
    if not config.api_key_env:
        return {}
    key = os.environ.get(config.api_key_env)
    if not key:
        raise DiscoveryError(f"provider {provider_name!r}: env var {config.api_key_env} unset")
    return {"Authorization": f"Bearer {key}"}
