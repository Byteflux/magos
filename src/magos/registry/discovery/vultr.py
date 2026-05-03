"""Vultr Cloud Inference ``GET /v1/models/lookup`` adapter.

Vultr's openai-compatible inference API also exposes a richer per-model
metadata endpoint. ``/v1/models/lookup`` returns ``context_length`` and
``cost_input`` / ``cost_output`` for each entry, which the standard
``/v1/models`` endpoint does not. We hit the lookup endpoint and map the
extra fields into a ``PartialEntry``.

LiteLLM has no vultr-specific provider, so the adapter defaults
``litellm_provider`` to ``custom_openai`` — LiteLLM's generic
openai-compatible shape, which requires an explicit ``api_base`` and
won't silently fall back to ``api.openai.com`` + ``OPENAI_API_KEY``
the way bare ``openai`` does. Operators must still supply ``base_url``
and ``api_key_env`` so the dispatcher hands both to LiteLLM. Override
``litellm_provider`` in the provider config if a future LiteLLM
release adds vendor-specific support.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.litellm_lookup import PartialEntry
from magos.registry.schema import ProviderConfig

_DEFAULT_BASE_URL = "https://api.vultrinference.com/v1"
_DEFAULT_LITELLM_PROVIDER = "custom_openai"

# Vultr's pricing fields are integer cents per million tokens (e.g. ``30``
# means $0.30 per million tokens). magos tracks USD per million tokens,
# so divide by 100 (cents -> dollars).
_CENTS_TO_DOLLARS = 100


class VultrAdapter:
    """Calls ``GET {base_url}/v1/models/lookup`` and maps the model array."""

    name = "vultr"
    default_base_url: str | None = _DEFAULT_BASE_URL

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        base = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        # Vultr's base_url commonly includes ``/v1`` already; tolerate both
        # ``https://api.vultrinference.com`` and ``.../v1`` shapes.
        url = base + ("/models/lookup" if base.endswith("/v1") else "/v1/models/lookup")
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
        data = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise DiscoveryError(f"{url}: missing or non-list 'models' field")
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
                    partial=_partial_from_vultr_entry(raw),
                )
            )
        return DiscoveryResult(models=tuple(models))


def _partial_from_vultr_entry(raw: dict[str, Any]) -> PartialEntry:
    return PartialEntry(
        context_size=_coerce_int(raw.get("context_length")),
        input_cost=_cents_to_dollars_per_million(raw.get("cost_input")),
        output_cost=_cents_to_dollars_per_million(raw.get("cost_output")),
    )


def _cents_to_dollars_per_million(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value) / _CENTS_TO_DOLLARS


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _auth_headers(provider_name: str, config: ProviderConfig) -> dict[str, str]:
    """Bearer-format the API key from env if ``api_key_env`` is set."""
    if not config.api_key_env:
        return {}
    key = os.environ.get(config.api_key_env)
    if not key:
        raise DiscoveryError(f"provider {provider_name!r}: env var {config.api_key_env} unset")
    return {"Authorization": f"Bearer {key}"}
