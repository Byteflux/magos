"""OpenRouter ``GET /api/v1/models`` adapter.

Catalog includes context size, pricing, modalities, and max-output;
all mapped into ``PartialEntry``. Pricing is per-token USD upstream;
scaled to per-million on ingest.
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

_DEFAULT_BASE_URL = "https://openrouter.ai/api"
_DEFAULT_LITELLM_PROVIDER = "openrouter"


class OpenRouterAdapter:
    name = "openrouter"
    # LiteLLM's openrouter provider already knows the host.
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
            models.append(
                DiscoveredModel(
                    raw_id=raw_id,
                    litellm_id=f"{litellm_provider}/{raw_id}",
                    partial=_partial_from_openrouter_entry(raw),
                )
            )
        return DiscoveryResult(models=tuple(models))


def _partial_from_openrouter_entry(raw: dict[str, Any]) -> PartialEntry:
    pricing = _dict_field(raw, "pricing")
    architecture = _dict_field(raw, "architecture")
    top_provider = _dict_field(raw, "top_provider")
    context_size = _coerce_int(raw.get("context_length"))
    max_output = _coerce_int(top_provider.get("max_completion_tokens"))
    # OpenRouter's data is occasionally self-inconsistent: max_completion_tokens
    # exceeds context_length on a handful of catalog entries. Per their docs
    # context_length is total, so drop the bogus output cap.
    if context_size is not None and max_output is not None and max_output > context_size:
        max_output = None
    return PartialEntry(
        context_size=context_size,
        max_output=max_output,
        # OpenRouter uses -1 in pricing.* to mean "varies by underlying model"
        # for meta routes (auto, bodybuilder, pareto-code). Treat as unknown.
        # Otherwise, scale per-token USD into per-million USD.
        input_cost=_per_token_to_per_million(_coerce_float(pricing.get("prompt"))),
        output_cost=_per_token_to_per_million(_coerce_float(pricing.get("completion"))),
        cache_read_cost=_per_token_to_per_million(_coerce_float(pricing.get("input_cache_read"))),
        cache_write_cost=_per_token_to_per_million(_coerce_float(pricing.get("input_cache_write"))),
        input_modalities=_input_modalities(architecture),
        output_modalities=_output_modalities(architecture),
    )


def _per_token_to_per_million(value: float | None) -> float | None:
    if value is None or value < 0:
        return None
    return value * 1_000_000


def _dict_field(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


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


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _input_modalities(architecture: dict[str, Any]) -> tuple[str, ...] | None:
    """Prefer the explicit ``architecture.input_modalities`` array.

    Older catalog entries only carry a legacy ``modality: "X+Y->Z"``
    string; we split on the arrow and take the left side.
    """
    explicit = _modality_array(architecture.get("input_modalities"))
    if explicit is not None:
        return explicit
    return _legacy_modality_side(architecture.get("modality"), side=0)


def _output_modalities(architecture: dict[str, Any]) -> tuple[str, ...] | None:
    """Same shape as :func:`_input_modalities`, but the output side."""
    explicit = _modality_array(architecture.get("output_modalities"))
    if explicit is not None:
        return explicit
    return _legacy_modality_side(architecture.get("modality"), side=1)


def _modality_array(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    items = tuple(v.strip() for v in value if isinstance(v, str) and v.strip())
    return items or None


def _legacy_modality_side(value: Any, *, side: int) -> tuple[str, ...] | None:
    """Pull one side of OpenRouter's legacy ``"text+image->text"`` field."""
    if not isinstance(value, str):
        return None
    halves = value.split("->", 1)
    if len(halves) <= side:
        return None
    parts = tuple(p.strip() for p in halves[side].split("+") if p.strip())
    return parts or None


def _auth_headers(provider_name: str, config: ProviderConfig) -> dict[str, str]:
    """Bearer-format the API key from env if ``api_key_env`` is set."""
    if not config.api_key_env:
        return {}
    key = os.environ.get(config.api_key_env)
    if not key:
        raise DiscoveryError(f"provider {provider_name!r}: env var {config.api_key_env} unset")
    return {"Authorization": f"Bearer {key}"}
