"""OpenRouter ``GET /api/v1/models`` adapter.

Catalog includes context size, pricing, modalities, and max-output;
all mapped into ``PartialEntry``. Pricing is per-token USD upstream;
scaled to per-million on ingest.
"""

from __future__ import annotations

from typing import Any

from magos.registry.discovery._auth import bearer_auth
from magos.registry.discovery._coerce import coerce_float, coerce_int, per_token_to_per_million
from magos.registry.discovery.base import JsonListAdapter
from magos.registry.litellm_lookup import PartialEntry

_DEFAULT_BASE_URL = "https://openrouter.ai/api"
_DEFAULT_LITELLM_PROVIDER = "openrouter"


class OpenRouterAdapter(JsonListAdapter):
    """Calls ``GET {base_url}/v1/models`` and maps the enriched catalog to entries."""

    name = "openrouter"
    # LiteLLM's openrouter provider already knows the host.
    default_base_url: str | None = None

    _path_suffix = "/v1/models"
    _data_field = "data"
    _default_litellm_provider = _DEFAULT_LITELLM_PROVIDER
    _auth_headers = staticmethod(bearer_auth)
    _partial_from_entry = staticmethod(lambda raw, litellm_id: _partial_from_openrouter_entry(raw))

    def _build_url(self, config: Any) -> str:
        base = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        return base + self._path_suffix


def _partial_from_openrouter_entry(raw: dict[str, Any]) -> PartialEntry:
    pricing = _dict_field(raw, "pricing")
    architecture = _dict_field(raw, "architecture")
    top_provider = _dict_field(raw, "top_provider")
    context_size = coerce_int(raw.get("context_length"))
    max_output = coerce_int(top_provider.get("max_completion_tokens"))
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
        input_cost=per_token_to_per_million(coerce_float(pricing.get("prompt"))),
        output_cost=per_token_to_per_million(coerce_float(pricing.get("completion"))),
        cache_read_cost=per_token_to_per_million(coerce_float(pricing.get("input_cache_read"))),
        cache_write_cost=per_token_to_per_million(coerce_float(pricing.get("input_cache_write"))),
        input_modalities=_input_modalities(architecture),
        output_modalities=_output_modalities(architecture),
    )


def _dict_field(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


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
