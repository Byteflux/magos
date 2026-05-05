"""Vultr Cloud Inference ``GET /v1/models/lookup`` adapter.

Richer than the bare ``/v1/models`` endpoint: includes context size and
pricing. Defaults to ``litellm_provider: custom_openai`` (LiteLLM has no
vultr-native provider); operators must supply ``base_url`` + ``api_key_env``.
"""

from __future__ import annotations

from typing import Any

from magos.registry.discovery._auth import bearer_auth
from magos.registry.discovery._coerce import coerce_int
from magos.registry.discovery.base import JsonListAdapter
from magos.registry.litellm_lookup import PartialEntry

_DEFAULT_BASE_URL = "https://api.vultrinference.com/v1"
_DEFAULT_LITELLM_PROVIDER = "custom_openai"

# Vultr's pricing fields are integer cents per million tokens (e.g. ``30``
# means $0.30 per million tokens). magos tracks USD per million tokens,
# so divide by 100 (cents -> dollars).
_CENTS_TO_DOLLARS = 100


class VultrAdapter(JsonListAdapter):
    """Calls ``GET {base_url}/v1/models/lookup`` and maps the model array."""

    name = "vultr"
    default_base_url: str | None = _DEFAULT_BASE_URL

    _data_field = "models"
    _default_litellm_provider = _DEFAULT_LITELLM_PROVIDER
    _auth_headers = staticmethod(bearer_auth)
    _partial_from_entry = staticmethod(lambda raw, litellm_id: _partial_from_vultr_entry(raw))

    def _build_url(self, config: Any) -> str:
        base = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        # Vultr's base_url commonly includes ``/v1`` already; tolerate both
        # ``https://api.vultrinference.com`` and ``.../v1`` shapes.
        return base + ("/models/lookup" if base.endswith("/v1") else "/v1/models/lookup")


def _partial_from_vultr_entry(raw: dict[str, Any]) -> PartialEntry:
    return PartialEntry(
        context_size=coerce_int(raw.get("context_length")),
        input_cost=_cents_to_dollars_per_million(raw.get("cost_input")),
        output_cost=_cents_to_dollars_per_million(raw.get("cost_output")),
    )


def _cents_to_dollars_per_million(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value) / _CENTS_TO_DOLLARS
