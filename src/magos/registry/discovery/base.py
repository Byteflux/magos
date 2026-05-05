"""Discovery adapter Protocol + shared types.

Adapters are async/stateless: ``(ProviderConfig, httpx.AsyncClient) ->
DiscoveryResult``. Raise ``DiscoveryError`` on transport/auth/parse
failures; an empty result is success (provider serves zero models).

``JsonListAdapter`` is a concrete base for the common pattern: GET a JSON
endpoint, extract a named array field, call a per-entry builder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from magos.registry.litellm_lookup import PartialEntry
from magos.registry.schema import ProviderConfig


class DiscoveryError(Exception):
    """Raised by adapters on transport, auth, or parse failures."""


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    """One adapter-discovered model pre-merge; ``raw_id`` is provider-native,
    ``litellm_id`` is the adapter-default dispatch id (override can replace).
    """

    raw_id: str
    litellm_id: str
    partial: PartialEntry = field(default_factory=PartialEntry)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """All models a provider currently serves, plus optional adapter notes."""

    models: tuple[DiscoveredModel, ...] = ()
    notes: tuple[str, ...] = ()


class DiscoveryAdapter(Protocol):
    """Async callable: given config + client, enumerate models."""

    name: str

    # Adapter-canonical fallback URL when the operator omits ``base_url``.
    # ``None`` means the adapter has no opinion (provider has no fixed host,
    # or LiteLLM already knows the default). Required for ``custom_openai``
    # third parties (e.g. Vultr) since LiteLLM has no built-in host.
    default_base_url: str | None

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult: ...


class JsonListAdapter:
    """Shared base for adapters that GET a JSON endpoint and iterate a list field.

    Subclasses set class attributes to parameterise the shared logic:

    - ``name``: adapter key (matches ``discovery:`` yaml value)
    - ``default_base_url``: fallback when operator omits ``base_url``
    - ``_path_suffix``: appended to the (stripped) base URL
    - ``_data_field``: top-level JSON key that holds the model array
    - ``_default_litellm_provider``: prefix for ``litellm_id`` construction
    - ``_auth_headers``: callable ``(provider_name, config) -> dict``
    - ``_partial_from_entry``: callable ``(raw_dict, litellm_provider) -> PartialEntry``

    The default ``_partial_from_entry`` stamps only ``litellm_id`` (suitable
    for id-only endpoints like OpenAI and Anthropic). Override for richer
    catalog endpoints.
    """

    name: str
    default_base_url: str | None = None

    _path_suffix: str = "/v1/models"
    _data_field: str = "data"
    _default_litellm_provider: str = "openai"
    _auth_headers: Callable[[str, ProviderConfig], dict[str, str]]
    _partial_from_entry: Callable[[dict[str, object], str], PartialEntry] | None = None

    def _build_url(self, config: ProviderConfig) -> str:
        base = (config.base_url or self.default_base_url or "").rstrip("/")
        return base + self._path_suffix

    @staticmethod
    def _default_partial(raw: dict[str, object], litellm_id: str) -> PartialEntry:
        # Stamp ``litellm_id`` so merge records ``discovery`` in sources;
        # endpoint returns no other enrichable fields.
        return PartialEntry(litellm_id=litellm_id)

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        url = self._build_url(config)
        headers = self._auth_headers(provider_name, config)
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
        data = payload.get(self._data_field) if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise DiscoveryError(f"{url}: missing or non-list {self._data_field!r} field")
        litellm_provider = config.litellm_provider or self._default_litellm_provider
        partial_fn = self._partial_from_entry or JsonListAdapter._default_partial
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
                    partial=partial_fn(raw, litellm_id),
                )
            )
        return DiscoveryResult(models=tuple(models))
