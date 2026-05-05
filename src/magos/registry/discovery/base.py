"""Discovery adapter Protocol + shared types.

Adapters are async/stateless: ``(ProviderConfig, httpx.AsyncClient) ->
DiscoveryResult``. Raise ``DiscoveryError`` on transport/auth/parse
failures; an empty result is success (provider serves zero models).
"""

from __future__ import annotations

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
