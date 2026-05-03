"""Adapter Protocol and shared types for discovery.

Adapters are async, stateless objects taking a ``ProviderConfig`` plus an
``httpx.AsyncClient`` and producing a ``DiscoveryResult``. They are
expected to raise ``DiscoveryError`` on transport failures, auth failures,
and malformed responses; the refresher catches these and applies retry
policy.

A successful empty list is *not* a failure: some providers legitimately
serve zero models, and the refresher treats that as "this provider is
known to have nothing right now" rather than "this provider is broken".
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
    """One model returned by an adapter, pre-merge.

    ``raw_id`` is the provider-native identifier (e.g.
    ``anthropic/claude-sonnet-4-6`` for OpenRouter, or ``gpt-4o`` for
    OpenAI). ``litellm_id`` is the adapter-default dispatch id; the
    override layer can replace it during merge.
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

    # Adapter-canonical base URL used for both discovery and LiteLLM
    # dispatch when the operator hasn't set ``base_url`` in providers
    # config. ``None`` means: the adapter has no opinion (either the
    # provider has no fixed host, e.g. self-hosted vLLM, or the host is
    # already covered by a LiteLLM-native provider that knows its own
    # default URL). For openai-compatible third parties routed through
    # ``custom_openai`` (e.g. Vultr), this value is the only place the
    # dispatch URL comes from when operators omit ``base_url``.
    default_base_url: str | None

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult: ...
