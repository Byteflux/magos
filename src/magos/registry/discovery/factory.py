"""Adapter factory: map ``ProviderConfig`` to a concrete adapter.

When ``discovery`` is unset, the host of ``base_url`` picks a default;
unset ``base_url`` falls back to ``noop`` (manual-only). See
``docs/registry/config.md``.
"""

from __future__ import annotations

from urllib.parse import urlparse

from magos.registry.discovery.anthropic import AnthropicAdapter
from magos.registry.discovery.base import DiscoveryAdapter
from magos.registry.discovery.noop import NoopAdapter
from magos.registry.discovery.openai import OpenAIAdapter
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.discovery.vultr import VultrAdapter
from magos.registry.schema import ProviderConfig

_ADAPTERS: dict[str, type[DiscoveryAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "openrouter": OpenRouterAdapter,
    "vultr": VultrAdapter,
    "noop": NoopAdapter,
}


def adapter_for(config: ProviderConfig) -> DiscoveryAdapter:
    """Return a fresh adapter instance for ``config``.

    Explicit ``discovery`` wins; otherwise the host of ``base_url`` is
    inspected to pick a sensible default.
    """
    name = config.discovery or _infer_adapter(config.base_url)
    cls = _ADAPTERS[name]
    return cls()


def _infer_adapter(base_url: str | None) -> str:
    if not base_url:
        return "noop"
    host = (urlparse(base_url).hostname or "").lower()
    if "openrouter.ai" in host:
        return "openrouter"
    if "anthropic.com" in host:
        return "anthropic"
    if "vultrinference.com" in host:
        return "vultr"
    return "openai"
