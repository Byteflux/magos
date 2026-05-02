"""Adapter factory: map provider config to a concrete adapter instance.

Kept separate from ``__init__`` so adapters can be added without touching
the package init, and so the factory can be replaced wholesale in tests
when adapter selection logic itself is under test.
"""

from __future__ import annotations

from magos.registry.discovery.anthropic_models import AnthropicModelsAdapter
from magos.registry.discovery.base import DiscoveryAdapter
from magos.registry.discovery.noop import NoopAdapter
from magos.registry.discovery.openai_models import OpenAIModelsAdapter
from magos.registry.discovery.openrouter import OpenRouterAdapter
from magos.registry.schema import ProviderConfig

_ADAPTERS: dict[str, type[DiscoveryAdapter]] = {
    "openai_models": OpenAIModelsAdapter,
    "anthropic_models": AnthropicModelsAdapter,
    "openrouter": OpenRouterAdapter,
    "noop": NoopAdapter,
}


def adapter_for(config: ProviderConfig) -> DiscoveryAdapter:
    """Return a fresh adapter instance for ``config``.

    Unset ``discovery`` falls back to the no-op adapter, matching the
    design that adapter-unset providers are manual-only.
    """
    name = config.discovery or "noop"
    cls = _ADAPTERS[name]
    return cls()
