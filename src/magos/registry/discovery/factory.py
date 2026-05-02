"""Adapter factory: map provider config to a concrete adapter instance.

Kept separate from ``__init__`` so adapters can be added without touching
the package init, and so the factory can be replaced wholesale in tests
when adapter selection logic itself is under test.

Inference rules when ``discovery`` is unset:

- ``base_url`` matches openrouter.ai     → ``openrouter``
- ``base_url`` matches api.anthropic.com → ``anthropic_models``
- ``base_url`` is set (anything else)    → ``openai_models``
- ``base_url`` is unset                  → ``noop`` (manual-only)

Operators can always force an adapter explicitly via ``discovery:`` —
inference only fires when the field is omitted.
"""

from __future__ import annotations

from urllib.parse import urlparse

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
        return "anthropic_models"
    return "openai_models"
