"""Discovery adapters: pluggable per-provider model enumeration.

Each adapter consumes a `ProviderConfig` plus an HTTP client and returns
a `DiscoveryResult` with one `DiscoveredModel` per model the provider
serves. Built-in adapters cover OpenAI-shape `/v1/models`,
Anthropic `/v1/models`, OpenRouter's richer catalog endpoint, and a
`noop` adapter for manual-only providers.

Adapters are pure consumers of injected HTTP clients; the refresher owns
retry/timeout policy and never imports specific adapter modules directly.
The `adapter_for` factory resolves the right implementation given a
provider's `discovery` setting.
"""

from __future__ import annotations

from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryAdapter,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.discovery.factory import adapter_for

__all__ = [
    "DiscoveredModel",
    "DiscoveryAdapter",
    "DiscoveryError",
    "DiscoveryResult",
    "adapter_for",
]
