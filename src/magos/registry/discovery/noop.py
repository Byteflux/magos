"""No-op adapter: empty ``DiscoveryResult``.

Manual ``models:`` entries are permanent (no deprecation cycle without
live discovery contribution). See ``docs/registry/config.md``.
"""

from __future__ import annotations

import httpx

from magos.registry.discovery.base import DiscoveryResult
from magos.registry.schema import ProviderConfig


class NoopAdapter:
    name = "noop"
    default_base_url: str | None = None

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        return DiscoveryResult()
