"""No-op adapter: provider has no live discovery.

Used when ``discovery`` is unset or explicitly ``noop``. Returns an empty
``DiscoveryResult``. Manual entries from the provider's ``models`` block
flow into the registry through the merge layer alone, with no live
contribution and no deprecation cycle (per the design: manual entries
are permanent until removed from yaml).
"""

from __future__ import annotations

import httpx

from magos.registry.discovery.base import DiscoveryResult
from magos.registry.schema import ProviderConfig


class NoopAdapter:
    name = "noop"

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        return DiscoveryResult()
