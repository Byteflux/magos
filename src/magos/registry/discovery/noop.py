"""No-op adapter: provider has no live discovery.

Used when ``discovery`` is unset or explicitly ``noop``. Returns an empty
``DiscoveryResult``. Manual entries from the provider's ``models`` block
are combined with discovery and override entries by the precedence merge
in ``magos.registry.merge``; with no live contribution there is no
deprecation cycle, so manual entries are permanent until removed from
yaml.
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
