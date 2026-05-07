"""Shared helpers for discovery-adapter tests.

``ok`` / ``err`` build ``httpx.MockTransport`` responses; ``run_discover``
drives any adapter through one ``discover`` call against that transport.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from magos.registry.discovery.base import DiscoveryAdapter, DiscoveryResult
from magos.registry.schema import ProviderConfig


def ok(payload: dict[str, Any]) -> httpx.MockTransport:
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(_h)


def err(status: int, body: str = "") -> httpx.MockTransport:
    def _h(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return httpx.MockTransport(_h)


def run_discover(
    adapter: DiscoveryAdapter,
    name: str,
    cfg: ProviderConfig,
    transport: httpx.MockTransport,
) -> DiscoveryResult:
    async def _run() -> DiscoveryResult:
        async with httpx.AsyncClient(transport=transport) as client:
            return await adapter.discover(name, cfg, client)

    return asyncio.run(_run())
