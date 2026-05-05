"""Async lifecycle owner for the registry.

Per-provider refresh tasks; boot discovery has tighter timeouts and
fewer attempts than background. Sole writer to ``models.json`` (under
one ``asyncio.Lock``). See ``docs/registry/overview.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol

import backoff
import httpx

from magos.registry import telemetry as registry_telemetry
from magos.registry.deprecation import apply_deprecation
from magos.registry.discovery import (
    DiscoveryAdapter,
    DiscoveryError,
    DiscoveryResult,
    adapter_for,
)
from magos.registry.litellm_lookup import GetModelInfoFn
from magos.registry.pipeline import ProviderDiff, diff_provider, merge_provider
from magos.registry.schema import (
    ProviderConfig,
    RegistrySettings,
    RegistryYaml,
)
from magos.registry.state import ModelEntry, RegistryState
from magos.registry.store import load as load_state
from magos.registry.store import save as save_state
from magos.telemetry import get_logger

log = get_logger("magos.registry.refresher")


AdapterFactory = Callable[[ProviderConfig], DiscoveryAdapter]
ClientFactory = Callable[[float], httpx.AsyncClient]


class Clock(Protocol):
    def __call__(self) -> datetime: ...


def _default_client_factory(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout))


class Refresher:
    """Background owner of ``RegistryState`` with per-provider refresh tasks."""

    def __init__(
        self,
        config: RegistryYaml,
        models_path: Path,
        *,
        adapter_factory: AdapterFactory = adapter_for,
        client_factory: ClientFactory = _default_client_factory,
        litellm_lookup: GetModelInfoFn | None = None,
        clock: Clock = lambda: datetime.now().astimezone(),
    ) -> None:
        self._config = config
        self._models_path = models_path
        self._adapter_factory = adapter_factory
        self._client_factory = client_factory
        self._litellm_lookup = litellm_lookup
        self._clock = clock
        self._lock = asyncio.Lock()
        self._state = RegistryState()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopped = asyncio.Event()

    @property
    def state(self) -> RegistryState:
        return self._state

    async def start(self) -> None:
        """Load disk, run boot discovery for empty providers, start tasks."""
        loaded = load_state(self._models_path)
        async with self._lock:
            self._state = loaded
        await self._boot_discover_missing_providers()
        for provider_name in self._config.providers:
            self._tasks[provider_name] = asyncio.create_task(
                self._refresh_loop(provider_name),
                name=f"registry.refresh.{provider_name}",
            )

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def refresh(self, provider_name: str) -> None:
        """Force-refresh a single provider with background-tier patience."""
        await self._refresh_one(
            provider_name,
            timeout_seconds=self._registry_settings.discovery_timeout_seconds,
            max_attempts=self._registry_settings.discovery_max_attempts,
        )

    @property
    def _registry_settings(self) -> RegistrySettings:
        return self._config.registry

    def _interval_for(self, provider_name: str) -> int:
        cfg = self._config.providers[provider_name]
        return cfg.refresh_interval or self._registry_settings.refresh_interval

    async def _boot_discover_missing_providers(self) -> None:
        """One-shot tight-timeout discovery for providers absent from disk.

        Providers already present skip; failure leaves the provider empty
        until the background loop catches up.
        """
        boot_timeout = self._registry_settings.boot_discovery_timeout_seconds
        boot_attempts = self._registry_settings.boot_discovery_max_attempts
        present_providers = set(self._state.by_provider)
        missing = [p for p in self._config.providers if p not in present_providers]
        if not missing:
            return
        async with asyncio.TaskGroup() as tg:
            for name in missing:
                tg.create_task(
                    self._refresh_one_safe(
                        name,
                        timeout_seconds=boot_timeout,
                        max_attempts=boot_attempts,
                    )
                )

    async def _refresh_one_safe(
        self,
        provider_name: str,
        *,
        timeout_seconds: int,
        max_attempts: int,
    ) -> None:
        """Run one refresh, swallowing every exception so the loop survives.

        Unhandled exceptions kill the asyncio Task silently because
        ``self._tasks`` holds a strong reference; that suppresses the
        "Task exception was never retrieved" warning and refresh stops
        forever. ``DiscoveryError`` is expected (transport/auth/parse);
        anything else is a bug we surface via ``log.exception``.
        """
        try:
            await self._refresh_one(
                provider_name, timeout_seconds=timeout_seconds, max_attempts=max_attempts
            )
        except DiscoveryError as exc:
            log.warning(
                "registry.refresh.failed",
                provider=provider_name,
                error=str(exc),
            )
        except Exception as exc:
            log.exception(
                "registry.refresh.unexpected_failure",
                provider=provider_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def _refresh_loop(self, provider_name: str) -> None:
        """Per-provider background loop; sleeps then refreshes."""
        interval = self._interval_for(provider_name)
        cfg = self._config.providers[provider_name]
        if cfg.discovery == "noop" or cfg.discovery is None:
            # noop providers process their manual entries once, then idle.
            await self._merge_manual_only(provider_name)
            return
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=interval)
                return  # stopped
            except TimeoutError:
                pass
            await self._refresh_one_safe(
                provider_name,
                timeout_seconds=self._registry_settings.discovery_timeout_seconds,
                max_attempts=self._registry_settings.discovery_max_attempts,
            )

    async def _refresh_one(
        self,
        provider_name: str,
        *,
        timeout_seconds: int,
        max_attempts: int,
    ) -> None:
        cfg = self._config.providers[provider_name]
        adapter = self._adapter_factory(cfg)
        registry_telemetry.record_refresh_attempt(provider_name)
        started = time.perf_counter()
        try:
            result = await self._discover_with_retry(
                provider_name,
                cfg,
                adapter,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
            fresh_entries = merge_provider(provider_name, cfg, result, self._litellm_lookup)
            diff = await self._apply(provider_name, fresh_entries)
        except DiscoveryError as exc:
            registry_telemetry.record_refresh_failure(
                provider_name,
                duration_seconds=time.perf_counter() - started,
                error=exc,
            )
            raise
        registry_telemetry.record_refresh_success(
            provider_name,
            duration_seconds=time.perf_counter() - started,
            total=diff.total,
            added=diff.added,
            deprecated=diff.deprecated,
            pruned=diff.pruned,
        )

    async def _discover_with_retry(
        self,
        provider_name: str,
        cfg: ProviderConfig,
        adapter: DiscoveryAdapter,
        *,
        timeout_seconds: int,
        max_attempts: int,
    ) -> DiscoveryResult:
        @backoff.on_exception(
            backoff.expo,
            DiscoveryError,
            max_tries=max_attempts,
            jitter=backoff.full_jitter,
            logger=None,
        )
        async def _attempt() -> DiscoveryResult:
            async with self._client_factory(timeout_seconds) as client:
                return await adapter.discover(provider_name, cfg, client)

        return await _attempt()

    async def _merge_manual_only(self, provider_name: str) -> None:
        """One-shot merge for noop providers: process overrides only."""
        cfg = self._config.providers[provider_name]
        fresh = merge_provider(provider_name, cfg, DiscoveryResult(), self._litellm_lookup)
        await self._apply(provider_name, fresh)

    async def _apply(
        self, provider_name: str, fresh_entries: Mapping[str, ModelEntry]
    ) -> ProviderDiff:
        """Atomic state swap + persist; returns per-provider delta counts."""
        async with self._lock:
            now = self._clock()
            grace = self._registry_settings.deprecation_grace_seconds
            prev_entries = self._state.entries
            next_entries = apply_deprecation(
                provider=provider_name,
                prev_entries=prev_entries,
                fresh_entries=fresh_entries,
                now=now,
                grace_seconds=grace,
            )
            diff = diff_provider(provider_name, prev_entries, next_entries)
            next_refreshed = dict(self._state.refreshed_at)
            next_refreshed[provider_name] = now
            self._state = RegistryState(entries=next_entries, refreshed_at=next_refreshed)
            save_state(self._state, self._models_path)
        return diff
