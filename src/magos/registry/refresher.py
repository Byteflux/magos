"""Async lifecycle controller for the registry.

Owns the in-memory ``RegistryState`` and persists to disk after every
successful refresh. Per-provider refresh tasks run on each provider's
interval (default from ``RegistrySettings.refresh_interval``, overridable
per provider). Boot discovery uses tighter timeouts and fewer attempts
than background refresh, matching the design that boot should give up
fast (so unrelated providers can come up) while background can be patient
(prior state is preserved on failure).

The Refresher is the only writer to ``models.json``; every replacement
goes through ``_apply`` under a single ``asyncio.Lock`` so on-disk and
in-memory state never disagree. Adapters and HTTP-client factory are
injectable so tests can pass synthetic implementations.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import backoff
import httpx

from magos.obs import get_logger
from magos.registry import obs as registry_obs
from magos.registry.deprecation import apply_deprecation
from magos.registry.discovery import (
    DiscoveryAdapter,
    DiscoveryError,
    DiscoveryResult,
    adapter_for,
)
from magos.registry.litellm_lookup import GetModelInfoFn, PartialEntry, lookup
from magos.registry.merge import merge
from magos.registry.models import ModelEntry, RegistryState
from magos.registry.schema import (
    ModelOverride,
    ProviderConfig,
    RegistrySettings,
    RegistryYaml,
)
from magos.registry.store import load as load_state
from magos.registry.store import save as save_state

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
        """Aggressive parallel discovery for providers absent from disk state.

        On boot, providers that already have entries in ``models.json``
        (i.e. survived a prior run) skip discovery — they will be
        refreshed on their normal interval. Providers with zero entries
        get a one-shot discovery attempt with tight timeouts; failure
        just leaves the provider empty until the background loop catches
        up, per the "boot with empty registry for failing providers"
        decision.
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

    async def _refresh_loop(self, provider_name: str) -> None:
        """Per-provider background loop; sleeps then refreshes."""
        interval = self._interval_for(provider_name)
        cfg = self._config.providers[provider_name]
        if cfg.discovery == "noop" or cfg.discovery is None:
            # No-op providers don't refresh; their manual entries are
            # already merged by start() during boot discovery (or, if the
            # disk state had them, by the load).
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
        registry_obs.record_refresh_attempt(provider_name)
        started = time.perf_counter()
        try:
            result = await self._discover_with_retry(
                provider_name,
                cfg,
                adapter,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
            )
            fresh_entries = self._merge_provider(provider_name, cfg, result)
            diff = await self._apply(provider_name, fresh_entries)
        except DiscoveryError as exc:
            registry_obs.record_refresh_failure(
                provider_name,
                duration_seconds=time.perf_counter() - started,
                error=exc,
            )
            raise
        registry_obs.record_refresh_success(
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

    def _merge_provider(
        self,
        provider_name: str,
        cfg: ProviderConfig,
        result: DiscoveryResult,
    ) -> dict[str, ModelEntry]:
        """Build the fresh entry set for ``provider_name`` from a discovery.

        Discovery models contribute via the discovery slot; for each model
        we layer the matching ``models[raw_id]`` override on top and pull
        a litellm fallback for any field both layers leave unset.

        Manual-only entries (override keys not seen in discovery) are
        synthesized as if discovery had returned an empty PartialEntry
        for them — the override layer supplies the dispatch id and any
        fields the operator declared.
        """
        fresh: dict[str, ModelEntry] = {}
        seen_raw_ids: set[str] = set()
        for discovered in result.models:
            raw_id = discovered.raw_id
            seen_raw_ids.add(raw_id)
            override_partial = _override_to_partial(cfg.models.get(raw_id))
            litellm_fallback = self._litellm_partial(
                _effective_litellm_id(discovered.litellm_id, cfg.models.get(raw_id))
            )
            entry = merge(
                provider=provider_name,
                raw_id=raw_id,
                default_litellm_id=discovered.litellm_id,
                override=override_partial,
                discovered=discovered.partial,
                litellm_fallback=litellm_fallback,
            )
            fresh[entry.namespaced_id] = entry

        for raw_id, override in cfg.models.items():
            if raw_id in seen_raw_ids:
                continue
            default_litellm_id = override.litellm_id or _default_manual_litellm_id(
                provider_name, cfg, raw_id
            )
            override_partial = _override_to_partial(override)
            litellm_fallback = self._litellm_partial(default_litellm_id)
            entry = merge(
                provider=provider_name,
                raw_id=raw_id,
                default_litellm_id=default_litellm_id,
                override=override_partial,
                discovered=None,
                litellm_fallback=litellm_fallback,
            )
            fresh[entry.namespaced_id] = entry
        return fresh

    def _litellm_partial(self, litellm_id: str) -> PartialEntry:
        if self._litellm_lookup is None:
            return lookup(litellm_id)
        return lookup(litellm_id, get_info=self._litellm_lookup)

    async def _merge_manual_only(self, provider_name: str) -> None:
        """One-shot merge for noop providers: process overrides only."""
        cfg = self._config.providers[provider_name]
        fresh = self._merge_provider(provider_name, cfg, DiscoveryResult())
        await self._apply(provider_name, fresh)

    async def _apply(
        self, provider_name: str, fresh_entries: Mapping[str, ModelEntry]
    ) -> _RefreshDiff:
        """Atomic state swap + disk persistence for one provider's slice.

        Returns a ``_RefreshDiff`` with the per-provider counts the
        observability layer needs (added / deprecated / pruned / total).
        """
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
            diff = _diff_provider(provider_name, prev_entries, next_entries)
            next_refreshed = dict(self._state.refreshed_at)
            next_refreshed[provider_name] = now
            self._state = RegistryState(entries=next_entries, refreshed_at=next_refreshed)
            save_state(self._state, self._models_path)
        return diff


def _override_to_partial(override: ModelOverride | None) -> PartialEntry | None:
    if override is None:
        return None
    return PartialEntry(
        litellm_id=override.litellm_id,
        context_size=override.context_size,
        max_output=override.max_output,
        input_cost=override.input_cost,
        output_cost=override.output_cost,
        modalities=override.modalities,
    )


def _effective_litellm_id(default: str, override: ModelOverride | None) -> str:
    if override is None or override.litellm_id is None:
        return default
    return override.litellm_id


def _default_manual_litellm_id(provider_name: str, cfg: ProviderConfig, raw_id: str) -> str:
    """Construct a dispatch id for a manual-only entry.

    Mirrors how each adapter would have prefixed it: ``litellm_provider``
    if set, otherwise the magos provider name.
    """
    prefix = cfg.litellm_provider or provider_name
    return f"{prefix}/{raw_id}"


_AwaitableHook = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _RefreshDiff:
    """Per-provider counts surfaced from ``_apply`` for observability."""

    total: int
    added: int
    deprecated: int
    pruned: int


def _diff_provider(
    provider: str,
    prev_entries: Mapping[str, ModelEntry],
    next_entries: Mapping[str, ModelEntry],
) -> _RefreshDiff:
    """Compute per-provider deltas across one refresh cycle.

    ``added`` counts namespaced ids that were absent before; ``deprecated``
    counts entries that gained a ``deprecated_at`` mark this cycle (was
    None before, set now); ``pruned`` counts entries removed entirely.
    ``total`` is the post-refresh active count (including still-marked
    deprecated entries that haven't aged out).
    """
    prev_for_provider = {k: e for k, e in prev_entries.items() if e.provider == provider}
    next_for_provider = {k: e for k, e in next_entries.items() if e.provider == provider}
    added = sum(1 for k in next_for_provider if k not in prev_for_provider)
    deprecated = sum(
        1
        for k, entry in next_for_provider.items()
        if entry.deprecated_at is not None
        and (k not in prev_for_provider or prev_for_provider[k].deprecated_at is None)
    )
    pruned = sum(1 for k in prev_for_provider if k not in next_for_provider)
    return _RefreshDiff(
        total=len(next_for_provider),
        added=added,
        deprecated=deprecated,
        pruned=pruned,
    )
