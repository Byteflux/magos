"""Tests for `magos.registry.refresher` lifecycle controller."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryAdapter,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.litellm_lookup import PartialEntry
from magos.registry.refresher import Refresher
from magos.registry.schema import ProviderConfig, RegistrySettings, RegistryYaml
from magos.registry.state import ModelEntry, RegistryState
from magos.registry.store import save as save_state


class _StaticAdapter:
    """Adapter that returns a preset list and counts invocations."""

    default_base_url: str | None = None

    def __init__(self, name: str, models: Iterable[DiscoveredModel]) -> None:
        self.name = name
        self._models = tuple(models)
        self.calls = 0

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        self.calls += 1
        return DiscoveryResult(models=self._models)


class _FailingAdapter:
    name = "failing"
    default_base_url: str | None = None

    def __init__(self) -> None:
        self.calls = 0

    async def discover(
        self,
        provider_name: str,
        config: ProviderConfig,
        client: httpx.AsyncClient,
    ) -> DiscoveryResult:
        self.calls += 1
        raise DiscoveryError("synthetic failure")


def _no_litellm(model: str) -> dict[str, Any]:
    raise ValueError("not in litellm registry")


def _noop_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))


def _config(providers: dict[str, ProviderConfig], **registry_overrides: Any) -> RegistryYaml:
    settings = (
        RegistrySettings.model_validate(registry_overrides)
        if registry_overrides
        else RegistrySettings()
    )
    return RegistryYaml(providers=providers, registry=settings)


def _fixed_clock(when: datetime) -> Callable[[], datetime]:
    def _now() -> datetime:
        return when

    return _now


def test_start_loads_existing_disk_state(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    entry = ModelEntry(
        provider="openrouter",
        raw_id="anthropic/claude-sonnet-4-6",
        litellm_id="openrouter/anthropic/claude-sonnet-4-6",
    )
    save_state(
        RegistryState(
            entries={entry.namespaced_id: entry},
            refreshed_at={"openrouter": datetime(2026, 5, 1, tzinfo=UTC)},
        ),
        target,
    )

    cfg = _config({"openrouter": ProviderConfig.model_validate({"discovery": "openrouter"})})
    adapter = _StaticAdapter("openrouter", [])
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
    )

    async def run() -> None:
        await refresher.start()
        try:
            assert "openrouter/anthropic/claude-sonnet-4-6" in refresher.state.entries
            # Disk state present means boot discovery is skipped.
            assert adapter.calls == 0
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_boot_discovery_populates_empty_provider(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config({"openrouter": ProviderConfig.model_validate({"discovery": "openrouter"})})
    adapter = _StaticAdapter(
        "openrouter",
        [
            DiscoveredModel(
                raw_id="anthropic/claude-sonnet-4-6",
                litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                partial=PartialEntry(context_size=200000),
            )
        ],
    )
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        await refresher.start()
        try:
            assert adapter.calls == 1
            entry = refresher.state.entries["openrouter/anthropic/claude-sonnet-4-6"]
            assert entry.context_size == 200000
            assert entry.litellm_id == "openrouter/anthropic/claude-sonnet-4-6"
            assert target.exists(), "should persist after refresh"
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_boot_discovery_failure_leaves_provider_empty(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config(
        {
            "openrouter": ProviderConfig.model_validate({"discovery": "openrouter"}),
            "anthropic": ProviderConfig.model_validate(
                {"discovery": "anthropic", "api_key_env": "ANTHROPIC_KEY"}
            ),
        },
        boot_discovery_max_attempts=1,
    )
    failing = _FailingAdapter()
    healthy = _StaticAdapter(
        "anthropic",
        [DiscoveredModel(raw_id="claude-sonnet-4-6", litellm_id="anthropic/claude-sonnet-4-6")],
    )

    def factory(provider_cfg: ProviderConfig) -> DiscoveryAdapter:
        if provider_cfg.discovery == "openrouter":
            return failing
        return healthy

    refresher = Refresher(
        cfg,
        target,
        adapter_factory=factory,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        await refresher.start()
        try:
            # Failing provider stays empty; healthy provider succeeds.
            assert refresher.state.by_provider.get("openrouter") in (None, frozenset())
            assert "anthropic/claude-sonnet-4-6" in refresher.state.entries
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_refresh_loop_swallows_unexpected_exception_to_keep_polling(tmp_path: Path) -> None:
    """Non-`DiscoveryError` exceptions must not escape the loop wrapper.

    asyncio Tasks store unhandled exceptions silently while a strong
    reference is held (see `Refresher._tasks`), so any exception that
    escapes `_refresh_one_safe` would kill periodic refresh forever
    without a single log line. The wrapper must catch `Exception` and
    log it so the next interval still fires.
    """
    target = tmp_path / "models.json"
    cfg = _config({"openrouter": ProviderConfig.model_validate({"discovery": "openrouter"})})

    class _BoomAdapter:
        name = "boom"
        default_base_url: str | None = None
        calls = 0

        async def discover(
            self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
        ) -> DiscoveryResult:
            self.calls += 1
            raise RuntimeError("simulated bug, not a DiscoveryError")

    adapter = _BoomAdapter()
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        # Two back-to-back wrapper calls; if the first one re-raised,
        # the second wouldn't run and adapter.calls would stay at 1.
        await refresher._refresh_one_safe("openrouter", timeout_seconds=10, max_attempts=1)
        await refresher._refresh_one_safe("openrouter", timeout_seconds=10, max_attempts=1)

    asyncio.run(run())
    assert adapter.calls == 2


def test_refresh_failure_after_success_preserves_prior_state(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config({"openrouter": ProviderConfig.model_validate({"discovery": "openrouter"})})

    class _Toggling:
        name = "toggling"
        default_base_url: str | None = None

        def __init__(self) -> None:
            self.call = 0

        async def discover(
            self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
        ) -> DiscoveryResult:
            self.call += 1
            if self.call == 1:
                return DiscoveryResult(
                    models=(
                        DiscoveredModel(
                            raw_id="anthropic/claude-sonnet-4-6",
                            litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                        ),
                    )
                )
            raise DiscoveryError("transient outage")

    adapter = _Toggling()
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        await refresher.start()
        try:
            first_state = refresher.state
            assert "openrouter/anthropic/claude-sonnet-4-6" in first_state.entries
            with pytest.raises(DiscoveryError):
                await refresher.refresh("openrouter")
            assert refresher.state is first_state, "failed refresh must not swap state"
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_manual_only_provider_registers_override_entries(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config(
        {
            "manual": ProviderConfig.model_validate(
                {
                    "litellm_provider": "openai",
                    "models": {
                        "custom-model": {"context_size": 32000, "litellm_id": "openai/custom"},
                    },
                }
            )
        }
    )
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: _StaticAdapter("noop", []),
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        await refresher.start()
        try:
            entry = refresher.state.entries.get("manual/custom-model")
            assert entry is not None
            assert entry.litellm_id == "openai/custom"
            assert entry.context_size == 32000
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_override_field_takes_precedence_over_discovery(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config(
        {
            "openrouter": ProviderConfig.model_validate(
                {
                    "discovery": "openrouter",
                    "models": {
                        "anthropic/claude-sonnet-4-6": {
                            "context_size": 1_000_000,
                        },
                    },
                }
            )
        }
    )
    adapter = _StaticAdapter(
        "openrouter",
        [
            DiscoveredModel(
                raw_id="anthropic/claude-sonnet-4-6",
                litellm_id="openrouter/anthropic/claude-sonnet-4-6",
                partial=PartialEntry(context_size=200000),
            )
        ],
    )
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=_fixed_clock(datetime(2026, 5, 2, tzinfo=UTC)),
    )

    async def run() -> None:
        await refresher.start()
        try:
            entry = refresher.state.entries["openrouter/anthropic/claude-sonnet-4-6"]
            assert entry.context_size == 1_000_000
        finally:
            await refresher.stop()

    asyncio.run(run())


def test_disappeared_model_is_marked_deprecated(tmp_path: Path) -> None:
    target = tmp_path / "models.json"
    cfg = _config({"openrouter": ProviderConfig.model_validate({"discovery": "openrouter"})})

    class _ShrinkingAdapter:
        name = "shrinking"
        default_base_url: str | None = None

        def __init__(self) -> None:
            self.call = 0

        async def discover(
            self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
        ) -> DiscoveryResult:
            self.call += 1
            if self.call == 1:
                return DiscoveryResult(
                    models=(
                        DiscoveredModel(raw_id="a", litellm_id="openrouter/a"),
                        DiscoveredModel(raw_id="b", litellm_id="openrouter/b"),
                    )
                )
            return DiscoveryResult(models=(DiscoveredModel(raw_id="a", litellm_id="openrouter/a"),))

    boot_time = datetime(2026, 5, 2, tzinfo=UTC)
    refresh_time = boot_time + timedelta(hours=1)
    times = [boot_time, refresh_time]

    def clock() -> datetime:
        return times.pop(0) if len(times) > 1 else times[-1]

    adapter = _ShrinkingAdapter()
    refresher = Refresher(
        cfg,
        target,
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=clock,
    )

    async def run() -> None:
        await refresher.start()
        try:
            await refresher.refresh("openrouter")
            entry_a = refresher.state.entries["openrouter/a"]
            entry_b = refresher.state.entries["openrouter/b"]
            assert entry_a.deprecated_at is None
            assert entry_b.deprecated_at == refresh_time
        finally:
            await refresher.stop()

    asyncio.run(run())
