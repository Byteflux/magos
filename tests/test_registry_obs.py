"""Observability tests: metrics + structlog events for the refresher."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricsData,
)

from magos.registry import obs as registry_obs
from magos.registry.discovery.base import (
    DiscoveredModel,
    DiscoveryError,
    DiscoveryResult,
)
from magos.registry.refresher import Refresher
from magos.registry.schema import ProviderConfig, RegistrySettings, RegistryYaml

# OTel only allows one ``set_meter_provider`` call; subsequent calls warn
# and are ignored. We install a module-level provider with a single reader
# at import time and clear the reader's snapshot between tests by reading
# (and discarding) the metrics. The ``reset_for_tests`` helper clears the
# obs module's gauge snapshot so previous tests don't pollute it.
_test_reader = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[_test_reader]))


@pytest.fixture
def metric_reader() -> Iterator[InMemoryMetricReader]:
    registry_obs.reset_for_tests()
    _test_reader.get_metrics_data()  # drain prior snapshot
    yield _test_reader


def _no_litellm(model: str) -> dict[str, Any]:
    raise ValueError("not in litellm registry")


def _noop_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _r: httpx.Response(200)))


class _StaticAdapter:
    name = "static"
    default_base_url: str | None = None

    def __init__(self, models: tuple[DiscoveredModel, ...]) -> None:
        self._models = models

    async def discover(
        self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
    ) -> DiscoveryResult:
        return DiscoveryResult(models=self._models)


class _FailingAdapter:
    name = "failing"
    default_base_url: str | None = None

    async def discover(
        self, provider_name: str, config: ProviderConfig, client: httpx.AsyncClient
    ) -> DiscoveryResult:
        raise DiscoveryError("synthetic failure")


def _config(**registry_overrides: Any) -> RegistryYaml:
    cfg = ProviderConfig.model_validate({"discovery": "openrouter"})
    settings = (
        RegistrySettings.model_validate(registry_overrides)
        if registry_overrides
        else RegistrySettings()
    )
    return RegistryYaml(providers={"openrouter": cfg}, registry=settings)


def _metric_points(data: MetricsData | None, name: str) -> list[Any]:
    """Flatten ``MetricsData`` into a list of data points for ``name``."""
    points: list[Any] = []
    if data is None:
        return points
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


def test_successful_refresh_emits_metrics_and_event(
    tmp_path: Path, metric_reader: InMemoryMetricReader
) -> None:
    cfg = _config(boot_discovery_max_attempts=1)
    adapter = _StaticAdapter(
        (
            DiscoveredModel(raw_id="a", litellm_id="openrouter/a"),
            DiscoveredModel(raw_id="b", litellm_id="openrouter/b"),
        )
    )
    refresher = Refresher(
        cfg,
        tmp_path / "models.json",
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=lambda: datetime(2026, 5, 2, tzinfo=UTC),
    )

    with structlog.testing.capture_logs() as captured:

        async def run() -> None:
            await refresher.start()
            await refresher.stop()

        asyncio.run(run())

    success_events = [e for e in captured if e.get("event") == "registry.refresh.success"]
    assert len(success_events) == 1
    assert success_events[0]["provider"] == "openrouter"
    assert success_events[0]["added"] == 2
    assert success_events[0]["total"] == 2

    data = metric_reader.get_metrics_data()
    success_points = _metric_points(data, "magos.registry.refresh.total")
    success_attrs = [dict(p.attributes) for p in success_points]
    assert any(a == {"provider": "openrouter", "status": "success"} for a in success_attrs)
    assert any(a == {"provider": "openrouter", "status": "attempt"} for a in success_attrs)

    added_points = _metric_points(data, "magos.registry.models.added")
    assert sum(p.value for p in added_points) == 2

    gauge_points = _metric_points(data, "magos.registry.models.total")
    assert any(
        p.value == 2 and dict(p.attributes) == {"provider": "openrouter"} for p in gauge_points
    )


def test_failed_refresh_emits_failure_metric_and_warning(
    tmp_path: Path, metric_reader: InMemoryMetricReader
) -> None:
    cfg = _config(boot_discovery_max_attempts=1)
    refresher = Refresher(
        cfg,
        tmp_path / "models.json",
        adapter_factory=lambda _c: _FailingAdapter(),
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=lambda: datetime(2026, 5, 2, tzinfo=UTC),
    )

    with structlog.testing.capture_logs() as captured:

        async def run() -> None:
            await refresher.start()
            await refresher.stop()

        asyncio.run(run())

    failure_events = [e for e in captured if e.get("event") == "registry.refresh.failure"]
    assert len(failure_events) == 1
    assert failure_events[0]["error_type"] == "DiscoveryError"

    data = metric_reader.get_metrics_data()
    failure_points = _metric_points(data, "magos.registry.refresh.failures")
    assert any(
        dict(p.attributes) == {"provider": "openrouter", "error_type": "DiscoveryError"}
        for p in failure_points
    )


def test_deprecated_and_pruned_counts_propagate(
    tmp_path: Path, metric_reader: InMemoryMetricReader
) -> None:
    boot_time = datetime(2026, 5, 2, tzinfo=UTC)
    refresh_time = boot_time + timedelta(hours=1)
    times = [boot_time, refresh_time]

    def clock() -> datetime:
        return times.pop(0) if len(times) > 1 else times[-1]

    class _Shrinking:
        name = "shrinking"
        default_base_url: str | None = None
        call = 0

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

    cfg = _config(boot_discovery_max_attempts=1)
    adapter = _Shrinking()
    refresher = Refresher(
        cfg,
        tmp_path / "models.json",
        adapter_factory=lambda _c: adapter,
        client_factory=_noop_client,
        litellm_lookup=_no_litellm,
        clock=clock,
    )

    async def run() -> None:
        await refresher.start()
        await refresher.refresh("openrouter")
        await refresher.stop()

    asyncio.run(run())

    data = metric_reader.get_metrics_data()
    deprecated_points = _metric_points(data, "magos.registry.models.deprecated")
    assert sum(p.value for p in deprecated_points) == 1
