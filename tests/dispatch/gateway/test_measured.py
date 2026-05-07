"""``MeasuredGateway``: counter increments and duration histogram per dispatch."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, MetricsData

import magos.dispatch.gateway.measured as measured_mod
from magos.dispatch.gateway import (
    CountTokensGateway,
    MeasuredGateway,
    PassthroughGateway,
    RoutedGateway,
    TranslateGateway,
)
from magos.routing import RoutingConfig
from magos.routing.decision import RouteDecision
from magos.routing.engine import route
from magos.routing.request import RoutedRequest


def _data_points(data: MetricsData | None, name: str) -> list[Any]:
    points: list[Any] = []
    if data is None:
        return points
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    points.extend(metric.data.data_points)
    return points


def _decision() -> RouteDecision:
    req = RoutedRequest(
        endpoint="/v1/chat/completions",
        headers={},
        body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        raw_body=b"",
    )
    cfg = RoutingConfig.model_validate(
        {
            "rules": [
                {
                    "match": {"endpoint": {"literal": "/v1/chat/completions"}},
                    "target": {"provider": "openai", "gateway": "translate"},
                }
            ]
        }
    )
    decision = route(req, cfg)
    assert isinstance(decision, RouteDecision)
    return decision


def _inner_gateway() -> RoutedGateway:
    return RoutedGateway(
        passthrough=PassthroughGateway(),
        translate=TranslateGateway(),
        count_tokens=CountTokensGateway(),
    )


@pytest.fixture
def patched_meter() -> Any:
    """Install a local InMemoryMetricReader and patch the module-level counters.

    This avoids calling ``metrics.set_meter_provider`` at module level,
    which would clobber the global provider used by ``tests/registry/test_telemetry.py``.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    local_meter = provider.get_meter("magos.gateway")

    original_counter = measured_mod._dispatches_total
    original_histogram = measured_mod._duration_ms

    measured_mod._dispatches_total = local_meter.create_counter(
        "magos.gateway.dispatches",
        description="Gateway dispatches, grouped by gateway and outcome",
    )
    measured_mod._duration_ms = local_meter.create_histogram(
        "magos.gateway.duration_ms",
        description="Gateway dispatch duration in milliseconds",
        unit="ms",
    )
    yield reader
    measured_mod._dispatches_total = original_counter
    measured_mod._duration_ms = original_histogram


def test_measured_gateway_increments_counter_on_success(
    patched_meter: InMemoryMetricReader,
) -> None:
    """Successful dispatch increments dispatches counter with outcome=ok."""
    gateway = MeasuredGateway(_inner_gateway())
    decision = _decision()

    async def fake_completion(**_: Any) -> Any:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    asyncio.run(gateway.dispatch(decision, completion=fake_completion))

    data = patched_meter.get_metrics_data()
    points = _data_points(data, "magos.gateway.dispatches")
    assert any(dict(p.attributes).get("outcome") == "ok" for p in points), (
        f"Expected outcome=ok in {[dict(p.attributes) for p in points]}"
    )


def test_measured_gateway_increments_counter_on_error(patched_meter: InMemoryMetricReader) -> None:
    """Failed dispatch (inner raises) increments counter with outcome=error and re-raises."""

    class _BoomGateway(RoutedGateway):
        async def dispatch(self, decision: RouteDecision, *, completion: Any) -> Any:
            raise RuntimeError("inner boom")

    gateway = MeasuredGateway(
        _BoomGateway(
            passthrough=PassthroughGateway(),
            translate=TranslateGateway(),
            count_tokens=CountTokensGateway(),
        )
    )
    decision = _decision()

    async def fake_completion(**_: Any) -> Any:
        return {}

    with pytest.raises(RuntimeError, match="inner boom"):
        asyncio.run(gateway.dispatch(decision, completion=fake_completion))

    data = patched_meter.get_metrics_data()
    points = _data_points(data, "magos.gateway.dispatches")
    assert any(dict(p.attributes).get("outcome") == "error" for p in points), (
        f"Expected outcome=error in {[dict(p.attributes) for p in points]}"
    )


def test_measured_gateway_records_duration_histogram(patched_meter: InMemoryMetricReader) -> None:
    """Duration histogram records a non-negative value on success."""
    gateway = MeasuredGateway(_inner_gateway())
    decision = _decision()

    async def fake_completion(**_: Any) -> Any:
        return {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    asyncio.run(gateway.dispatch(decision, completion=fake_completion))

    data = patched_meter.get_metrics_data()
    points = _data_points(data, "magos.gateway.duration_ms")
    assert points, "Expected at least one duration_ms data point"
    assert all(p.sum >= 0.0 for p in points), "Duration must be non-negative"
