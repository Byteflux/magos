"""``MeasuredGateway``: decorator that emits OTel metrics per dispatch.

Wraps any :class:`Gateway`. Wired by the composition root when
``MagosSettings.metrics_enabled`` is true.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse
from opentelemetry import metrics

from magos.dispatch import CompletionFn
from magos.routing import RouteDecision

from .base import Gateway

_meter = metrics.get_meter("magos.gateway")
_dispatches_total = _meter.create_counter(
    "magos.gateway.dispatches",
    description="Gateway dispatches, grouped by gateway and outcome",
)
_duration_ms = _meter.create_histogram(
    "magos.gateway.duration_ms",
    description="Gateway dispatch duration in milliseconds",
    unit="ms",
)


class MeasuredGateway(Gateway):
    """Decorator: count + time each ``inner.dispatch`` call."""

    def __init__(self, inner: Gateway) -> None:
        self._inner = inner

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        gateway_name = decision.target.gateway
        endpoint = decision.request.endpoint
        attrs = {"gateway": gateway_name, "endpoint": endpoint}
        start = time.perf_counter()
        try:
            result = await self._inner.dispatch(decision, completion=completion)
        except Exception:
            _dispatches_total.add(1, {**attrs, "outcome": "error"})
            _duration_ms.record((time.perf_counter() - start) * 1000.0, attrs)
            raise
        _dispatches_total.add(1, {**attrs, "outcome": "ok"})
        _duration_ms.record((time.perf_counter() - start) * 1000.0, attrs)
        return result
