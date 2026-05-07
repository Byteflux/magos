"""``MeasuredRouter``: decorator that emits an OTel counter per routing decision.

Wraps any :class:`Router`. Wired by the composition root in Phase F when
``cfg.settings.metrics_enabled``.
"""

from __future__ import annotations

from opentelemetry import metrics

from magos.routing.decision import RouteDecision
from magos.routing.engine.base import Router
from magos.routing.errors import RouteError
from magos.routing.request import RoutedRequest

_meter = metrics.get_meter("magos.router")
_decisions_total = _meter.create_counter(
    "magos.router.decisions",
    description="Routing decisions emitted, grouped by outcome",
)


class MeasuredRouter(Router):
    """Decorator: count each ``inner.route(req)`` outcome."""

    def __init__(self, inner: Router) -> None:
        self._inner = inner

    def route(self, req: RoutedRequest) -> RouteDecision | RouteError:
        outcome = self._inner.route(req)
        if isinstance(outcome, RouteError):
            _decisions_total.add(1, {"kind": "error", "code": outcome.code})
        else:
            _decisions_total.add(1, {"kind": "ok"})
        return outcome
