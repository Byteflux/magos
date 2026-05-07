"""``TracingGateway``: decorator that opens an OTel span per dispatch.

Wraps any :class:`Gateway`. Always wired in the composition root; the
OTel tracer is a no-op until ``configure_tracing`` runs (gated by
``MAGOS_OTEL_ENABLED=1``).
"""

from __future__ import annotations

from typing import Any

from fastapi import Response
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from magos.dispatch import CompletionFn
from magos.dispatch.gateway.base import Gateway
from magos.routing import RouteDecision

_tracer = trace.get_tracer("magos.gateway")


class TracingGateway(Gateway):
    """Decorator: open a span around ``inner.dispatch`` with target attributes."""

    def __init__(self, inner: Gateway) -> None:
        self._inner = inner

    async def dispatch(
        self,
        decision: RouteDecision,
        *,
        completion: CompletionFn,
    ) -> Response | StreamingResponse | dict[str, Any]:
        target = decision.target
        with _tracer.start_as_current_span(
            "gateway.dispatch",
            attributes={
                "magos.gateway": target.gateway,
                "magos.provider": target.provider,
                "magos.endpoint": decision.request.endpoint,
                "magos.dispatch_model": decision.dispatch_model,
            },
        ):
            return await self._inner.dispatch(decision, completion=completion)
