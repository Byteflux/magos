"""OpenTelemetry tracer setup and the ``traced`` decorator.

Spans only ship when ``MAGOS_OTEL_ENABLED=1`` (or ``enabled=True``);
until ``configure_tracing`` runs, ``traced`` wraps calls in OTel's
no-op tracer.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

P = ParamSpec("P")
R = TypeVar("R")

_TRACER_NAME = "magos"


def configure_tracing(
    *,
    service_name: str = "magos",
    endpoint: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Wire OTel TracerProvider + OTLP exporter.

    When ``enabled`` is ``None``, falls back to ``MAGOS_OTEL_ENABLED=1``.
    """
    if enabled is None:
        enabled = os.environ.get("MAGOS_OTEL_ENABLED", "0") == "1"
    if not enabled:
        return
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint is not None else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def traced(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a function in an OTel span. No-op until configure_tracing runs."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
