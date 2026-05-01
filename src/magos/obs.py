"""Observability scaffolding: structlog logging + OpenTelemetry tracing.

Both subsystems are safe to import unconditionally. ``configure_logging`` and
``configure_tracing`` set up the SDKs; until they run, ``get_logger`` falls
back to structlog defaults and ``traced`` wraps calls in OTel's no-op tracer.

Tracing only ships spans when ``MAGOS_OTEL_ENABLED=1``. Logging always emits;
``MAGOS_LOG_JSON=1`` flips the renderer from console to JSON.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

P = ParamSpec("P")
R = TypeVar("R")

_TRACER_NAME = "magos"


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog. JSON renderer if requested or MAGOS_LOG_JSON=1."""
    use_json = json if json is not None else os.environ.get("MAGOS_LOG_JSON", "0") == "1"
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def configure_tracing(
    *,
    service_name: str = "magos",
    endpoint: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Wire OTel TracerProvider + OTLP exporter.

    When ``enabled`` is ``None``, falls back to ``MAGOS_OTEL_ENABLED=1`` for
    backward compatibility. Pass ``enabled`` explicitly from a settings object
    to keep configuration declarative.
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


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; safe before configure_logging runs."""
    return cast(
        structlog.stdlib.BoundLogger,
        structlog.get_logger(name) if name else structlog.get_logger(),
    )


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
