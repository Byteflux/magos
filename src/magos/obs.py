"""Observability scaffolding: structlog logging + OpenTelemetry tracing.

Both subsystems are safe to import unconditionally. ``configure_logging`` and
``configure_tracing`` set up the SDKs; until they run, ``get_logger`` falls
back to structlog defaults and ``traced`` wraps calls in OTel's no-op tracer.

Tracing only ships spans when ``MAGOS_OTEL_ENABLED=1``. Logging always emits;
``MAGOS_LOG_JSON=1`` flips the renderer from console to JSON.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
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
    """Configure structlog and bridge stdlib logging through it.

    Anything emitted via ``logging`` (uvicorn startup, access logs, third-party
    libraries) is rendered with the same processors as native structlog calls,
    so the operator sees one consistent stream.
    """
    use_json = json if json is not None else os.environ.get("MAGOS_LOG_JSON", "0") == "1"
    if use_json:
        renderer: Any = structlog.processors.JSONRenderer()
        timestamp_fmt = "iso"
    else:
        # Auto-color when stderr is a TTY; allow MAGOS_LOG_COLOR=0/1 to override.
        color_env = os.environ.get("MAGOS_LOG_COLOR")
        colors = (color_env == "1") if color_env is not None else sys.stderr.isatty()
        renderer = structlog.dev.ConsoleRenderer(
            colors=colors,
            force_colors=colors,
            pad_event_to=0,
        )
        timestamp_fmt = "%H:%M:%S"
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt=timestamp_fmt, utc=False),
    ]
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn ships its own LOGGING_CONFIG; we pass log_config=None to uvicorn
    # so its loggers default to propagate=True. Reset any prior handlers
    # (e.g. from a previous configure_logging call in tests) to be sure.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # Silence transformers' weight-load report (emitted when Kompress loads
    # ModernBERT without the LM head). Env var is read at transformers import;
    # the module call covers the case where it's already imported.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        with contextlib.suppress(AttributeError):
            transformers.logging.set_verbosity_error()


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
