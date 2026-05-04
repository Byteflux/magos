"""Telemetry: structured logs and OpenTelemetry tracing.

Public surface re-exports the four entry points; submodules own behavior:

- :mod:`magos.telemetry.logging` — :func:`configure_logging`, :func:`get_logger`
- :mod:`magos.telemetry.tracing` — :func:`configure_tracing`, :func:`traced`

A future :mod:`magos.telemetry.metrics` will own the Prometheus
exporter + OTel meter provider currently set up inside FastAPI's
lifespan (extracted in a later phase).
"""

from __future__ import annotations

from magos.telemetry.logging import configure_logging, get_logger
from magos.telemetry.tracing import configure_tracing, traced

__all__ = [
    "configure_logging",
    "configure_tracing",
    "get_logger",
    "traced",
]
