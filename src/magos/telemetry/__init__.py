"""Telemetry: structured logs, OpenTelemetry tracing, and metrics.

Public surface re-exports the entry points; submodules own behavior:

- :mod:`magos.telemetry.logging`: :func:`configure_logging`, :func:`get_logger`
- :mod:`magos.telemetry.tracing`: :func:`configure_tracing`, :func:`traced`
- :mod:`magos.telemetry.metrics`: Prometheus exporter + OTel meter provider
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
