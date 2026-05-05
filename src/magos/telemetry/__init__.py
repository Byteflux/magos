"""Telemetry: structured logs, OpenTelemetry tracing, and metrics."""

from __future__ import annotations

from magos.telemetry.logging import configure_logging, get_logger
from magos.telemetry.tracing import configure_tracing, traced

__all__ = [
    "configure_logging",
    "configure_tracing",
    "get_logger",
    "traced",
]
