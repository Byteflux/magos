"""Prometheus exporter wiring for OTel meters.

``configure_meter_provider`` installs the global ``MeterProvider`` with
the Prometheus reader; ``mount_metrics_endpoint`` exposes the result at
``GET /metrics``. Mounting (rather than ``start_http_server``) keeps
HTTP API + admin + metrics on one port.
"""

from __future__ import annotations

from fastapi import FastAPI, Response

from magos.telemetry.logging import get_logger

log = get_logger("magos.telemetry.metrics")


def configure_meter_provider() -> None:
    """Install a global OTel MeterProvider with the Prometheus exporter.

    Idempotent: ``set_meter_provider`` only honors the first real provider per process.
    """
    try:
        from opentelemetry import metrics  # noqa: PLC0415
        from opentelemetry.exporter.prometheus import PrometheusMetricReader  # noqa: PLC0415
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    except ImportError as exc:
        log.warning(
            "metrics.exporter_unavailable",
            error=str(exc),
            hint="install opentelemetry-exporter-prometheus to enable /metrics",
        )
        return

    reader = PrometheusMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    log.info("metrics.provider_configured", exporter="prometheus")


def mount_metrics_endpoint(app: FastAPI) -> None:
    """Expose Prometheus-format metrics at ``GET /metrics``."""
    try:
        from prometheus_client import (  # noqa: PLC0415
            CONTENT_TYPE_LATEST,
            REGISTRY,
            generate_latest,
        )
    except ImportError:
        log.warning("metrics.endpoint_skipped", reason="prometheus_client missing")
        return

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
