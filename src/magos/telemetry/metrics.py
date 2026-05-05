"""Prometheus exporter wiring for OTel meters.

Two seams:

- :func:`configure_meter_provider` installs a global OTel ``MeterProvider``
  with the Prometheus exporter as its reader. Called once at startup
  when ``MAGOS_METRICS_ENABLED=1``.
- :func:`mount_metrics_endpoint` exposes the resulting metrics under
  ``GET /metrics`` on a FastAPI app.

``prometheus_client.start_http_server`` is intentionally avoided;
exposing through the FastAPI mount keeps the server bound to one port
for everything (HTTP API + admin + metrics).
"""

from __future__ import annotations

from fastapi import FastAPI, Response

from magos.telemetry.logging import get_logger

log = get_logger("magos.telemetry.metrics")


def configure_meter_provider() -> None:
    """Install a global OTel MeterProvider with the Prometheus exporter.

    Idempotent in practice: ``set_meter_provider`` only honors the first
    real provider per process, so re-invocation logs a warning and is a
    no-op.
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
    """Expose Prometheus-format metrics at ``GET /metrics``.

    ``prometheus_client``'s default ``REGISTRY`` is what the OTel
    ``PrometheusMetricReader`` writes into, so generating the text export
    here returns whatever the OTel meters have produced.
    """
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
