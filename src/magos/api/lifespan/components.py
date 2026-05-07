"""Generic lifespan phases: metrics meter, compression-pipeline warmup,
registry refresher. Kompress-specific components live in `kompress`.
"""

from __future__ import annotations

from typing import cast

from fastapi import FastAPI

from magos.config.settings import MagosSettings
from magos.registry.refresher import Refresher
from magos.routing import RoutingConfig, config_uses_compress
from magos.telemetry import get_logger
from magos.telemetry.metrics import configure_meter_provider

log = get_logger("magos.api.lifespan")


class MetricsMeter:
    """Install the OTel meter provider when metrics are enabled."""

    name = "metrics_meter"

    async def start(self, app: FastAPI) -> None:
        settings = cast(MagosSettings, app.state.settings)
        if settings.metrics_enabled:
            configure_meter_provider()

    async def stop(self, app: FastAPI) -> None:
        pass


class MagosCompressionWarmup:
    """Build pipelines for both providers and eagerly load compressors.

    The proxy uses two pipelines (Anthropic + OpenAI) sharing transform
    instances; magos's registry deduplicates the same way. Failures are
    non-fatal — compression is best-effort.
    """

    name = "compression_warmup"

    async def start(self, app: FastAPI) -> None:
        cfg = cast(RoutingConfig, app.state.routing)
        if not config_uses_compress(cfg):
            return
        try:
            from magos.compression import prebuild_from_routing  # noqa: PLC0415
        except Exception as exc:
            log.warning(
                "compress.pipeline_warm_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return

        try:
            prebuild_from_routing(cfg)
            log.info("compress.pipeline_warmed")
        except Exception as exc:
            log.warning(
                "compress.pipeline_warm_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

    async def stop(self, app: FastAPI) -> None:
        pass


class RegistryRefresher:
    """Start the registry Refresher on startup and stop it on shutdown."""

    name = "registry_refresher"

    async def start(self, app: FastAPI) -> None:
        refresher: Refresher | None = cast(Refresher | None, app.state.refresher)
        if refresher is not None:
            await refresher.start()
            log.info(
                "registry.refresher.started",
                providers=list(refresher._config.providers),
            )

    async def stop(self, app: FastAPI) -> None:
        refresher: Refresher | None = cast(Refresher | None, app.state.refresher)
        if refresher is not None:
            await refresher.stop()
            log.info("registry.refresher.stopped")
