"""FastAPI lifespan. See ``docs/architecture/startup.md`` for phase order."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI

from magos import __version__
from magos.config.settings import MagosSettings
from magos.registry.refresher import Refresher
from magos.routing import Compress, RoutingConfig
from magos.telemetry import get_logger
from magos.telemetry.metrics import configure_meter_provider

log = get_logger("magos.ingress.http.lifespan")


def _config_uses_compress(cfg: RoutingConfig) -> bool:
    if any(isinstance(rw, Compress) for rw in cfg.pre_rewrites):
        return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.rewrites)


def _force_kompress_pytorch() -> None:
    """Force kompress to the PyTorch branch by stubbing
    ``_is_onnx_available``. No-op if Headroom isn't importable."""
    try:
        from headroom.transforms import kompress_compressor  # noqa: PLC0415
    except Exception as exc:
        log.warning(
            "compress.kompress_force_pytorch_skipped",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return

    kompress_compressor._is_onnx_available = lambda: False
    log.info("compress.kompress_backend_forced", backend="pytorch")


async def _preload_kompress_model() -> None:
    """Warm kompress weights off the event loop. ``_load_kompress`` is a
    thread-locked singleton, so racing requests reuse the cached model.
    Leading-underscore symbol may be renamed upstream; ImportError falls
    back to lazy load."""
    try:
        from headroom.transforms.kompress_compressor import (  # noqa: PLC0415
            HF_MODEL_ID,
            _load_kompress,
        )
    except ImportError as exc:
        log.warning("compress.kompress_preload_unavailable", error=str(exc))
        return
    log.info("compress.kompress_preload_started", model=HF_MODEL_ID)
    started = time.perf_counter()
    try:
        await asyncio.to_thread(_load_kompress, HF_MODEL_ID, "auto")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log.info("compress.kompress_warmed", model=HF_MODEL_ID, elapsed_ms=elapsed_ms)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log.warning(
            "compress.kompress_warm_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=elapsed_ms,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = MagosSettings()
    if settings.kompress_backend == "pytorch":
        _force_kompress_pytorch()
    if settings.metrics_enabled:
        configure_meter_provider()

    cfg = cast(RoutingConfig, app.state.routing)
    preload_task: asyncio.Task[None] | None = None
    if _config_uses_compress(cfg):
        try:
            from headroom.compress import _get_pipeline  # noqa: PLC0415

            _get_pipeline()
            log.info("compress.pipeline_warmed")
        except Exception as exc:
            log.warning(
                "compress.pipeline_warm_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        if settings.kompress_preload:
            preload_task = asyncio.create_task(
                _preload_kompress_model(), name="magos.kompress.preload"
            )

    refresher: Refresher | None = cast(Refresher | None, app.state.refresher)
    if refresher is not None:
        await refresher.start()
        log.info("registry.refresher.started", providers=list(refresher._config.providers))
    log.info(
        "server.ready",
        version=__version__,
        rules=len(cfg.rules),
        metrics=settings.metrics_enabled,
    )
    try:
        yield
    finally:
        log.info("server.shutting_down")
        if preload_task is not None and not preload_task.done():
            preload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await preload_task
        if refresher is not None:
            await refresher.stop()
            log.info("registry.refresher.stopped")
