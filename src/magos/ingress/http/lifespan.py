"""FastAPI lifespan: warm Headroom, apply Kompress override, start registry.

Phases (in order, each gated on its respective config):

1. **Kompress backend monkey-patch**: when
   ``MAGOS_KOMPRESS_BACKEND=pytorch``, replace Headroom's
   ``_is_onnx_available`` with a False-stub so the loader picks the
   PyTorch path. See :func:`_force_kompress_pytorch`.
2. **OTel meter provider**: when ``MAGOS_METRICS_ENABLED=1``, install
   the global ``MeterProvider`` with the Prometheus exporter via
   :func:`magos.telemetry.metrics.configure_meter_provider`.
3. **Headroom pipeline warmup**: when any rewrite is a ``Compress``,
   trigger Headroom's lazy thread-locked singleton init so the first
   user request doesn't pay multi-second latency.
4. **Kompress preload background task**: when (3) ran AND
   ``MAGOS_KOMPRESS_PRELOAD=1``, kick off
   :func:`_preload_kompress_model` so the HF download happens off the
   request path. Cancelled on shutdown.
5. **Refresher startup**: if ``providers:`` is non-empty in yaml,
   ``await refresher.start()``. Boot discovery uses tighter timeouts
   than background refresh so unrelated providers can come up fast.

Shutdown reverses 4 + 5: cancel the preload task, stop the Refresher.
"""

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
    """True iff any rewrite (pre or per-rule) is a Compress."""
    if any(isinstance(rw, Compress) for rw in cfg.pre_rewrites):
        return True
    return any(isinstance(rw, Compress) for rule in cfg.rules for rw in rule.rewrites)


def _force_kompress_pytorch() -> None:
    """Make Headroom's Kompress loader skip the ONNX path.

    Headroom's ``_load_kompress`` checks ``_is_onnx_available()`` from the
    module namespace at call time and prefers ONNX when both backends are
    installed. Replacing that name with a False-returning stub flips the
    loader to the PyTorch branch (``_load_kompress_pytorch``), which
    auto-selects CUDA/MPS/CPU via ``device='auto'``. No Headroom patch
    needed: Python late-binding does the work.

    Silently no-ops if Kompress isn't importable (no compress rules, or
    deps missing).
    """
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
    """Warm Kompress model weights off the event loop.

    Headroom's ``_load_kompress`` is a thread-locked, double-checked
    singleton populator (see ``_kompress_cache``); a request that races
    in via ``compress()`` blocks on the same lock and reuses the cached
    model. The leading underscore is a stability risk: a Headroom
    version bump may rename it, so ImportError falls back to lazy load.
    """
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
    """Lifespan context manager passed to the FastAPI constructor."""
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
