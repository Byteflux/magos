"""FastAPI lifespan. See ``docs/architecture/startup.md`` for phase order."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Protocol, cast

from fastapi import FastAPI

from magos import __version__
from magos.config.settings import MagosSettings
from magos.registry.refresher import Refresher
from magos.routing import Compress, RoutingConfig
from magos.telemetry import get_logger
from magos.telemetry.metrics import configure_meter_provider

log = get_logger("magos.ingress.http.lifespan")


# ---------------------------------------------------------------------------
# Helpers (used by components below)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LifespanComponent protocol
# ---------------------------------------------------------------------------


class LifespanComponent(Protocol):
    """Each component owns one startup/shutdown phase of the lifespan."""

    name: str

    async def start(self, app: FastAPI) -> None: ...

    async def stop(self, app: FastAPI) -> None: ...


# ---------------------------------------------------------------------------
# Concrete components
# ---------------------------------------------------------------------------


class KompressBackendOverride:
    """Force kompress to the PyTorch backend when configured.

    ``MAGOS_KOMPRESS_BACKEND=pytorch`` stubs ``_is_onnx_available`` so
    Headroom's loader takes the PyTorch branch on first compress call.
    No shutdown action needed.
    """

    name = "kompress_backend_override"

    async def start(self, app: FastAPI) -> None:
        settings = MagosSettings()
        if settings.kompress_backend == "pytorch":
            _force_kompress_pytorch()

    async def stop(self, app: FastAPI) -> None:
        pass


class MetricsMeter:
    """Install the OTel meter provider when metrics are enabled."""

    name = "metrics_meter"

    async def start(self, app: FastAPI) -> None:
        settings = MagosSettings()
        if settings.metrics_enabled:
            configure_meter_provider()

    async def stop(self, app: FastAPI) -> None:
        pass


class HeadroomWarmup:
    """Warm the Headroom compress pipeline when any rule uses Compress.

    Failures are non-fatal: compression is best-effort and a broken
    pipeline init must not prevent the server from starting.
    """

    name = "headroom_warmup"

    async def start(self, app: FastAPI) -> None:
        cfg = cast(RoutingConfig, app.state.routing)
        if not _config_uses_compress(cfg):
            return
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

    async def stop(self, app: FastAPI) -> None:
        pass


class KompressPreload:
    """Schedule background kompress weight preload; cancel on shutdown.

    The preload runs off the event loop (``asyncio.to_thread``) so it
    does not block request handling. ``_load_kompress`` is a thread-locked
    singleton, so racing requests reuse the cached model once it resolves.
    """

    name = "kompress_preload"

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    async def start(self, app: FastAPI) -> None:
        cfg = cast(RoutingConfig, app.state.routing)
        settings = MagosSettings()
        if _config_uses_compress(cfg) and settings.kompress_preload:
            self._task = asyncio.create_task(
                _preload_kompress_model(), name="magos.kompress.preload"
            )

    async def stop(self, app: FastAPI) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


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


# ---------------------------------------------------------------------------
# Lifespan runner
# ---------------------------------------------------------------------------

# Startup order: kompress backend env → metrics → headroom warmup →
#                kompress preload (background) → refresher start.
# Shutdown order (AsyncExitStack LIFO): refresher stop → preload cancel →
#                metrics (no-op) → backend override (no-op).
_COMPONENTS: list[LifespanComponent] = [
    KompressBackendOverride(),
    MetricsMeter(),
    HeadroomWarmup(),
    KompressPreload(),
    RegistryRefresher(),
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = cast(RoutingConfig, app.state.routing)
    settings = MagosSettings()

    async with AsyncExitStack() as stack:
        for component in _COMPONENTS:
            await component.start(app)
            stack.push_async_callback(component.stop, app)

        log.info(
            "server.ready",
            version=__version__,
            rules=len(cfg.rules),
            metrics=settings.metrics_enabled,
        )
        yield
        log.info("server.shutting_down")
