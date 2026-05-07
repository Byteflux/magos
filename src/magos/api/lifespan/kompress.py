"""Kompress lifecycle: backend selection + background weight preload.

Both pieces touch Headroom internals (``_is_onnx_available``,
``_load_kompress``); leading-underscore symbols may be renamed
upstream. Imports are guarded so a missing/incompatible Headroom
falls back to lazy load without breaking startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import cast

from fastapi import FastAPI

from magos.config.settings import MagosSettings
from magos.routing import RoutingConfig, config_uses_compress
from magos.telemetry import get_logger

log = get_logger("magos.api.lifespan")


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


class KompressBackendOverride:
    """Force kompress to the PyTorch backend when configured.

    ``MAGOS_KOMPRESS_BACKEND=pytorch`` stubs ``_is_onnx_available`` so
    Headroom's loader takes the PyTorch branch on first compress call.
    No shutdown action needed.
    """

    name = "kompress_backend_override"

    async def start(self, app: FastAPI) -> None:
        settings = cast(MagosSettings, app.state.settings)
        if settings.kompress_backend == "pytorch":
            _force_kompress_pytorch()

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
        settings = cast(MagosSettings, app.state.settings)
        if config_uses_compress(cfg) and settings.kompress_preload:
            self._task = asyncio.create_task(
                _preload_kompress_model(), name="magos.kompress.preload"
            )

    async def stop(self, app: FastAPI) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
