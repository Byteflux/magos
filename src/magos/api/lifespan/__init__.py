"""FastAPI lifespan. See ``docs/architecture/startup.md`` for phase order.

Components split across siblings:

- :mod:`kompress` — kompress backend override + background weight
  preload (the only components that require Headroom imports).
- :mod:`components` — generic phases: metrics meter, magos.compression
  warmup, registry refresher.

The ``LifespanComponent`` Protocol and the runner live here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Protocol, cast

from fastapi import FastAPI

from magos import __version__
from magos.config.settings import MagosSettings
from magos.routing import RoutingConfig
from magos.telemetry import get_logger

from .components import MagosCompressionWarmup, MetricsMeter, RegistryRefresher
from .kompress import KompressBackendOverride, KompressPreload

log = get_logger("magos.api.lifespan")


class LifespanComponent(Protocol):
    """Each component owns one startup/shutdown phase of the lifespan."""

    name: str

    async def start(self, app: FastAPI) -> None: ...

    async def stop(self, app: FastAPI) -> None: ...


# Startup order: kompress backend env -> metrics -> compression warmup ->
#                kompress preload (background) -> refresher start.
# Shutdown order (AsyncExitStack LIFO): refresher stop -> preload cancel ->
#                metrics (no-op) -> backend override (no-op).
_COMPONENTS: list[LifespanComponent] = [
    KompressBackendOverride(),
    MetricsMeter(),
    MagosCompressionWarmup(),
    KompressPreload(),
    RegistryRefresher(),
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = cast(RoutingConfig, app.state.routing)
    settings = cast(MagosSettings, app.state.settings)

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


__all__ = ["LifespanComponent", "lifespan"]
