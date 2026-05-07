"""Process orchestrator: run FastAPI and (optional) mitmproxy on one loop.

See ``docs/architecture/startup.md`` and ``docs/architecture/env-vars.md``.
"""

from __future__ import annotations

import asyncio

import uvicorn

from magos import __version__
from magos.api import build_api
from magos.config.loader import MagosConfig, load_full_config
from magos.config.schema import HttpIngressConfig, MitmIngressConfig
from magos.config.settings import MagosSettings
from magos.proxy import build_proxy
from magos.proxy.log_bridge import install_log_bridge
from magos.telemetry import get_logger

log = get_logger("magos.serve")

# uvicorn flips ``started`` only after lifespan completes (Headroom warmup,
# registry refresher init), so the mitm listener can't accept a request
# that races a half-warm app.
_FASTAPI_READY_POLL_SECONDS = 0.05


def resolve_bind(settings: MagosSettings, http_cfg: HttpIngressConfig) -> tuple[str, int]:
    """Resolve FastAPI bind host/port; env wins, empty-string env treated as unset."""
    host = settings.host or http_cfg.host
    port = settings.port if settings.port is not None else http_cfg.port
    return host, port


def resolve_mitm(settings: MagosSettings, mitm_cfg: MitmIngressConfig) -> MitmIngressConfig:
    """Merge ``MAGOS_MITM_*`` env overrides over the yaml ``ingress.mitm`` block."""
    enabled = settings.mitm_enabled if settings.mitm_enabled is not None else mitm_cfg.enabled
    host = settings.mitm_host or mitm_cfg.host
    port = settings.mitm_port if settings.mitm_port is not None else mitm_cfg.port
    intercept_hosts = (
        settings.mitm_intercept_hosts
        if settings.mitm_intercept_hosts is not None
        else mitm_cfg.intercept_hosts
    )
    return MitmIngressConfig(
        enabled=enabled,
        host=host,
        port=port,
        intercept_hosts=intercept_hosts,
    )


def serve(*, settings: MagosSettings | None = None) -> None:
    """Synchronous entrypoint."""
    settings = settings or MagosSettings()
    asyncio.run(serve_async(settings=settings))


async def serve_async(*, settings: MagosSettings) -> None:
    """Run uvicorn + (optional) mitm ingress; first to exit brings down the other."""
    cfg: MagosConfig = load_full_config(settings.config_path)
    bind_host, bind_port = resolve_bind(settings, cfg.ingress.http)
    mitm_cfg = resolve_mitm(settings, cfg.ingress.mitm)

    log.info(
        "serve.starting",
        version=__version__,
        host=bind_host,
        port=bind_port,
        ingress_enabled=mitm_cfg.enabled,
        config_path=settings.config_path,
    )

    app = build_api(routing=cfg.routing, registry=cfg.registry)
    uvi_config = uvicorn.Config(
        app,
        host=bind_host,
        port=bind_port,
        log_config=None,
        access_log=settings.access_log,
    )
    uvi_server = uvicorn.Server(uvi_config)

    fastapi_task = asyncio.create_task(uvi_server.serve(), name="magos.fastapi")

    if not mitm_cfg.enabled or not mitm_cfg.intercept_hosts:
        if mitm_cfg.enabled:
            log.warning(
                "ingress.no_intercept_hosts",
                hint="ingress.mitm.enabled is true but intercept_hosts is empty; ingress proxy not started",
            )
        await fastapi_task
        return

    # Block ingress until FastAPI lifespan finishes; uvicorn exposes no
    # awaitable event, so polling ``Server.started`` is the documented idiom.
    while not uvi_server.started and not fastapi_task.done():  # noqa: ASYNC110
        await asyncio.sleep(_FASTAPI_READY_POLL_SECONDS)
    if fastapi_task.done():
        fastapi_task.result()
        return

    install_log_bridge()
    master = build_proxy(mitm_cfg, target_host=bind_host, target_port=bind_port)
    mitm_task = asyncio.create_task(master.run(), name="magos.mitm")
    log.info(
        "ingress.started",
        listen=f"{mitm_cfg.host}:{mitm_cfg.port}",
        target=f"{bind_host}:{bind_port}",
        intercept_hosts=list(mitm_cfg.intercept_hosts),
    )

    done, pending = await asyncio.wait(
        {fastapi_task, mitm_task}, return_when=asyncio.FIRST_COMPLETED
    )
    # Use each task's own shutdown hook so listeners close cleanly before cancel.
    if mitm_task in pending:
        master.shutdown()  # type: ignore[no-untyped-call]
    if fastapi_task in pending:
        uvi_server.should_exit = True
    for task in pending:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
    for task in done:
        task.result()
